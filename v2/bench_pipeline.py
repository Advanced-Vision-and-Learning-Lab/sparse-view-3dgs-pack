# bench_pipeline.py
# Building run descriptors and executing train/render/metrics with summary append.

from __future__ import annotations
import csv
import json
import os
import re
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from bench_config import (
    deep_merge, as_namespace_string, flatten_repo_args, build_cli_args,
    ensure_dir, expand_grid
)


def build_runs(cfg: Dict[str, Any], out_root: Path, python_exe: str) -> List[Dict[str, Any]]:
    runs = []

    # Prepare grid expansions
    grid = cfg.get("grid", {})
    grid_expansions = list(expand_grid(grid))

    for scene in cfg.get("scenes", []):
        scene_name = scene["name"]
        for model in cfg.get("models", []):
            model_name = model["name"]
            ov = cfg.get("overrides", {}).get(model_name, {}).get(scene_name, {})
            base_args = deep_merge(cfg["defaults"].get("common_args", {}), model.get("args", {}))
            base_args["source_path"] = scene["source_path"]
            for grid_vars in grid_expansions:
                seed = grid_vars.get("seeds", cfg["defaults"].get("seeds", [42]))
                seed_list = seed if isinstance(seed, list) else [seed]
                for s in seed_list:
                    merged = deep_merge(base_args, ov.get("args", {}))
                    merged["seed"] = s
                    flat = flatten_repo_args(merged)

                    tag = f"{scene_name}__{model_name}__sh{flat.get('sh_degree', 'NA')}__it{flat.get('iterations', 'NA')}__seed{s}"
                    run_dir    = (out_root / tag).resolve()
                    model_path = (run_dir / "model").resolve()
                    log_dir    = (run_dir / "logs").resolve()
                    ensure_dir(model_path)
                    ensure_dir(log_dir)

                    flat["model_path"]  = str(model_path)
                    flat["source_path"] = str(Path(scene["source_path"]).resolve())

                    repo_dir = Path(model["repo_dir"]).resolve()
                    entry = (repo_dir / model.get("entry", "train.py")).resolve()
                    arg_map = model.get("arg_map", {})
                    if model.get("needs_cfg_args_file", False):
                        ns_str = as_namespace_string(flat, allowed_keys=set(arg_map.keys()))
                        with open(model_path / "cfg_args", "w") as f:
                            f.write(ns_str)
                    cli_flags = build_cli_args(arg_map, flat)
                    cmd = [python_exe, str(entry)] + cli_flags

                    runs.append({
                        "tag": tag,
                        "scene": scene_name,
                        "model": model_name,
                        "cmd": cmd,
                        "env": deep_merge(os.environ.copy(), cfg["defaults"].get("env", {})),
                        "repo_dir": str(repo_dir),
                        "log_path": str(log_dir / "train.log"),
                        "result_path": str(run_dir / "result.json"),
                        "seed": s,
                        "args_flat": flat,
                    })
    return runs


def _parse_psnr_from_log(text: str) -> float | None:
    try:
        m = re.findall(r"PSNR[:=]\s*([0-9.]+)", text)
        return float(m[-1]) if m else None
    except Exception:
        return None


def _maybe_run_render(r: Dict[str, Any], model_cfg: Dict[str, Any], python_exe: str) -> Tuple[int | None, float | None]:
    run_render = model_cfg.get("run_render", False)
    if not run_render:
        return None, None
    repo_dir = Path(r["repo_dir"])
    model_path = r["args_flat"]["model_path"]
    source_path = r["args_flat"]["source_path"]
    env = r["env"]
    render_script = model_cfg.get("render_entry", "render.py")
    render_log = Path(r["log_path"]).with_name("render.log")
    cmd_render = [python_exe, render_script, "--model_path", model_path, "--source_path", source_path, "--eval"]
    with open(render_log, "w") as logf:
        print(f"\n[Render] Running render for {r['tag']}")
        print("CMD:", " ".join(shlex.quote(x) for x in cmd_render))
        t0 = time.time()
        proc = subprocess.Popen(cmd_render, cwd=repo_dir, env=env, stdout=logf, stderr=subprocess.STDOUT)
        code = proc.wait()
        dur = time.time() - t0
        print(f"[Render] Finished {r['tag']} in {dur:.2f}s, exit={code}")
        return code, dur


def _maybe_run_metrics(r: Dict[str, Any], model_cfg: Dict[str, Any], python_exe: str) -> Tuple[int | None, float | None, Dict[str, float | None]]:
    run_metrics = model_cfg.get("run_metrics", False)
    if not run_metrics:
        return None, None, {}
    repo_dir = Path(r["repo_dir"])
    env = r["env"]
    model_path = r["args_flat"]["model_path"]
    metrics_script = model_cfg.get("metrics_entry", "metrics.py")
    metrics_log = Path(r["log_path"]).with_name("metrics.log")

    it_str = str(r["args_flat"].get("iterations", -1))
    if r["model"].lower() == "fsgs":
        cmd_metrics = [python_exe, metrics_script, "--model_path",  r["args_flat"]["model_path"],
                       "--source_path", r["args_flat"]["source_path"], "--iteration", it_str]
    else:
        cmd_metrics = [python_exe, metrics_script, "--model_paths", r["args_flat"]["model_path"]]

    with open(metrics_log, "w") as logf:
        print(f"\n[Metrics] Running metrics for {r['tag']}")
        print("CMD:", " ".join(shlex.quote(x) for x in cmd_metrics))
        t0 = time.time()
        proc = subprocess.Popen(cmd_metrics, cwd=repo_dir, env=env, stdout=logf, stderr=subprocess.STDOUT)
        code = proc.wait()
        dur = time.time() - t0
        print(f"[Metrics] Finished {r['tag']} in {dur:.2f}s, exit={code}")

    # collect metrics.json if produced
    out = {"psnr": None, "ssim": None, "lpips": None}
    try:
        method = f"ours_{r['args_flat'].get('iterations', 'NA')}"
        rp = Path(model_path) / "results.json"
        if rp.exists():
            with open(rp) as jf:
                data = json.load(jf)
            block = data.get(method) or (next(iter(data.values())) if isinstance(data, dict) and data else None)
            if isinstance(block, dict):
                if "PSNR" in block:  out["psnr"]  = float(block["PSNR"])
                if "SSIM" in block:  out["ssim"]  = float(block["SSIM"])
                if "LPIPS" in block: out["lpips"] = float(block["LPIPS"])
    except Exception:
        pass
    return code, dur, out


def execute_runs(runs: List[Dict[str, Any]], cfg: Dict[str, Any], dry: bool, python_exe: str) -> List[Dict[str, Any]]:
    summary_rows: List[Dict[str, Any]] = []
    for r in runs:
        print("\n==> Running:", r["tag"])
        print("CMD:", " ".join(shlex.quote(x) for x in r["cmd"]))
        if dry:
            continue
        with open(r["log_path"], "w") as logf:
            t0 = time.time()
            proc = subprocess.Popen(
                r["cmd"],
                cwd=r["repo_dir"],
                env=r["env"],
                stdout=logf,
                stderr=subprocess.STDOUT,
            )
            code = proc.wait()
            dur = time.time() - t0
            print(f"[Train] Finished {r['tag']} in {dur:.2f}s, exit={code}")

        metrics = {"exit_code": code, "duration_sec": round(dur, 2)}
        # Try to extract PSNR-like value from logs
        try:
            with open(r["log_path"], "r") as f:
                log_txt = f.read()
            psnr = _parse_psnr_from_log(log_txt)
            if psnr is not None:
                metrics["psnr"] = psnr
        except Exception:
            pass

        # Persist per-run results
        with open(r["result_path"], "w") as f:
            json.dump(metrics, f, indent=2)

        # Post steps (render/metrics) per model/defaults knobs
        model_cfg = next(m for m in cfg["models"] if m["name"] == r["model"])
        _ = _maybe_run_render(r, model_cfg, python_exe)
        m_code, m_dur, m_vals = _maybe_run_metrics(r, model_cfg, python_exe)

        for k in ("psnr", "ssim", "lpips"):
            if k not in metrics:
                metrics[k] = None
        # Merge metrics results if present
        metrics.update({k: v for k, v in m_vals.items() if v is not None})

        with open(r["result_path"], "w") as f:
            json.dump(metrics, f, indent=2)

        summary_rows.append({
            "tag": r["tag"],
            "scene": r["scene"],
            "model": r["model"],
            "seed": r["seed"],
            **metrics
        })

        print(f"\n[Done] Completed full pipeline for {r['tag']} ✅")

    return summary_rows


def append_summary(summary_rows: List[Dict[str, Any]], out_root: Path):
    if not summary_rows:
        return
    csv_path = out_root / "summary.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["tag","scene","model","seed","exit_code","duration_sec","psnr","ssim","lpips"]

    # Normalize rows
    for rsum in summary_rows:
        for k in fieldnames:
            rsum.setdefault(k, None)

    write_header = (not csv_path.exists()) or (csv_path.stat().st_size == 0)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerows(summary_rows)
    print("\nSummary appended to", csv_path)
