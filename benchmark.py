import argparse, itertools, json, os, re, shlex, subprocess, sys, time
from pathlib import Path

try:
    import yaml
except ImportError:
    print("pip install pyyaml")
    raise

def deep_merge(a, b):
    """Non-destructive recursive merge: values in b override a."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = dict(a)
        for k, v in b.items():
            out[k] = deep_merge(a.get(k), v)
        return out
    return b if b is not None else a
import re

def resolve_refs(d, ctx):
    """
    Recursively resolve ${a.b.c} anywhere in strings (embedded or whole).
    """
    def lookup(path):
        cur = ctx
        for p in path.split("."):
            cur = cur[p]
        return cur

    def _resolve_val(v):
        if isinstance(v, str):
            # replace all ${...} occurrences in the string
            def repl(m):
                return str(lookup(m.group(1)))
            return re.sub(r"\$\{([^}]+)\}", repl, v)
        return v

    if isinstance(d, dict):
        return {k: resolve_refs(_resolve_val(v), ctx) for k, v in d.items()}
    if isinstance(d, list):
        return [resolve_refs(_resolve_val(v), ctx) for v in d]
    return d

def as_namespace_string(dflat, allowed_keys=None):
    if allowed_keys is not None:
        dflat = {k: v for k, v in dflat.items() if k in allowed_keys}
    parts = [f"{k}={repr(v)}" for k, v in dflat.items()]
    return "Namespace(" + ", ".join(parts) + ")"

def flatten_repo_args(args_dict):
    """
    Turn nested groups into the flat keys expected by that repo's ArgParser.
    For GraphDECO, ModelParams, OptimizationParams, and PipelineParams are all flat flags.
    """
    flat = {}
    for k, v in args_dict.items():
        if isinstance(v, dict):
            # merge nested groups into flat keys
            for k2, v2 in v.items():
                flat[k2] = v2
        else:
            flat[k] = v
    return flat

def build_cli_args(arg_map, flat_args):
    """
    Emit CLI flags only for keys present in arg_map.
    Prevents unknown flags like 'seed' from leaking to the repo.
    """
    cli = []
    for norm_key, cli_name in arg_map.items():
        if norm_key not in flat_args:
            continue
        value = flat_args[norm_key]
        flag = f"--{cli_name}"
        if isinstance(value, bool):
            if value:
                cli.append(flag)
        elif isinstance(value, list):
            cli.append(flag)
            cli.extend([str(x) for x in value])
        elif value is None:
            continue
        else:
            cli += [flag, str(value)]
    return cli


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def expand_grid(base_cfg, grid):
    keys = list(grid.keys())
    vals = [grid[k] for k in keys]
    for combo in itertools.product(*vals):
        yield dict(zip(keys, combo))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="config.yaml")
    ap.add_argument("--python", default=sys.executable, help="Python to use for subprocs")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        raw = yaml.safe_load(f)

    # Fill context and resolve ${...} references
    ctx = {
        "defaults": raw.get("defaults", {}),
        "models": raw.get("models", []),
        "grid": raw.get("grid", {}),
        "overrides": raw.get("overrides", {}),
        "scenes": {},
    }

    cfg = resolve_refs(raw, {"defaults": ctx["defaults"], "scenes": {}}) # scenes resolve later

    # Prepare output root
    out_root = Path(cfg["defaults"].get("out_root", "./bench_runs")).expanduser().resolve()
    ensure_dir(out_root)

    runs = []

    # Build expanded runs
    grid = cfg.get("grid", {})
    grid_expansions = list(expand_grid({}, grid)) if grid else [{}]


    # If you use ${defaults...} inside scenes, resolve them now
    def _resolve_scene(scene):
        return resolve_refs(scene, {"defaults": ctx["defaults"], "scenes": {}})

    scenes_resolved = []
    user_scenes = raw.get("scenes", [])

    if user_scenes and any("name" in s or "source_path" in s for s in user_scenes):
        # Use whatever is explicitly listed
        for s in user_scenes:
            s = resolve_refs(s, {"defaults": ctx["defaults"], "scenes": {}})
            scenes_resolved.append({"name": s["name"], "source_path": s["source_path"]})
            ctx["scenes"][s["name"]] = s
    else:
        # Auto-discover all subfolders under dataset_root
        root = Path(cfg["defaults"]["dataset_root"]).expanduser().resolve()
        for sub in sorted(root.iterdir()):
            if sub.is_dir():
                scenes_resolved.append({"name": sub.name, "source_path": str(sub)})
                ctx["scenes"][sub.name] = {"name": sub.name, "source_path": str(sub)}

    cfg["scenes"] = scenes_resolved

    # If you previously used cfg.get("scenes"), switch to scenes_resolved in the run builder loop:
    # for scene in cfg.get("scenes", []):
    for scene in scenes_resolved:

        scene_name = scene["name"]
        # For resolving ${scenes.<name>.*} if needed later
        ctx["scenes"][scene_name] = scene

        for model in cfg.get("models", []):
            model_name = model["name"]
            ov = cfg.get("overrides", {}).get(model_name, {}).get(scene_name, {})
            base_args = deep_merge(cfg["defaults"].get("common_args", {}), model.get("args", {}))
            base_args["source_path"] = scene["source_path"]
            # model_path will be set per run instance
            for grid_vars in grid_expansions:
                seed = grid_vars.get("seeds", cfg["defaults"].get("seeds", [42]))  # seed may be a list
                seed_list = seed if isinstance(seed, list) else [seed]
                for s in seed_list:
                    merged = deep_merge(base_args, ov.get("args", {}))
                    # fill seed into args if the repo accepts it, otherwise just propagate for naming
                    merged["seed"] = s

                    # flattened args for the repo
                    flat = flatten_repo_args(merged)
                    # compute model_path per run
                    tag = f"{scene_name}__{model_name}__sh{flat.get('sh_degree', 'NA')}__it{flat.get('iterations', 'NA')}__seed{s}"
                    run_dir    = (out_root / tag).resolve()
                    model_path = (run_dir / "model").resolve()
                    log_dir    = (run_dir / "logs").resolve()
                    ensure_dir(model_path)
                    ensure_dir(log_dir)

                    flat["model_path"]  = str(model_path)
                    flat["source_path"] = str(Path(scene["source_path"]).resolve())

                    # Build command
                    repo_dir = Path(model["repo_dir"]).resolve()
                    entry = (repo_dir / model.get("entry", "train.py")).resolve()
                    arg_map = model.get("arg_map", {})
                    if model.get("needs_cfg_args_file", False):
                        ns_str = as_namespace_string(flat, allowed_keys=set(arg_map.keys()))
                        with open(model_path / "cfg_args", "w") as f:
                            f.write(ns_str)
                    cli_flags = build_cli_args(arg_map, flat)

                    cmd = [args.python, str(entry)] + cli_flags

                    # arg_map = model.get("arg_map", {})

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
                        "args_flat": flat
                    })

    # Execute
    summary_rows = []
    for r in runs:
        print("\n==> Running:", r["tag"])
        print("CMD:", " ".join(shlex.quote(x) for x in r["cmd"]))
        if args.dry:
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

        # Try to collect metrics if your repo writes any JSON or can be parsed from logs.
        # This block is conservative: it extracts last PSNR-like number if present.
        metrics = {"exit_code": code, "duration_sec": round(dur, 2)}
        try:
            with open(r["log_path"], "r") as f:
                log_txt = f.read()
            m = re.findall(r"PSNR[:=]\s*([0-9.]+)", log_txt)
            if m:
                metrics["psnr"] = float(m[-1])
        except Exception:
            pass

        # Write basic result file
        with open(r["result_path"], "w") as f:
            json.dump(metrics, f, indent=2)

        # --- NEW: Post-Training Steps (Render + Metrics) ---
        model_cfg = next(m for m in cfg["models"] if m["name"] == r["model"])
        run_render = model_cfg.get("run_render", cfg["defaults"].get("run_render", False))
        run_metrics = model_cfg.get("run_metrics", cfg["defaults"].get("run_metrics", False))

        repo_dir = Path(r["repo_dir"])
        model_path = r["args_flat"]["model_path"]
        source_path = r["args_flat"]["source_path"]
        env = r["env"]

        # --- Run render.py ---
        if run_render:
            print(f"\n[Render] Running render for {r['tag']}")
            render_script = model_cfg.get("render_entry", "render.py")
            render_log = Path(r["log_path"]).with_name("render.log")
            cmd_render = [args.python, render_script,
              "--model_path", model_path,
              "--source_path", source_path,
              "--eval"]  # <<< add this

            with open(render_log, "w") as logf:
                print(f"\n[Render] Running render for {r['tag']}")
                print("CMD:", " ".join(shlex.quote(x) for x in cmd_render))
                t0 = time.time()
                proc = subprocess.Popen(
                    cmd_render, cwd=repo_dir, env=env,
                    stdout=logf, stderr=subprocess.STDOUT
                )
                render_code = proc.wait()
                render_dur = time.time() - t0
                print(f"[Render] Finished {r['tag']} in {render_dur:.2f}s, exit={render_code}")



        # --- Run metrics.py ---
        if run_metrics:
            print(f"\n[Metrics] Running metrics for {r['tag']}")
            metrics_script = model_cfg.get("metrics_entry", "metrics.py")
            metrics_log = Path(r["log_path"]).with_name("metrics.log")
            # metrics.py expects one or more model paths after --model_paths / -m
            it_str = str(r["args_flat"].get("iterations", -1))
            if r["model"].lower() == "fsgs":
                cmd_metrics = [
                    args.python, metrics_script,
                    "--model_path",  r["args_flat"]["model_path"],
                    "--source_path", r["args_flat"]["source_path"],
                    "--iteration",   it_str,
                ]
            else:
                cmd_metrics = [
                    args.python, metrics_script,
                    "--model_paths", r["args_flat"]["model_path"],
                ]
            with open(metrics_log, "w") as logf:
                print(f"\n[Metrics] Running metrics for {r['tag']}")
                print("CMD:", " ".join(shlex.quote(x) for x in cmd_metrics))
                t0 = time.time()
                proc = subprocess.Popen(
                    cmd_metrics, cwd=repo_dir, env=env,
                    stdout=logf, stderr=subprocess.STDOUT
                )
                metrics_code = proc.wait()
                metrics_dur = time.time() - t0
                print(f"[Metrics] Finished {r['tag']} in {metrics_dur:.2f}s, exit={metrics_code}")



            # --- Merge results.json (from metrics.py) into our 'metrics' dict ---
            for k in ("psnr", "ssim", "lpips"):
                metrics.setdefault(k, None)

            try:
                method = f"ours_{r['args_flat'].get('iterations', 'NA')}"
                rp = Path(model_path) / "results.json"
                if rp.exists():
                    with open(rp) as jf:
                        data = json.load(jf)  # expected: {"ours_1000": {"SSIM":..., "PSNR":..., "LPIPS":...}}
                    # prefer exact method; fallback to first entry
                    block = data.get(method) or (next(iter(data.values())) if isinstance(data, dict) and data else None)
                    if isinstance(block, dict):
                        if "PSNR" in block:  metrics["psnr"]  = float(block["PSNR"])
                        if "SSIM" in block:  metrics["ssim"]  = float(block["SSIM"])
                        if "LPIPS" in block: metrics["lpips"] = float(block["LPIPS"])
            except Exception:
                pass

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



        

    # Write/append a single summary.csv across runs
    if not args.dry and summary_rows:
        import csv

        csv_path = Path(cfg["defaults"].get("out_root", "./bench_runs")) / "summary.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        # Pick stable columns; add more if you like
        fieldnames = ["tag","scene","model","seed","exit_code","duration_sec","psnr","ssim","lpips"]

        # Ensure all rows have these keys
        for rsum in summary_rows:
            for k in fieldnames:
                rsum.setdefault(k, None)

        write_header = (not csv_path.exists()) or (csv_path.stat().st_size == 0)

        with open(csv_path, "a", newline="") as f:   # <-- append mode
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                w.writeheader()
            w.writerows(summary_rows)

        print("\nSummary appended to", csv_path)


if __name__ == "__main__":
    main()
