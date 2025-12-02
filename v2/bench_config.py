# bench_config.py
# Utilities for configuration parsing, reference resolution, and argument building.
# Split out from benchmark.py

from __future__ import annotations
import itertools
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

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


def resolve_refs(d, ctx):
    """Recursively resolve ${a.b.c} anywhere in strings (embedded or whole)."""
    def lookup(path):
        cur = ctx
        for p in path.split("."):
            cur = cur[p]
        return cur

    def _resolve_val(v):
        if isinstance(v, str):
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
    """Turn nested groups into the flat keys expected by the external repo ArgParser."""
    flat = {}
    for k, v in args_dict.items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                flat[k2] = v2
        else:
            flat[k] = v
    return flat


def build_cli_args(arg_map, flat_args):
    """Emit CLI flags only for keys present in arg_map."""
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


def expand_grid(grid: Dict[str, Iterable[Any]]):
    if not grid:
        yield {}
        return
    keys = list(grid.keys())
    vals = [grid[k] for k in keys]
    for combo in itertools.product(*vals):
        yield dict(zip(keys, combo))


def load_config(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return raw


def _load_models_from_dir(models_dir: Path) -> list:
    models = []
    if not models_dir.exists():
        raise FileNotFoundError(f"models_dir not found: {models_dir}")
    for y in sorted(models_dir.glob("*.y*ml")):
        with open(y, "r") as f:
            m = yaml.safe_load(f) or {}
        if "name" not in m:
            # Default name = filename (without extension)
            m["name"] = y.stem
        models.append(m)
    return models


def resolve_config(raw: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], list]:
    """
    Returns (cfg, ctx, scenes_resolved)
    - cfg is the fully-resolved config with scenes list injected
    - ctx contains defaults/models/grid/overrides/scenes
    - scenes_resolved is a normalized list of scenes {name, source_path}
    """
    models_dir = raw.get("models_dir")
    models_list = raw.get("models", [])
    if models_dir and not models_list:
        models_list = _load_models_from_dir(Path(models_dir))
    ctx = {
        "defaults": raw.get("defaults", {}),
        "models": models_list,
        "grid": raw.get("grid", {}),
        "overrides": raw.get("overrides", {}),
        "scenes": {},
    }

    cfg = resolve_refs(raw, {"defaults": ctx["defaults"], "scenes": {}})

    # Scenes: explicit or auto-discover
    scenes_resolved = []
    user_scenes = raw.get("scenes", [])
    if user_scenes and any("name" in s or "source_path" in s for s in user_scenes):
        for s in user_scenes:
            s = resolve_refs(s, {"defaults": ctx["defaults"], "scenes": {}})
            scenes_resolved.append({"name": s["name"], "source_path": s["source_path"]})
            ctx["scenes"][s["name"]] = s
    else:
        root = Path(cfg["defaults"]["dataset_root"]).expanduser().resolve()
        for sub in sorted(root.iterdir()):
            if sub.is_dir():
                scenes_resolved.append({"name": sub.name, "source_path": str(sub)})
                ctx["scenes"][sub.name] = {"name": sub.name, "source_path": str(sub)}

    cfg["scenes"] = scenes_resolved
    cfg["models"] = ctx["models"]
    return cfg, ctx, scenes_resolved