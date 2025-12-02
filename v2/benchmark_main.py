# benchmark_main.py
# Entry point that wires bench_config + bench_pipeline together.

from __future__ import annotations
import argparse
import sys
from pathlib import Path

from bench_config import load_config, resolve_config, ensure_dir
from bench_pipeline import build_runs, execute_runs, append_summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="config.yaml")
    ap.add_argument("--python", default=sys.executable, help="Python to use for subprocs")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    raw = load_config(Path(args.config))
    cfg, ctx, scenes_resolved = resolve_config(raw)

    out_root = Path(cfg["defaults"].get("out_root", "./bench_runs")).expanduser().resolve()
    ensure_dir(out_root)

    runs = build_runs(cfg, out_root, args.python)
    summary_rows = execute_runs(runs, cfg, dry=args.dry, python_exe=args.python)
    if not args.dry:
        append_summary(summary_rows, out_root)


if __name__ == "__main__":
    main()
