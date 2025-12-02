import argparse
import sys
from fs3dgs_benchmark import benchmark

def main():
    parser = argparse.ArgumentParser(description="Unified CLI for FS3DGS Benchmarking")
    subparsers = parser.add_subparsers(dest="command", required=True)

    runall_parser = subparsers.add_parser("runall", help="Run the full benchmark pipeline")
    runall_parser.add_argument(
        "--config",
        default="fs3dgs_benchmark/config.yaml",
        help="Path to configuration file",
    )

    args = parser.parse_args()

    if args.command == "runall":
        print(f"Starting full benchmark using config: {args.config}")

        # ✅ Remove 'runall' so benchmark.py sees only clean args
        sys.argv = [sys.argv[0]] + sys.argv[2:]

        benchmark.main()
