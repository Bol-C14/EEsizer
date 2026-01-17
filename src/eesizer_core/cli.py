from __future__ import annotations

import argparse
from pathlib import Path

from .analysis.compare_runs import compare_runs


def main() -> None:
    parser = argparse.ArgumentParser(prog="eesizer", description="EEsizer CLI")
    subparsers = parser.add_subparsers(dest="command")

    compare_parser = subparsers.add_parser("compare", help="Compare two run directories.")
    compare_parser.add_argument("--run-a", required=True, type=Path, help="Path to run A directory.")
    compare_parser.add_argument("--run-b", required=True, type=Path, help="Path to run B directory.")
    compare_parser.add_argument("--out", required=True, type=Path, help="Output directory for comparison artifacts.")

    args = parser.parse_args()
    if args.command == "compare":
        compare_runs(args.run_a, args.run_b, args.out)
        print(f"Comparison written to {args.out}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
