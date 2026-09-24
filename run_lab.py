"""Run the adverse-selection experiment."""

import argparse
from pathlib import Path

from mm_lab.experiment import run_experiment


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run fewer sessions to check the installation.",
    )

    parser.add_argument(
        "--kind",
        choices=["call", "put"],
        default="call",
        help="Option type to simulate.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Directory for generated results.",
    )

    args = parser.parse_args()

    root = Path(__file__).resolve().parent

    output = args.output or root / (
        "results_quick" if args.quick else "results"
    )

    run_experiment(
        output,
        quick=args.quick,
        kind=args.kind,
    )


if __name__ == "__main__":
    main()
