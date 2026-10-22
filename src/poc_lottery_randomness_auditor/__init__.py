"""Command line entry point for the lottery randomness auditor."""

import argparse
from datetime import datetime
from pathlib import Path

from poc_lottery_randomness_auditor.engine.simulate import MASTER_SEED


def main() -> None:
    """Run the audit from the command line."""
    parser = argparse.ArgumentParser(description="Audit lottery draw randomness.")
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=MASTER_SEED)
    parser.add_argument("--now", type=datetime.fromisoformat, default=None)
    args = parser.parse_args()

    from poc_lottery_randomness_auditor.audit import run

    run(args.snapshot, args.out, seed=args.seed, now=args.now)
