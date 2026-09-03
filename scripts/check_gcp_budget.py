"""Check a proposed run against the cumulative research budget without launch."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from training.budget import check_budget, conservative_cost  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", default="reports/costs/gcp_experiment_ledger.json")
    parser.add_argument("--hours", type=float, required=True)
    parser.add_argument("--hourly-rate", type=float, required=True)
    parser.add_argument("--budget", type=float, default=30.0)
    parser.add_argument("--uncertainty-fraction", type=float, default=0.15)
    args = parser.parse_args()
    decision = check_budget(
        ROOT / args.ledger,
        conservative_cost(args.hours, args.hourly_rate, args.uncertainty_fraction),
        args.budget,
    )
    print(json.dumps(decision.__dict__, indent=2))
    if not decision.allowed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
