"""Fail-closed cumulative GCP research-budget accounting."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BudgetDecision:
    budget_usd: float
    spent_usd: float
    projected_run_usd: float
    projected_total_usd: float
    remaining_usd: float
    allowed: bool
    reason: str


def load_ledger(path: str | Path) -> dict[str, object]:
    ledger_path = Path(path)
    if not ledger_path.is_file():
        raise FileNotFoundError(f"cumulative GCP ledger is required: {ledger_path}")
    payload = json.loads(ledger_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("entries"), list):
        raise ValueError("GCP ledger must contain an entries array")
    return payload


def spent_from_ledger(ledger: dict[str, object]) -> float:
    entries = ledger.get("entries")
    if not isinstance(entries, list):
        raise ValueError("GCP ledger entries must be a list")
    total = 0.0
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("GCP ledger entry must be an object")
        total += float(entry["estimated_cost_usd"])
    return total


def check_budget(
    ledger_path: str | Path,
    projected_run_usd: float,
    budget_usd: float = 30.0,
    safety_margin_usd: float = 0.05,
) -> BudgetDecision:
    if projected_run_usd < 0 or budget_usd <= 0:
        raise ValueError("budget values must be positive")
    spent = spent_from_ledger(load_ledger(ledger_path))
    projected_total = spent + projected_run_usd
    remaining = budget_usd - spent
    allowed = projected_total + safety_margin_usd <= budget_usd
    reason = (
        "projected cumulative cost is within the hard budget"
        if allowed
        else f"refusing launch: {projected_total:.4f} USD plus safety margin exceeds {budget_usd:.2f} USD"
    )
    return BudgetDecision(budget_usd, spent, projected_run_usd, projected_total, remaining, allowed, reason)


def conservative_cost(hours: float, hourly_rate_usd: float, uncertainty_fraction: float = 0.15) -> float:
    """Price a run conservatively; billing lag is never treated as free compute."""
    if hours < 0 or hourly_rate_usd < 0 or uncertainty_fraction < 0:
        raise ValueError("cost inputs must be non-negative")
    return hours * hourly_rate_usd * (1.0 + uncertainty_fraction)
