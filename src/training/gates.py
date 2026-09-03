"""Hard stage gates that prevent accidental post-training before recovery."""

import json
from pathlib import Path


def assert_reconstruction_gate(path: str | Path, minimum_retention: float = 0.95) -> dict[str, object]:
    report_path = Path(path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    raw_retention = report.get("teacher_capability_retention")
    try:
        retention = float(raw_retention)
    except (TypeError, ValueError):
        retention = float("nan")
    if not retention >= minimum_retention or report.get("status") != "pass":
        value = "unavailable" if raw_retention is None else f"{retention:.4f}"
        raise RuntimeError(f"reconstruction gate failed: retention={value}, status={report.get('status')!r}")
    return report
