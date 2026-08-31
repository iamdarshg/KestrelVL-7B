"""Hard stage gates that prevent accidental post-training before recovery."""

import json
from pathlib import Path


def assert_reconstruction_gate(path: str | Path, minimum_retention: float = 0.95) -> dict[str, object]:
    report_path = Path(path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    retention = float(report.get("teacher_capability_retention", 0.0))
    if retention < minimum_retention or report.get("status") != "pass":
        raise RuntimeError(f"reconstruction gate failed: retention={retention:.4f}, status={report.get('status')!r}")
    return report

