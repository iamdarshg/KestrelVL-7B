"""Restartable JSONL streaming with an explicit example cursor."""

import json
from pathlib import Path
from typing import Iterator


class ResumableJSONL:
    def __init__(self, path: str | Path, start_line: int = 0) -> None:
        self.path = Path(path)
        self.start_line = start_line

    def __iter__(self) -> Iterator[tuple[int, dict[str, object]]]:
        with self.path.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle):
                if line_no < self.start_line or not line.strip():
                    continue
                yield line_no, json.loads(line)

