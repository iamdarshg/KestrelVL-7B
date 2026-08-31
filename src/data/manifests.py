"""Provenance-aware JSONL/SQLite manifest primitives."""

import hashlib
import json
import sqlite3
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Iterator


@dataclass
class SampleRecord:
    sample_id: str
    source: str
    source_id: str
    license: str
    modality: str
    quality_score: float
    teacher_model: str | None = None
    teacher_provenance_confidence: float = 0.0
    duplicate_group: str | None = None
    contamination_status: str = "unchecked"
    text_hash: str | None = None
    renderer_seed: int | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))


class ManifestStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("CREATE TABLE IF NOT EXISTS samples (sample_id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
        self.connection.commit()

    def add(self, record: SampleRecord) -> None:
        self.connection.execute("INSERT OR REPLACE INTO samples VALUES (?, ?)", (record.sample_id, record.to_json()))
        self.connection.commit()

    def add_many(self, records: Iterable[SampleRecord]) -> int:
        count = 0
        for record in records:
            self.add(record)
            count += 1
        return count

    def __iter__(self) -> Iterator[SampleRecord]:
        rows = self.connection.execute("SELECT payload FROM samples ORDER BY sample_id")
        for (payload,) in rows:
            yield SampleRecord(**json.loads(payload))

    def digest(self) -> str:
        digest = hashlib.sha256()
        for record in self:
            digest.update(record.to_json().encode())
        return digest.hexdigest()

    def close(self) -> None:
        self.connection.close()

