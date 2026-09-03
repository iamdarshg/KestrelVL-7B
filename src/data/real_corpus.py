"""Governed, deterministic streaming corpus for production experiments.

This module deliberately does not provide a synthetic fallback.  A production
experiment must point each configured source at a local JSONL/JSONL.GZ stream or
an explicitly pinned Hugging Face streaming dataset whose rows contain text.
Metadata-only datasets such as Stack-Edu and RefineCode are accepted as
provenance references, but fail closed until a content resolver is configured.

The stream is deterministic by construction: source choice, split assignment,
deduplication and FIM decisions are pure functions of the manifest, seed and
source cursors.  The returned state is sufficient for checkpoint/resume.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch

from .fim import make_fim, should_fim
from .manifests import ManifestStore, SampleRecord


SOURCE_WEIGHTS = {
    "stack-edu": 0.35,
    "refinecode": 0.25,
    "stack-v2": 0.20,
    "docs": 0.10,
    "history": 0.10,
}


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize_text(text: str) -> str:
    """Normalize without destroying identifiers needed by source metadata."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _repository_key(row: dict[str, Any], source: "RealSourceSpec", sample_id: str) -> str:
    for key in source.repository_fields:
        value = row.get(key)
        if value not in (None, ""):
            return f"{source.name}:{value}"
    return f"{source.name}:sample:{sample_id}"


def _minhash_signature(text: str, width: int = 5, permutations: int = 8) -> tuple[int, ...]:
    words = normalize_text(text).lower().split()
    shingles = [" ".join(words[i : i + width]) for i in range(max(0, len(words) - width + 1))]
    if not shingles:
        shingles = ["<empty>"]
    return tuple(
        min(int.from_bytes(hashlib.blake2b(f"{seed}:{item}".encode(), digest_size=8).digest(), "big")
            for item in shingles)
        for seed in range(permutations)
    )


class StreamingCorpusError(RuntimeError):
    """Raised when a governed corpus cannot safely provide real samples."""


@dataclass(frozen=True)
class RealSourceSpec:
    name: str
    weight: float
    kind: str
    uri: str
    revision: str
    split: str = "train"
    dataset_config: str | None = None
    text_field: str = "text"
    id_fields: tuple[str, ...] = ("id", "sample_id", "blob_id", "path")
    repository_fields: tuple[str, ...] = ("repo_name", "repository", "repo", "project_id")
    license_fields: tuple[str, ...] = ("license", "license_type", "detected_licenses")
    required_license: bool = True
    content_resolver: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "weight": self.weight,
            "kind": self.kind,
            "uri": self.uri,
            "revision": self.revision,
            "split": self.split,
            "dataset_config": self.dataset_config,
            "text_field": self.text_field,
            "id_fields": list(self.id_fields),
            "repository_fields": list(self.repository_fields),
            "license_fields": list(self.license_fields),
            "required_license": self.required_license,
            "content_resolver": self.content_resolver,
        }


@dataclass(frozen=True)
class RealCorpusSpec:
    """Immutable identity and governance policy for one corpus experiment."""

    name: str
    seed: int
    sequence_length: int
    source_block_tokens: int
    fim_rate: float
    tokenizer_id: str
    source_specs: tuple[RealSourceSpec, ...]
    architecture_validation_fraction: float = 0.02
    recovery_validation_fraction: float = 0.02
    min_chars: int = 32
    max_chars: int = 2_000_000
    dedup_jaccard_threshold: float = 0.92
    held_out_ids: tuple[str, ...] = ()
    manifest_version: str = "real-stream-v1"

    def __post_init__(self) -> None:
        if not self.source_specs:
            raise ValueError("a real corpus needs at least one source")
        names = {source.name for source in self.source_specs}
        if names != set(SOURCE_WEIGHTS):
            raise ValueError(f"sources must preserve the final composition categories: {sorted(names)}")
        total = sum(source.weight for source in self.source_specs)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"source weights must sum to one, got {total}")
        if self.sequence_length < 2 or self.source_block_tokens < 1:
            raise ValueError("sequence_length and source_block_tokens must be positive")
        if not 0 <= self.fim_rate <= 1 or not 0 <= self.architecture_validation_fraction < 1:
            raise ValueError("invalid corpus fractions")
        if not 0 <= self.recovery_validation_fraction < 1:
            raise ValueError("invalid recovery_validation_fraction")
        if self.min_chars < 1 or self.max_chars < self.min_chars:
            raise ValueError("invalid character bounds")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RealCorpusSpec":
        import yaml

        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        sources: list[RealSourceSpec] = []
        for name, raw in payload.get("sources", {}).items():
            if not isinstance(raw, dict):
                raise ValueError(f"real source {name} must be a mapping")
            sources.append(RealSourceSpec(
                name=str(name),
                weight=float(raw["weight"]),
                kind=str(raw["kind"]),
                uri=str(raw["uri"]),
                revision=str(raw["revision"]),
                split=str(raw.get("split", "train")),
                dataset_config=raw.get("dataset_config"),
                text_field=str(raw.get("text_field", "text")),
                id_fields=tuple(map(str, raw.get("id_fields", ["id", "sample_id", "blob_id", "path"]))),
                repository_fields=tuple(map(str, raw.get("repository_fields", ["repo_name", "repository", "repo", "project_id"]))),
                license_fields=tuple(map(str, raw.get("license_fields", ["license", "license_type", "detected_licenses"]))),
                required_license=bool(raw.get("required_license", True)),
                content_resolver=raw.get("content_resolver"),
            ))
        governance = payload.get("governance", {})
        transforms = payload.get("transforms", {})
        return cls(
            name=str(payload["name"]),
            seed=int(payload["seed"]),
            sequence_length=int(payload["sequence_length"]),
            source_block_tokens=int(payload["source_block_tokens"]),
            fim_rate=float(transforms.get("fim_rate", 0.35)),
            tokenizer_id=str(payload.get("tokenizer_id", "unresolved")),
            source_specs=tuple(sources),
            architecture_validation_fraction=float(governance.get("architecture_validation_fraction", 0.02)),
            recovery_validation_fraction=float(governance.get("recovery_validation_fraction", 0.02)),
            min_chars=int(governance.get("min_chars", 32)),
            max_chars=int(governance.get("max_chars", 2_000_000)),
            dedup_jaccard_threshold=float(governance.get("dedup_jaccard_threshold", 0.92)),
            held_out_ids=tuple(map(str, governance.get("held_out_ids", []))),
            manifest_version=str(payload.get("manifest_version", "real-stream-v1")),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "manifest_version": self.manifest_version,
            "name": self.name,
            "seed": self.seed,
            "sequence_length": self.sequence_length,
            "source_block_tokens": self.source_block_tokens,
            "fim_rate": self.fim_rate,
            "tokenizer_id": self.tokenizer_id,
            "sources": [source.to_dict() for source in self.source_specs],
            "governance": {
                "architecture_validation_fraction": self.architecture_validation_fraction,
                "recovery_validation_fraction": self.recovery_validation_fraction,
                "min_chars": self.min_chars,
                "max_chars": self.max_chars,
                "dedup_jaccard_threshold": self.dedup_jaccard_threshold,
                "held_out_ids": list(self.held_out_ids),
            },
        }

    def fingerprint(self) -> str:
        return _sha256(_canonical(self.to_dict()))


def _cycle(spec: RealCorpusSpec) -> tuple[str, ...]:
    # The requested percentages are exact over a 20-slot deterministic cycle.
    slots: list[str] = []
    for source in spec.source_specs:
        slots.extend([source.name] * round(source.weight * 20))
    if len(slots) != 20:
        raise ValueError("the locked source composition must map to 20 deterministic slots")
    return tuple(slots)


class _DuplicateIndex:
    def __init__(self, threshold: float) -> None:
        self.threshold = threshold
        self.exact: dict[str, str] = {}
        self.normalized: dict[str, str] = {}
        self.signatures: dict[tuple[int, ...], tuple[str, str]] = {}

    def check(self, sample_id: str, text: str) -> str:
        raw = _sha256(text)
        norm = _sha256(normalize_text(text).lower())
        if raw in self.exact or norm in self.normalized:
            return "exact_duplicate"
        signature = _minhash_signature(text)
        for prior, (prior_id, prior_split) in self.signatures.items():
            equal = sum(left == right for left, right in zip(signature, prior)) / len(signature)
            if equal >= self.threshold:
                return "near_duplicate"
        return "clean"

    def add(self, sample_id: str, text: str, split: str) -> None:
        self.exact[_sha256(text)] = sample_id
        self.normalized[_sha256(normalize_text(text).lower())] = sample_id
        self.signatures[_minhash_signature(text)] = (sample_id, split)

    def state_dict(self) -> dict[str, object]:
        return {
            "exact": dict(self.exact),
            "normalized": dict(self.normalized),
            "signatures": [[list(signature), sample_id, split] for signature, (sample_id, split) in self.signatures.items()],
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.exact = {str(key): str(value) for key, value in dict(state.get("exact", {})).items()}
        self.normalized = {str(key): str(value) for key, value in dict(state.get("normalized", {})).items()}
        self.signatures = {
            tuple(map(int, item[0])): (str(item[1]), str(item[2]))
            for item in state.get("signatures", [])
        }


class RealStreamingCorpus:
    """Read governed examples while retaining all source cursors in state."""

    def __init__(self, spec: RealCorpusSpec, manifest_store: ManifestStore | None = None) -> None:
        self.spec = spec
        self.manifest_store = manifest_store
        self._source_by_name = {source.name: source for source in spec.source_specs}
        self._iters: dict[str, Iterator[dict[str, Any]]] = {}
        self._source_cursors = {source.name: 0 for source in spec.source_specs}
        self._index = _DuplicateIndex(spec.dedup_jaccard_threshold)
        self._accepted = 0
        self._token_buffers: dict[str, list[int]] = {source.name: [] for source in spec.source_specs}
        self._token_cursor = 0

    def validate_configuration(self) -> None:
        """Fail before model construction when a local source is unavailable."""
        for source in self.spec.source_specs:
            if source.kind not in {"jsonl", "jsonl.gz", "hf_streaming"}:
                raise StreamingCorpusError(f"unsupported real source kind: {source.kind}")
            if source.kind in {"jsonl", "jsonl.gz"}:
                path = Path(source.uri)
                if not path.is_absolute():
                    path = Path.cwd() / path
                if not path.is_file():
                    raise StreamingCorpusError(f"real corpus source is missing: {path}")

    def _iter_source(self, source: RealSourceSpec) -> Iterator[dict[str, Any]]:
        if source.kind in {"jsonl", "jsonl.gz"}:
            path = Path(source.uri)
            if not path.exists():
                raise StreamingCorpusError(f"real corpus source is missing: {path}")
            opener = gzip.open if source.kind.endswith(".gz") else open
            with opener(path, "rt", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        row = json.loads(line)
                        if not isinstance(row, dict):
                            raise StreamingCorpusError(f"source {source.name} yielded a non-object row")
                        yield row
            return
        if source.kind == "hf_streaming":
            try:
                from datasets import load_dataset
            except ImportError as exc:
                raise StreamingCorpusError("datasets is required for hf_streaming sources") from exc
            dataset = load_dataset(source.uri, name=source.dataset_config, split=source.split,
                                   revision=source.revision, streaming=True)
            for row in dataset:
                yield dict(row)
            return
        raise StreamingCorpusError(f"unsupported real source kind: {source.kind}")

    @staticmethod
    def _resolve_content(source: RealSourceSpec, row: dict[str, Any]) -> str | None:
        """Resolve a content-addressed row through the declared SWH resolver."""
        if source.content_resolver != "softwareheritage_s3":
            return None
        blob_id = row.get("blob_id")
        if not isinstance(blob_id, str) or not blob_id:
            return None
        try:
            import boto3
        except ImportError as exc:
            raise StreamingCorpusError(
                "boto3 is required for the declared softwareheritage_s3 content resolver"
            ) from exc
        try:
            import gzip as _gzip

            client = boto3.client("s3")
            response = client.get_object(Bucket="softwareheritage", Key=f"content/{blob_id}")
            return _gzip.GzipFile(fileobj=response["Body"]).read().decode("utf-8", errors="ignore")
        except Exception as exc:  # pragma: no cover - requires external S3 access
            raise StreamingCorpusError(f"SWH content resolution failed for blob_id={blob_id}") from exc

    def _iterator(self, source: RealSourceSpec) -> Iterator[dict[str, Any]]:
        if source.name not in self._iters:
            iterator = self._iter_source(source)
            for _ in range(self._source_cursors[source.name]):
                next(iterator, None)
            self._iters[source.name] = iterator
        return self._iters[source.name]

    def _row_to_record(self, source: RealSourceSpec, row: dict[str, Any], ordinal: int, split: str) -> SampleRecord:
        sample_id = next((str(row[key]) for key in source.id_fields if row.get(key) not in (None, "")), f"{source.name}:{ordinal}")
        text_value = row.get(source.text_field)
        if not isinstance(text_value, str) or not text_value.strip():
            text_value = self._resolve_content(source, row)
        if not isinstance(text_value, str) or not text_value.strip():
            if source.content_resolver:
                raise StreamingCorpusError(
                    f"source {source.name} row {sample_id} has no text; content resolver {source.content_resolver!r} is not implemented"
                )
            raise StreamingCorpusError(
                f"source {source.name} row {sample_id} has no {source.text_field!r}; metadata-only sources must be resolved explicitly"
            )
        text = text_value
        if len(text) < self.spec.min_chars or len(text) > self.spec.max_chars:
            raise ValueError("length_filtered")
        license_value: object = None
        for key in source.license_fields:
            if row.get(key) not in (None, "", [], {}):
                license_value = row[key]
                break
        if source.required_license and license_value in (None, "", "no_license", "unknown", []):
            raise ValueError("missing_license")
        contamination = "benchmark_holdout" if sample_id in self.spec.held_out_ids else "clean"
        if contamination != "clean":
            raise ValueError("benchmark_holdout")
        duplicate = self._index.check(sample_id, text)
        if duplicate != "clean":
            raise ValueError(duplicate)
        repository = _repository_key(row, source, sample_id)
        record = SampleRecord(
            sample_id=f"{source.name}:{sample_id}",
            source=source.name,
            source_id=sample_id,
            license=json.dumps(license_value, sort_keys=True) if isinstance(license_value, (list, dict)) else str(license_value),
            modality="text",
            quality_score=float(row.get("quality_score", row.get("score", 1.0))),
            contamination_status=contamination,
            text_hash=_sha256(text),
            metadata={
                "split": split,
                "repository_id": repository,
                "source_revision": source.revision,
                "source_split": source.split,
                "raw_keys": sorted(row),
                "transform": "none_until_after_split",
            },
        )
        self._index.add(sample_id, text, split)
        if self.manifest_store:
            self.manifest_store.add(record)
        return record

    def split_for(self, source: RealSourceSpec, row: dict[str, Any], sample_id: str) -> str:
        repository = _repository_key(row, source, sample_id)
        bucket = int(_sha256(f"{self.spec.seed}:{repository}")[:8], 16) / 2**32
        if bucket < self.spec.recovery_validation_fraction:
            return "recovery_validation"
        if bucket < self.spec.recovery_validation_fraction + self.spec.architecture_validation_fraction:
            return "architecture_validation"
        return "train"

    def next_record(self, token_index: int, requested_split: str = "train") -> tuple[SampleRecord, str]:
        cycle = _cycle(self.spec)
        source = self._source_by_name[cycle[(token_index // self.spec.source_block_tokens) % len(cycle)]]
        iterator = self._iterator(source)
        while True:
            try:
                row = next(iterator)
            except StopIteration:
                self._iters.pop(source.name, None)
                self._source_cursors[source.name] = 0
                iterator = self._iterator(source)
                row = next(iterator)
            self._source_cursors[source.name] += 1
            sample_id = next((str(row[key]) for key in source.id_fields if row.get(key) not in (None, "")), f"{source.name}:{self._source_cursors[source.name] - 1}")
            split = self.split_for(source, row, sample_id)
            if split != requested_split:
                continue
            try:
                record = self._row_to_record(source, row, self._source_cursors[source.name] - 1, split)
            except ValueError:
                continue
            self._accepted += 1
            return record, str(row[source.text_field])

    def state_dict(self) -> dict[str, object]:
        return {
            "manifest_fingerprint": self.spec.fingerprint(),
            "seed": self.spec.seed,
            "source_cursors": dict(self._source_cursors),
            "accepted": self._accepted,
            "token_cursor": self._token_cursor,
            "token_buffers": {name: list(tokens) for name, tokens in self._token_buffers.items() if tokens},
            "dedup_index": self._index.state_dict(),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        if state.get("manifest_fingerprint") != self.spec.fingerprint():
            raise StreamingCorpusError("corpus fingerprint mismatch on resume")
        self._source_cursors = {str(k): int(v) for k, v in dict(state["source_cursors"]).items()}
        self._iters.clear()
        self._accepted = int(state.get("accepted", 0))
        self._token_cursor = int(state.get("token_cursor", 0))
        raw_buffers = state.get("token_buffers", {})
        self._token_buffers = {source.name: list(map(int, dict(raw_buffers).get(source.name, []))) for source in self.spec.source_specs}
        self._index.load_state_dict(dict(state.get("dedup_index", {})))

    def reset(self, preserve_index: bool = False) -> None:
        """Reset iteration state; useful for deterministic validation passes."""
        self._iters.clear()
        self._source_cursors = {source.name: 0 for source in self.spec.source_specs}
        self._token_buffers = {source.name: [] for source in self.spec.source_specs}
        if not preserve_index:
            self._index = _DuplicateIndex(self.spec.dedup_jaccard_threshold)
        self._accepted = 0
        self._token_cursor = 0

    def token_batch(
        self,
        tokenizer: object,
        token_cursor: int,
        sequence_length: int | None = None,
        requested_split: str = "train",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a causal batch from the governed text stream.

        The tokenizer is called only after split and duplicate decisions.  The
        cursor is a token cursor, while source cursors and partial token buffers
        are kept in :meth:`state_dict` for exact checkpoint resume.
        """
        length = sequence_length or self.spec.sequence_length
        if length < 2:
            raise ValueError("sequence_length must be at least 2")
        if token_cursor != getattr(self, "_token_cursor", 0):
            raise StreamingCorpusError(
                "token cursor does not match stream state; restore corpus state before requesting a batch"
            )
        encode = getattr(tokenizer, "encode", None)
        if encode is None:
            raise TypeError("tokenizer must expose encode(text, add_special_tokens=False)")
        output: list[int] = []
        while len(output) < length:
            absolute = token_cursor + len(output)
            cycle = _cycle(self.spec)
            source_name = cycle[(absolute // self.spec.source_block_tokens) % len(cycle)]
            buffer = self._token_buffers[source_name]
            block_remaining = self.spec.source_block_tokens - (absolute % self.spec.source_block_tokens)
            while not buffer:
                _, text = self.next_record(absolute, requested_split=requested_split)
                transformed = self.transformed_text(text, self._accepted - 1)
                encoded = list(map(int, encode(transformed, add_special_tokens=False)))
                if encoded:
                    buffer.extend(encoded)
            take = min(length - len(output), block_remaining, len(buffer))
            output.extend(buffer[:take])
            del buffer[:take]
        self._token_cursor = token_cursor + length
        tokens = torch.tensor(output, dtype=torch.long).view(1, length)
        return tokens[:, :-1], tokens[:, 1:]

    def iter_token_batches(
        self,
        tokenizer: object,
        token_budget: int,
        sequence_length: int | None = None,
        start_cursor: int = 0,
        requested_split: str = "train",
    ) -> Iterator[tuple[int, torch.Tensor, torch.Tensor]]:
        if start_cursor == 0:
            self._token_cursor = 0
        elif getattr(self, "_token_cursor", 0) != start_cursor:
            raise StreamingCorpusError("nonzero start_cursor requires a restored corpus state")
        length = sequence_length or self.spec.sequence_length
        while self._token_cursor + length <= token_budget:
            cursor = self._token_cursor
            inputs, labels = self.token_batch(tokenizer, cursor, length, requested_split)
            yield cursor, inputs, labels

    def transformed_text(self, text: str, sample_index: int) -> str:
        if not should_fim(sample_index, self.spec.fim_rate):
            return text
        midpoint = len(text) // 2
        prefix, middle = text[: midpoint // 2], text[midpoint // 2 : midpoint]
        suffix = text[midpoint:]
        return make_fim(prefix, middle, suffix, random.Random(self.spec.seed + sample_index))


def write_canonical_manifest(spec: RealCorpusSpec, path: str | Path) -> str:
    """Write the exact manifest consumed by future runs and return its hash."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"fingerprint": spec.fingerprint(), "spec": spec.to_dict()}
    target.write_text(_canonical(payload) + "\n", encoding="utf-8")
    return spec.fingerprint()
