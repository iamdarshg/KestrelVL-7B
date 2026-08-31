from pathlib import Path


def read_jsonl_tail(path: str | Path, lines: int = 100) -> list[str]:
    return Path(path).read_text(encoding="utf-8").splitlines()[-lines:]
