import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FILES = [
    "references/openreasoning_nemotron_7b_config.json",
    "references/internvit_300m_448_v2_5_config.json",
    "references/deepseek_v4_flash_base_config.json",
]


def main() -> None:
    hashes = {}
    for name in FILES:
        digest = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        hashes[name] = digest
        print(f"{digest}  {name}")
    manifest = json.loads((ROOT / "references/manifest.json").read_text(encoding="utf-8"))
    manifest["sha256"] = hashes
    (ROOT / "references/manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

