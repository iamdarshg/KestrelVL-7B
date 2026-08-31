"""Refuse unsafe/fake GGUF exports until a real converter is configured."""

import argparse
import shutil


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--converter", default="convert_hf_to_gguf.py")
    args = parser.parse_args()
    if shutil.which(args.converter) is None:
        raise SystemExit(f"GGUF export blocked: converter {args.converter!r} is not installed")
    raise SystemExit("configure and audit the llama.cpp converter invocation before exporting")


if __name__ == "__main__":
    main()

