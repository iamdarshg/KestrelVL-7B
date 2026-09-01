# TIPSv2 L/14 local load report

Date: 2026-09-01

Source: `google/tipsv2-l14`, Hugging Face `main`, observed repository commit
`52847a71eb02a3082c370fbcd98cd799687bd4d0`.

The checkpoint is present at `data/raw/tipsv2-l14` outside Git. The complete
`model.safetensors` file is 1,951,820,760 bytes and has SHA-256
`c75707eabca655e641a6c47a59dd26b347290ceadb993b34af184e7f654c5809`, matching
the repository's LFS metadata. The download manifest records hashes for all
nine fetched files.

## Verified behavior

The custom remote-code model loads locally through the Kestrel vision adapter.
The installed Transformers development build currently assumes optional
safetensors `format` metadata is non-null and fails on this valid bare file.
The adapter therefore has a TIPSv2-specific fallback: instantiate the pinned
custom config, load the safetensors state directly, verify missing/unexpected
keys, and then apply the requested dtype.

The image smoke used one 448px tile of synthetic RGB input:

| device | output | dtype | finite | peak allocated | peak reserved |
| --- | --- | --- | --- | ---: | ---: |
| CPU | `[1, 1024, 1024]` | FP32 | yes | not recorded | not recorded |
| RTX 4060 Laptop GPU | `[1, 1024, 1024]` | BF16 | yes | 0.997 GiB | 1.053 GiB |

The adapter selects the 1024 patch tokens and excludes the class/register
tokens from the language projector stream. It preserves TIPSv2's processor
contract: RGB values in `[0, 1]`, 448px input, and no channel normalization.
The staged `projector`, `last4`, `upper12`, and `all` controls discover the
TIPSv2 `vision_encoder.blocks` layout. xFormers was unavailable in this local
environment, so this is a correctness smoke, not a performance result.

## Reproduction

```powershell
python scripts/download_tipsv2.py
python scripts/verify_vision_checkpoint.py .\data\raw\tipsv2-l14 --device cuda
```

This report establishes checkpoint integrity and a single-tile forward only.
It does not establish multimodal alignment quality, screenshot OCR quality,
long-context behavior, Q4 compatibility, or any public benchmark result.
