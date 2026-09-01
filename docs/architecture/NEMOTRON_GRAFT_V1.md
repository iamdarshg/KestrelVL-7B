# Nemotron graft V1

This document describes the first real-weight Kestrel graft. It is an
architecture and execution contract, not a claim that recovery or multimodal
quality training has finished.

## Language body

The source is `nvidia/OpenReasoning-Nemotron-7B`. Its 3584-dimensional dense
Qwen body, embeddings, RMSNorms, SwiGLU FFNs, tokenizer, and LM head are
preserved. Initial training updates only the replacement attention, mHC, and
the vision projector; broader unfreezing is gated by teacher representation
recovery.

The original Q/K/V/O matrices are projected into the replacement mixer by
truncated SVD and GQA averaging. The new mixer uses 28 query heads, a shared
one-head K/V target, 128-dimensional heads, partial 16-dimensional RoPE,
grouped low-rank output, attention sinks, and a bounded Lightning Indexer.

## Mixer schedule

```text
Nemotron hidden state
        |
        +--> Q low-rank path --> partial RoPE ------------------+
        |                                                       |
        +--> shared K/V --> compression --> indexer --> CSA/HCA +--> grouped output
        |                                                       |
        +--> local causal window --------------------------------+
                                |
                         mHC residual mixer
```

The first two layers are causal sliding-window layers. Remaining layers
alternate CSA and HCA. Compressed state is append-only, projected index keys
are compact, and no dense million-token attention mask may be materialized.
For million-token inference, the reference runner supports an explicit
CPU-resident compressed cache: local K/V remains on the accelerator and
retrieval moves only bounded candidate chunks to the query device. This is
inference-only; gradient training uses the same-device cache.

## Vision graft

The original reference is `OpenGVLab/InternViT-300M-448px-V2_5`; the adapter
also supports the fetched `google/tipsv2-l14` checkpoint. Both expose a
1024-dimensional, 448px/14px-patch spatial stream. InternViT uses its
OpenCLIP-style channel statistics; TIPSv2 follows its shipped processor and
expects RGB values in `[0, 1]` without channel normalization. The adapter
selects TIPSv2's patch tokens rather than its class/register tokens, then sends
them through the same 1024-to-3584 adaptive MLP/resampler projector.
Developer screens retain larger budgets than ordinary images. For frozen
vision at long context, the encoder may move to CPU, while the projected
visual tokens and language model remain on the language device.

Vision training is staged: projector, last four ViT blocks, upper twelve
blocks, and only then the full encoder at a low learning rate. Frozen encoder
outputs are cached by image digest for repeated long-context chunks.

## Commands

Download the real vision checkpoint outside Git:

```powershell
python scripts/download_internvit.py
```

The TIPSv2 alternative uses a ranged/resumable downloader:

```powershell
python scripts/download_tipsv2.py --output data/raw/tipsv2-l14
python scripts/verify_vision_checkpoint.py data/raw/tipsv2-l14
```

Initialize the real multimodal graft and save a safetensors/JSON resume
checkpoint:

```powershell
python scripts/graft_nemotron.py --stage vision_projector --smoke-prompt "Read the screenshot and identify the failing test." --smoke-image .\sample.png
```

For an attention-only initialization when the vision artifact is not yet
available:

```powershell
python scripts/graft_nemotron.py --without-vision --stage attention_initialized
```

Initialization is not attention recovery. The next permitted expensive stage
is reconstruction against the untouched Nemotron teacher, followed by the
95%-retention gate.

## Cache budget evidence

The analytical estimator can be run before a long-context job:

```powershell
python scripts/report_cache_budget.py --output reports/runtime/cache_budget.json
```

With the V1 defaults (INT8 index, 64-dimensional index heads, BF16
compressed K/V), the cache state is approximately 0.033 GiB at 4K, 0.245 GiB
at 32K, 0.971 GiB at 128K, 7.753 GiB at 1M, and 11.090 GiB at 1.5M. These
figures exclude model weights, activations, and allocator reserve; therefore
1M and 1.5M require host-resident compressed state on an 8 GiB card. They are
planning estimates, not measured throughput or quality results.
