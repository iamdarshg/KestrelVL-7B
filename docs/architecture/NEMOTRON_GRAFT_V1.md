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

## Vision graft

The real wrapper accepts the actual `OpenGVLab/InternViT-300M-448px-V2_5`
checkpoint through `--vision-model-id`. RGB input is dynamically tiled at
448px, normalized using the InternViT/CLIP channel statistics, encoded once,
and sent through a 1024-to-3584 adaptive MLP/resampler projector. Developer
screens retain larger budgets than ordinary images. For frozen vision at
long context, the encoder may move to CPU, while the projected visual tokens
and language model remain on the language device.

Vision training is staged: projector, last four ViT blocks, upper twelve
blocks, and only then the full encoder at a low learning rate. Frozen encoder
outputs are cached by image digest for repeated long-context chunks.

## Commands

Download the real vision checkpoint outside Git:

```powershell
python scripts/download_internvit.py
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
