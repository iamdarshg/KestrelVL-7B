# Implementation status

The repository was bootstrapped on 2026-08-31 from an empty worktree.

The current implementation includes immutable source configuration copies and
Luna targets, a Nemotron-sized V4-Flash-inspired reference attention stack with
partial RoPE, sinks, shared KV, configurable index top-k, grouped low-rank
output and mHC, a bounded local/compressed cache, append-only INT8/BF16 index
state with chunked retrieval,
explicit long-context execution modes, a multimodal model shell with cached
InternViT/TIPSv2 CPU-offload policy, data governance utilities, safetensors resume and
Q4 release tooling, resumable stage tooling, a GCP budget guard, a local 4060
profile, and correctness tests.

This file does not claim completed full-scale training, benchmark scores,
frontier parity, 1M-token throughput, or a Q4 artifact. Those require actual
weights, data access, measured runs, and held-out evaluation.

The reference long-context runner now has an explicit inference-only
``cache_device="cpu"`` policy. At the default INT8/64-dimensional index, the
analytical retained-cache estimate is 7.753 GiB at 1,048,576 tokens and 11.090
GiB at 1.5M, before weights or activations. The implementation consequently
keeps local K/V on the accelerator and transfers compressed candidate chunks
from host RAM; no 1M throughput claim is implied.

The real local ablation runner uses the actual
`nvidia/OpenReasoning-Nemotron-7B` checkpoint in NF4, a global 10M-token
budget across four candidates (2.5M each), composition-locked train/validation
streams, Muon for matrix parameters, and a minimal AdamW vector bucket. Its
output is valid only after the run completes and the four held-out validation
losses are recorded.

The local vision tree now contains a selectable TIPSv2 L/14 adapter in addition
to the InternViT reference. Its custom output is normalized into the common
patch-token contract, and the shipped no-normalization processor contract is
preserved. A complete TIPSv2 weight file and the image smoke script are still
required before using it for multimodal training.

The long-context training audit is now explicit: full-recompute optimizer
steps use checkpointed tensor cache boundaries and backward replay, while
stateful-truncated steps keep their deliberate detach boundaries. The real
Nemotron 10M-token local ablation was attempted on 2026-09-03 but was stopped
before candidate training because the hard 1.3 GiB host-RSS ceiling was
exceeded during first-layer construction. It produced no valid score or
checkpoint and remains blocked pending a permitted higher-memory execution
environment.
