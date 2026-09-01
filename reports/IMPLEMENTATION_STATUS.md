# Implementation status

The repository was bootstrapped on 2026-08-31 from an empty worktree.

The current implementation includes immutable source configuration copies and
Luna targets, a Nemotron-sized V4-Flash-inspired reference attention stack with
partial RoPE, sinks, shared KV, configurable index top-k, grouped low-rank
output and mHC, a bounded local/compressed cache, chunked index retrieval,
explicit long-context execution modes, a multimodal model shell with cached
InternViT CPU-offload policy, data governance utilities, safetensors resume and
Q4 release tooling, resumable stage tooling, a GCP budget guard, a local 4060
profile, and correctness tests.

This file does not claim completed full-scale training, benchmark scores,
frontier parity, 1M-token throughput, or a Q4 artifact. Those require actual
weights, data access, measured runs, and held-out evaluation.

The real local ablation runner uses the actual
`nvidia/OpenReasoning-Nemotron-7B` checkpoint in NF4, a global 10M-token
budget across four candidates (2.5M each), composition-locked train/validation
streams, Muon for matrix parameters, and a minimal AdamW vector bucket. Its
output is valid only after the run completes and the four held-out validation
losses are recorded.
