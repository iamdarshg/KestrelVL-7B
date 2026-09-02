# Full-recompute audit

Date: 2026-09-03

The old `run_chunked_forward(..., mode="full_recompute", optimizer=...)`
implementation retained one autograd loss graph per chunk and called
`backward()` only after the final chunk. That was chunked execution, but not
bounded activation recomputation.

The implementation now uses explicit tensor cache boundaries and
`torch.utils.checkpoint` around every chunk transition. Each checkpoint
reconstructs a fresh `KestrelCache` from the previous boundary, runs the
chunk, and returns the new local/compressed/index tensors plus a non-tensor
layout descriptor. Backward replays the chunk functions; mutable cache objects
from the original forward pass are never reused during replay.

The tiny regression test observed 4 forward chunks and 8 checkpoint calls
(forward plus backward replay), with `retained_loss_graphs=0` and a finite
optimizer step. The complete local suite passed 39/39 after this change.

The cache state itself is intentionally still append-only for compressed
history. This bounds activation graphs, not the required long-context memory;
the existing CPU compressed-state policy remains inference-only. A real
Nemotron 1M backward measurement still requires the authorized RTX PRO 6000
run.
