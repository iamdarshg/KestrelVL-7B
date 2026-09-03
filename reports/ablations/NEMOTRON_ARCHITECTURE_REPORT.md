# Real Nemotron architecture selection

## Status

No causal Nemotron architecture winner is selected in this phase. The prior
single-candidate L4 run was a real-weight pipeline demonstration, but its
`generic-code-v2` corpus was synthetic and therefore cannot support a language
quality or architecture-selection claim. The historical BERT and tiny-model
results remain proxy evidence only.

The governed real stream and teacher-relative evaluation are now implemented.
The remaining content gate is explicit: Stack-Edu/Stack-v2 content must be
resolved through an approved licensed resolver, RefineCode must provide usable
text rather than metadata alone, and the docs/history JSONL exports must be
provided with license and revision metadata.

## Candidate protocol

The controlled causal family is recorded in
`configs/ablations/nemotron_architecture_screen.yaml`: four mHC-on candidates
for a cheap equal-token screen, followed by two equal-token finalists and one
matched mHC-off control if the cumulative budget permits. Every candidate must
use the same frozen real validation/recovery splits, tokenizer, seed, precision,
optimizer, trainable-layer policy, and initialization policy.

Selection order is held-out real-data teacher-relative quality, robustness,
long-context memory implications, VRAM, and throughput. A candidate is not a
winner because it is cheaper or faster.

## Current evidence boundary

`reports/ablations/nemotron_architecture_results.json` is intentionally
`not_run` with `winner: null`. No architecture freeze ADR is created. The
production defaults remain configurable and provisional until real causal
validation and recovery metrics exist.
