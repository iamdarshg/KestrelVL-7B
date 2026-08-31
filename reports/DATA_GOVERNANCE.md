# Data governance contract

Every training example is represented by `SampleRecord` in
`src/data/manifests.py`. The record carries source/source ID, license,
teacher/provenance confidence, quality, duplicate group, and contamination
status. `data/provenance.sqlite` is the intended durable index; a JSONL export
is supported for streaming environments.

The contamination filter rejects exact normalized matches and explicit held-out
benchmark IDs. Near-duplicate and fork groups are retained in the index and
can be excluded at mixture-build time. Evaluation data is never used as a
fallback training source.

Synthetic CoderVision/CoderInk records retain the renderer seed, source text
hash, task label, and augmentation parameters so a result can be audited.
Frontier-labelled trajectories are low-volume until teacher provenance and
content quality have been verified.

