# Real corpus content gate

The corpus contract is valid and fingerprinted as
`a93a924f16aca3a845c18b0661cb815c66f6babaf3244d7843fe068efb0b4fc7`.

The five category weights remain locked at 35/25/20/10/10. The current source
identities are explicit and pinned in `configs/data/real_corpus.yaml`:

- `HuggingFaceTB/stack-edu`, Python split, with the declared Software Heritage
  content resolver;
- `OpenCoder-LLM/RefineCode-code-corpus-meta`, with an explicit content-resolver
  requirement because the public artifact is metadata-oriented;
- `bigcode/the-stack-v2-dedup`, Python split, with the declared Software
  Heritage content resolver;
- local, licensed `docs.jsonl` technical/documentation export;
- local, licensed `history.jsonl` commit/diff export.

The content gate is **not passed** on this checkout. The local docs/history
exports are absent, and metadata-only rows are rejected unless their declared
resolver supplies text and a usable license. Run
`python scripts/validate_real_corpus.py --require-local-content` after supplying
the approved exports. This check performs no download and no paid work.

No architecture ablation may use `CompositionLockedCorpus` after this gate;
that stream remains a synthetic test fixture only.
