# BERT-targeted local ablation report

Status: complete on 2026-09-01.

This is a proxy screen for the residual-mixer decision. It uses the real
`bert-base-uncased` weights and a deterministic masked-language-model target;
it is not a causal Nemotron/V4-Flash quality result.

## Fixed protocol

- Global processed-token budget: **10,000,000 tokens total**.
- Four candidates: **2,500,000 tokens each**.
- Per candidate: 2,400,000 training tokens plus 100,000 validation tokens.
- Same composition-locked `kestrel-final-cpt-v1` stream for every candidate.
- Sequence length: 512 in the runner.
- Top four BERT encoder layers trainable; lexical embedding and MLM decoder frozen.
- Muon for matrix parameters; minimal AdamW for vector/scalar parameters.
- One retained best checkpoint; losing checkpoints were pruned.

The corpus configuration records a source-block length of 256 and a nominal
sequence length of 1024. The executed ablation command explicitly used
`--sequence-length 512`; the runner report is authoritative for the executed
sequence length.

## Results

Lower validation loss is better.

| Candidate | Attention | mHC | Validation loss | Perplexity | Train time | GPU-hours | Peak VRAM |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `bert_global_mhc` | full bidirectional | yes | 0.0862550754 | 1.09008435 | 5600.60 s | 1.55572 | 0.81159 GiB |
| `bert_local128_mhc` | bidirectional local 128 | yes | **0.0347777526** | **1.03538957** | **2920.56 s** | **0.81127** | 0.81146 GiB |
| `bert_global_no_mhc` | full bidirectional | no | 0.0761201004 | 1.07909217 | 4204.97 s | 1.16805 | 0.81047 GiB |
| `bert_local128_no_mhc` | bidirectional local 128 | no | 0.0685610180 | 1.07096597 | 3790.95 s | 1.05304 | 0.81303 GiB |

The exact machine-readable outputs are
[`bert_ablation_results.json`](bert_ablation_results.json) and
[`bert_target_selection.json`](bert_target_selection.json).

## Decision

The BERT screen selects **local-128 attention with mHC enabled**. Relative to
the strongest no-mHC candidate, its validation loss is lower by approximately
54.4%; relative to global+mHC, it is lower by approximately 59.7%.

For Kestrel, this is sufficient evidence to keep mHC enabled in the V1
reference path and to retain a bounded local branch. It does **not** justify
replacing the causal Nemotron schedule with pure bidirectional local
attention. The Kestrel translation remains:

- layers 0 and 1: causal sliding-window attention of width 128;
- subsequent layers: CSA/HCA alternation with compressed retrieval;
- mHC: enabled around the attention and feed-forward residual updates;
- one shared compressed K/V representation as the deployment target;
- `index_topk=256` and HCA ratio 128 as the current quality/memory defaults,
  pending a real Nemotron-targeted screen.

The earlier tiny Kestrel proxy in `final_architecture.json` is retained as
historical evidence and is not overwritten by this BERT result. The BERT
target is encoder-only, has different masking and representation geometry,
and cannot settle the causal-specific choices of KV-head count, HCA ratio, or
Lightning-Indexer top-k.

## Checkpoint and restart note

The winning checkpoint remains locally at
`checkpoints/bert_ablations_best/01_bert_local128_mhc` and is intentionally not
committed because it is a large generated binary. Candidate 1 resumed from a
valid earlier checkpoint after a disk-capacity interruption; its final held-out
validation result is valid, but its uninterrupted training-loss trajectory is
not reconstructed. This does not affect the fixed-budget comparison or the
winner selection, which is based on held-out validation loss.

This report is a proxy architecture gate. It is not evidence of full Kestrel
Nemotron reasoning retention, 1M-context support, RTX 4060 Q4 acceptance, or
any public benchmark win.
