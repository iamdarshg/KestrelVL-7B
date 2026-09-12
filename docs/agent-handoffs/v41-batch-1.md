# V4.1 Batch 1 handoff — CED foundation (#2 epic, #3/#4 + #9 skeleton)

## 1. Current commit SHA

`8ac6f9c` (on `main`; parent `8cad27b`, then upstream `4025770`).

## 2. Commits created this batch

- `8cad27b` — `feat(ced)` runtime + bridge: `ced_config.py`, `ced.py`,
  `encoder_bridge.py`, `test_ced_runtime.py` (25 tests),
  `test_encoder_bridge.py` (23 tests). Issues #3, #4.
- `8ac6f9c` — `feat(distill)` skeleton: `ced_distill.py`,
  `qwen35_ced.yaml`, `test_ced_distill.py` (20 tests), cost
  ledger + summary. Issue #9 (instrumentation only).

Pre-existing dirty worktree files were left untouched (not staged/committed).

## 3. Issue status

- #3 (heterogeneous CED runtime): all acceptance criteria PASS on CPU with
  tiny doubles. Real-weight load is via strict state-dict path + revision
  pinning; real `Qwen3.5-2B/9B` checkpoints were NOT downloaded (no GPU spend).
  Blocker for full acceptance: needs one real-checkpoint load + smoke test.
- #4 (EncoderMemoryBridge): PASS on CPU (shapes, zero-gate, reuse,
  serialization, dual taps, dense/sparse stubs for #5/#6).
- #9 (distillation harness): measurement skeleton only — PASS. No training
  stages run; 50 A100h budget tracker implemented, never exercised on GPU.
- #2 epic boxes remain open pending real-checkpoint + CSA2/indexer work.

## 4. Files added (no existing files modified)

- `src/model/ced_config.py` — contract: IDs, 2048/24L, 4096/32L, vocab
  248320, `CEDSourceConfig.validate_architecture/require_pinned_revisions/
  check_tokenizer_compatibility`, `assert_no_hidden_size_splice`.
- `src/model/ced.py` — `TinyCausalEncoder`, `TinyAutoregressiveDecoder`
  (strict `load_source_state`, checksums), `ExternalMemoryHook` (gate=0.0,
  bit-exact identity when disabled), `CEDRuntime` (encode/decode phases,
  independent freeze, `trainable_parameter_counts`,
  `assert_hidden_contract`, `validate_sources`).
- `src/model/encoder_bridge.py` — `EncoderMemory` (fingerprint,
  `to_dict/from_dict`), `EncoderMemoryBridge` (RMSNorm→Linear→SwiGLU
  adapter→K/V proj, `output_gate`=0.0), `build_semantic/build_detail/
  build_memory`, `reuse_for_layers` (shared storage), `blank_memory_like`,
  `state_dict_snapshot/metadata/load_snapshot_strict`, `dense_reference`,
  `sparse_memory_stub(top_k)`. `splice_into_decoder_states` always raises.
- `src/training/ced_distill.py` — `DistillLossConfig`, `compute_losses`
  (per-term + log dict), `measure_recovery`, `bridge_memory_stats`,
  `parameter_accounting`, `FreezePolicy.stage_policy`
  (bridge_only/bridge_global_kv/lora_recovery/selective_unfreeze),
  `BudgetTracker` (50h default, hard-stop, save/load),
  `write_evidence_report`.
- `configs/model/qwen35_ced.yaml` — source IDs (revisions unpinned, must pin
  before real run), arch metadata, loss weights, freeze stages, budget.
- Tests: `tests/test_ced_runtime.py`, `tests/test_encoder_bridge.py`,
  `tests/test_ced_distill.py`.
- `reports/gcp-cost-ledger.jsonl`, `reports/gcp-cost-summary.md`.

## 5. Public APIs for Batch 2

```python
from model.ced_config import CEDSourceConfig  # validate_architecture(), require_pinned_revisions()
from model.ced import CEDRuntime, ExternalMemoryHook  # encode_prompt(ids)->PromptEncoding; decode_with/without_memory(ids,...)
from model.encoder_bridge import EncoderMemoryBridge, EncoderMemory  # build_memory(sem, detail)->EncoderMemory; reuse_for_layers(mem, n)
from training.ced_distill import compute_losses, measure_recovery, FreezePolicy, BudgetTracker, write_evidence_report
```

Wire-up: `enc(ids)` → `bridge.build_semantic(h)` [+`build_detail`] →
`bridge.build_memory(...)` → pass `EncoderMemory.global_k/v` to
`hook.forward(dec_hidden, memory)` / `rt.decode_with_memory`.
Under pytest, `src/` is on `sys.path` (`pythonpath=["src"]`); note
`ced.py` uses absolute `from model.ced_config` while `encoder_bridge.py`
uses relative imports — both resolve under pytest.

## 6. Tests run and results

- `tests/test_ced_runtime.py` + `test_encoder_bridge.py` +
  `test_ced_distill.py`: **68 passed** (5.6 s, CPU).
- Rest of suite (existing): **69 passed**. Total **137 passed**, 0 failed.
- Cross-module interop script: memory shape (1,8,32), shared `data_ptr`
  across 4 layer views, recovery `max_abs_diff=0.0`, `within_tol=True`,
  deterministic fingerprints. OK.
- `ruff check` on all 7 new py files: clean. `git diff --check`: clean.

## 7. Benchmark / recovery numbers

- Zero-gate recovery (tiny doubles, CPU): `max_abs_diff=0.0`,
  `mean_abs_diff=0.0`, KL≈−3e−08 (fp noise), `within_tol=True` at 1e−5.
- Bridge at init: `output_gate=0.0` → all memory K/V exactly zero.
- Real-model logit tolerance / throughput / VRAM: NOT measured (no GPU run).

## 8. GCP spend

- Estimated Batch-1 spend: **$0.00**. Billing-confirmed: n/a (no billing
  export configured; nothing to reconcile).
- Pricing survey 2026-09-12 (ledger entry `batch1-pricing-survey`):
  cheapest adequate A100 = `a2-highgpu-1g` (1×A100 40GB) $3.67/h on-demand
  us-central1; 80GB `a2-ultragpu-1g` $5.03/h; spot up to ~91% off (quote at
  launch). Sources in `reports/gcp-cost-summary.md`.
- `gcloud` present (SDK 582.0.0); `instances list` showed only a TERMINATED
  e2-micro — no running paid VM. No VM was created this batch.

## 9. Architectural decisions

- Tiny synthetic doubles mirror the real hidden/layer contract instead of
  downloading 2B/9B weights: scores verified-requirements per dollar.
- Decoder-side gate (`ExternalMemoryHook`) owned by `ced.py`; bridge owns
  only K/V memory production + its own `output_gate`. No hidden-state splice
  exists anywhere (bridge exposes a raising stub to prove it).
- Shared global K/V via storage-sharing views, not per-layer copies.
- `ced_distill.py` duck-types tensors (no top-level CED imports) so the
  three components evolve independently.

## 10. Rejected approaches

- Downloading real Qwen3.5 checkpoints on CPU/GPU this batch: rejected —
  unnecessary spend; strict-load path + pinning gives reproducibility without
  weights. Left for Batch 2's approved smoke test.
- Weaker-test substitution to dodge the $0.10 gate: rejected per policy;
  no GPU run was genuinely needed.
- Global unfreeze default: rejected — `FreezePolicy` starts
  encoder+decoder frozen, matching #9's progressive schedule.

## 11. Unresolved blockers

1. Real `Qwen/Qwen3.5-2B-Base` + `Qwen/Qwen3.5-9B` revisions unpinned in
   `qwen35_ced.yaml` (`encoder_revision`/`decoder_revision` empty) —
   `validate_sources(strict_revisions=True)` will fail loudly until pinned.
2. Real-checkpoint load + BF16/precision smoke + A100 smoke test not run
   (needs >$0.10 approval: ~$3.67/h × max runtime).
3. `measure_recovery` KL can print tiny negative values (~−3e−08) from fp
   noise — clamp if it pollutes reports.

## 12. Recommended first action for Batch 2

Pin real revisions in `configs/model/qwen35_ced.yaml`, then request GCP
approval for a spot `a2-highgpu-1g` smoke test (load both checkpoints,
`validate_sources`, zero-gate logit comparison vs standalone 9B decoder,
record to ledger); on PASS, proceed to #5 (CSA2) against the
`dense_reference`/`sparse_memory_stub` interfaces.

## 13. Batch 2 CPU addendum (commit `15ef3ae`, no GPU, $0)

- Revisions PINNED (`qwen35_ced.yaml`): 2B `b1485b2f...`, 9B `c2022362...`
  (HF Hub metadata 2026-09-13). Real `text_config` verified against contract:
  2048/24L (8Q/2KV, head_dim 256) and 4096/32L (16Q/4KV, head_dim 256),
  vocab 248320, max_pos 262144 both. Special-token IDs identical both sides
  (endoftext 248044, im_start 248045, im_end 248046); only the DEFAULT eos
  string differs (Base→endoftext, post-trained→im_end). Policy: generation
  uses decoder eos 248046. `CEDSourceConfig.check_tokenizer_compatibility`
  passes on IDs.
- #5 DONE (CPU): `src/model/csa2.py` — `CSA2Config` (configurable cadence,
  default full→reuse→reuse→reindex), `topk_deterministic`,
  `CSA2Layer` (chunked scoring, reindex pool-bound, reuse zero-scoring),
  `dense_reference_attention`, `CSA2Stack` (+`all_full_mode`, snapshot/
  resume). 22 tests. Note: sparse weights are softmax-over-Top-K, so
  full-vs-dense agreement holds exactly at top_k == S (test-pinned).
- #6 DONE (CPU): `src/model/sparse_indexer.py` — `IndexerConfig`
  (pool 2048, top_k 256, chunked, bf16|int8), `HierarchicalIndexer`
  (`build_coarse_index`/`retrieve`), `dense_reference_topk`,
  `recall_at_k`, `synthetic_copy_task`/`span_recall`. 15 tests.
- `ced_distill.py`: KL clamped at 0.0 (was −3e−08 fp noise).
- Full-chain CPU interop: bridge→indexer (recall_vs_dense 1.0)→CSA2 stack
  modes [full, reuse, reindex, reuse], scored [32, 0, 16, 0], reuse replays
  full exactly. Suite: **174 passed** (105 new + 69 existing). Ruff + diff
  check clean. Ledger appended (`batch2-cpu-csa2-indexer-revpin`, $0).
- Remaining spine: #7 (mHC upgrade), #8 (optional Engram), #9 training
  stages, #10 (InternViT + regression gates) — all need the GPU smoke test
  first. BLOCKED on GCP approval (see §8 rates).
