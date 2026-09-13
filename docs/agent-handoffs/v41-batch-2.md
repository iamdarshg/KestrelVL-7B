# V4.1 Batch 2 handoff — CSA2 / indexer / single-pass mHC / stage harness (#5, #6, #7, #9)

## 1. Commit range

Starting SHA: `3fb7cee` (L4 real-checkpoint smoke PASS, #3 unblocked).
Ending SHA: this batch's commit (see §2). One ledger repair included
(first ledger line held two concatenated JSON objects; split, no data changed).

## 2. Commits created this batch

- Code + tests: `SinglePassMHC` backend, `mhc_backend` selector,
  5-stage schedule + `TrainingRunState`, 3 new test files
  (47 new tests), ledger fix + $0 CPU entry.
- Prior in-range commits (already on `main`, not mine): `15ef3ae` (#5 CSA2 +
  #6 indexer CPU), `343e649` (Batch 2 CPU addendum), `1fa77c1`–`3fb7cee`
  (L4 smoke attempts 1–7, PASS on attempt 7).

## 3. Issue status

- #5 CSA2 Full/Reindex/Reuse: PASS (verified, not re-implemented).
  `src/model/csa2.py`: `CSA2Config` (configurable cadence),
  `topk_deterministic`, `CSA2Layer` (chunked Full, pool-bound Reindex,
  zero-scoring Reuse), `dense_reference_attention`,
  `CSA2Stack` (+`all_full_mode`, snapshot/resume). Old CSA/HCA retained
  (`src/model/csa.py`, `hca.py`, `attention/` untouched).
- #6 hierarchical indexer: PASS (verified). `src/model/sparse_indexer.py`:
  coarse pool → fine Top-K, deterministic tie-break, bf16/int8 modes,
  chunked scoring, telemetry (pool, Top-K, bytes/token, recall vs dense).
- #7 Single-Pass mHC: PASS (implemented this session). Selectable backend,
  old pair retained for A/B, NOT default. Evidence: exact agreement with
  sequential pair (see §6).
- #9 harness: READY for Batch 3 (no big run executed). Loss terms
  CE/KL/hidden/attn/KV/index all configurable; 5 trainable stages;
  resume-safe accounting (§7/§8).
- #3/#4: intact — zero-gate baseline re-proven by combined tests (§10).
- #1 bounded long-context: preserved — no dense T×T on the long path
  (chunked Full, pool-bound Reindex, zero-score Reuse; tests enforce).

## 4. CSA2 cadence / config

`CSA2Config(top_k, candidate_pool_size, cadence, num_decoder_layers,
head_dim, dtype)`. Default cadence `["full","reuse","reuse","reindex"]`
cyclic per layer — configurable, not hard-coded. Combined tests pin
`["full","reuse","reuse","reindex"]`: telemetry `positions_scored`
`[S, 0, 0, pool]`. `all_full_mode()` = migration reference.

## 5. Indexer interfaces / defaults

`IndexerConfig(candidate_pool_size=2048, top_k=256, chunk_size=512,
index_dtype="bfloat16", coarse_dim=64, deterministic=True)`.
`HierarchicalIndexer(mem_dim, query_dim, config)`:
`build_coarse_index(mem) -> CoarseIndex` (bf16 codes or int8 + scales);
`retrieve(query, mem_k, mem_v, source="semantic"|"detail") ->
(k_sel, v_sel, indices, telemetry)` with `recall_vs_dense` auto-computed
on tiny inputs. Invariant holds: fine stage scores `pool_size`, never `S`.

## 6. mHC interfaces / config

`src/model/attention/mhc_singlepass.py`:
`SinglePassMHC(streams, sinkhorn_iters, enabled)` with
`from_sequential(attn_mhc, mlp_mhc)`, `matrices()`, `coefficients()`
(alpha/beta/gamma closed form), `forward(base, attn, mlp)`,
`freeze()/unfreeze()`, `state_dict_snapshot()/metadata()/
load_snapshot_strict()`, `build_single_pass_from_pair`,
`mixed_attn_state`, `resolve_mhc_backend`.
`KestrelConfig.mhc_backend: "residual"` (default) | `"single_pass"`.
`RealDecoderLayer`: residual path byte-identical by default; fused path
when `mhc_backend="single_pass"` (+`promote_to_single_pass()` copies live
params but does NOT reroute — routing is explicit only).
Fused math: `out = α·base + β·attn + γ·mlp`, one mixing application;
x1 still materialised once (MLP input depends on it). No per-token state
beyond the two S×S logit pairs (same count as the replaced pair).

## 7. Distillation loss API

Unchanged from Batch 1: `DistillLossConfig(w_ce=1.0, w_kl=1.0,
w_hidden=0.5, w_attn=0.5, w_kv=0.25, w_index=0.25, temperature=1.0,
selected_layers=(4,8,12,16,20))`; `compute_losses(...) -> (losses,
log_dict)` with per-term `_active` flags; `measure_recovery`,
`bridge_memory_stats`, `parameter_accounting`, `BudgetTracker` (50h
hard-stop), `write_evidence_report`. All in `src/training/ced_distill.py`.

## 8. Training-stage API

`STAGE_ORDER_5 = ("stage_1",…,"stage_5")`; `STAGE_POLICIES_5[stage]` →
`{encoder_frozen, decoder_frozen, bridge_trainable,
memory_gates_trainable, global_kv_trainable, indexer_trainable, lora,
upper_encoder_layers_unfrozen, selective_fullrank}`:
1 bridge+gates → 2 +globalKV+indexer → 3 +decoder LoRA →
4 +2 upper encoder layers (decoder stays frozen) →
5 narrow selective inherited full-rank (opt-in; no `full_unfreeze` key
exists anywhere). `FreezePolicy.stage_policy_5(stage)`; legacy
`stage_policy()` names untouched. `trainable_groups_5(policy)`.
`TrainingRunState`: `record(tokens, wall_seconds, gpu_seconds,
gcp_cost_usd)` accumulates; `gpu_hours`; strictly-forward
`advance_stage`; `capture_rng/restore_rng` (CPU RNG); `set_checkpoint`,
`set_data_cursor`; `to_dict/from_dict`, `save/load` JSON, `summary()`
(includes stage's trainable groups).

## 9. Batch 3 launch commands

```bash
python -m pytest tests/test_csa2.py tests/test_sparse_indexer.py \
  tests/test_ced_runtime.py tests/test_encoder_bridge.py \
  tests/test_ced_distill.py tests/test_ced_distill_stages.py \
  tests/test_ced_combined.py tests/test_mhc_singlepass.py -q
gcloud compute instances list   # must show no paid GPU VM before/after
git diff --check
```

## 10. Tests / results

Full suite **222 passed, 0 failed** (~17 s CPU). New: 20 mHC
(init/reference agreement, delta measurement, grad flow, freeze,
snapshot guard, coefficient math, param-count parity, prefix stability,
CED-hook interaction, layer routing/A-B, promotion explicitness);
19 stage/accounting (monotonic groups, forward-only stages, RNG
roundtrip, save/load); 8 combined (zero-gate incl. T=256, shapes +
telemetry, chunked-vs-materialised equality, reuse ignores 1000× memory
perturbation, indexer→reindex interop, open-gate fwd/bwd with grads to
hook + bridge, all-full reference). `ruff check` clean on all touched
files; `git diff --check` clean.

## 11. Performance measurements

No GPU benchmarks this batch (per strategy: CPU correctness first; no
>$0.10 run needed). CPU reference facts: reuse layers score 0 positions;
reindex bounded by pool (test: 16 of 64); full never materialises
[B,T,S] unless `materialized_for_test_only=True`. A100/L4 kernel + dtype
bench is Batch 3 work after approval.

## 12. GCP spend

This session: **$0.00 estimated** (no VM launched; ledger entry
`batch2-cpu-mhc-singlepass-stages5-combined-integration`, CPU-only).
Cumulative ledger (10 prior entries + this one): prior L4 smoke attempts
≈ **$0.92 estimated actual** (attempts 1–7: 0.01+0.01+0.06+0.16+0.26+
0.12+0.30; PASS wall 703 s). Billing-confirmed: n/a (no export).
`gcp-cost-summary.md` not yet updated with attempt-7 totals — Batch 3
should roll the $0.92 into the summary table.

## 13. Unresolved blockers

1. GPU dtype/kernel validation (BF16 Sinkhorn, chunked CSA2 kernels,
   indexer int8 vs bf16 recall at scale) — needs approved A100/L4 run.
2. 50 h adaptation run untouched (Batch 3, needs >$0.10 approval).
3. `multimodal_model.py` mHC pair left on residual path (no fused
   wiring) — intentional; promote only on A/B evidence.
4. Dirty worktree has unrelated modifications from other work (see
   `git status`); this batch stages/commits ONLY its own files.

## 14. First recommended Batch 3 action

Request approval for a ≤$0.10-bounded micro GPU run (spot L4):
load pinned 2B/9B revs, run `test_ced_combined.py` + mHC A/B agreement +
indexer recall-vs-dense at S≥4K, log to ledger; on PASS, plan the staged
50 h adaptation (`TrainingRunState` per stage, `BudgetTracker` hard-stop).
