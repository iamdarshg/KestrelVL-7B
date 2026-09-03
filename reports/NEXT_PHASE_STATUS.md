# Kestrel next-phase status

Date: 2026-09-04  
Repository HEAD after implementation: `6b53caa`  
Open GitHub issue: #1 remains open.

## 1. What changed

- Added `data.real_corpus.RealStreamingCorpus`, a real-only streaming layer
  with pinned source identity, license/provenance fields, repository-stable
  split assignment, exact/normalized/MinHash-style duplicate filtering,
  post-split deterministic FIM, and checkpointable source/token cursors.
- Added `configs/data/real_corpus.yaml` preserving the required 35/25/20/10/10
  source composition. Metadata-only or missing sources fail closed.
- Added canonical corpus fingerprint generation and a committed manifest
  fingerprint for the current source contract.
- Added teacher-relative recovery diagnostics for NLL, delta NLL, forward and
  reverse KL, selected-layer cosine/reconstruction error, finite checks, and
  mHC manifold diagnostics.
- Added `reports/reconstruction_gate.json` as an explicit blocked gate; no
  unsupported 95% retention claim is made.
- Added the controlled causal screen protocol and a no-winner architecture
  report.
- Added the canonical cumulative GCP ledger and fail-closed budget checker.
- Extended the tiny reference model to expose hidden states, enabling a local
  recovery instrumentation test without changing its default behavior.

## 2. What we learned

The current local suite passes (`44 passed`). The identical tiny teacher/candidate
recovery smoke produces finite outputs, zero delta NLL/KL, and cosine 1.0. That
validates metric instrumentation, not Nemotron recovery.

The final local preflight also passes compilation, targeted lint, the L4 profile
dry-run, and the RTX PRO 6000 profile dry-run. The preflight remains overall
`fail` only because the real multimodal graft smoke cannot find the expected
TIPSv2 projector checkpoint pointer; this preserves the existing 1.3 GiB host-RSS
evidence boundary rather than weakening it.

The committed L4 run recorded about 9.954 GPU-hours and $4.2200 at its reference
rate, but it was one candidate on the synthetic `generic-code-v2` stream. The
two prior T4 attempts are counted conservatively at $0.20 and $1.00; neither
produced a valid ranking. Total conservative related spend is therefore
$5.4200, leaving $24.5800 before the $30 ceiling.

The corpus source contract cannot yet feed a paid run safely: official
Stack-Edu/Stack-v2-style indexes and the public RefineCode metadata artifact do
not by themselves provide all governed text content, while docs/history exports
are absent locally. This is a useful stop: paying to train against a synthetic
fallback would repeat the known-invalid experiment.

## 3. Current Kestrel V1 architecture

Not frozen. The implementation remains configurable with the existing V4-Flash
inspired defaults: local window 128, CSA ratio 4, HCA ratio 128, partial RoPE,
shared KV target 1, index top-k 256, grouped low-rank output, and mHC enabled.
These are implementation defaults, not evidence-backed causal Nemotron choices.
The screen protocol compares KV 1/2, HCA 128/64, top-k 64/128/256, and a matched
mHC-off control.

## 4. Teacher recovery

- Teacher baseline: not measured on the frozen real recovery split yet.
- Initial graft recovery: not measured on the frozen real recovery split yet.
- Post-training recovery: not run.
- >=95% gate: **not passed**; `reports/reconstruction_gate.json` is blocked.

The local identical-model diagnostic is not a teacher-retention score.

## 5. GCP spending

| Experiment | GPU | GPU-hours | Estimated cost | Cumulative |
|---|---|---:|---:|---:|
| T4 startup attempt | Tesla T4 Spot | unavailable | $0.20 | $0.20 |
| T4 partial screen | Tesla T4 Spot | unavailable | $1.00 | $1.20 |
| L4 single real-option pipeline demo | L4 Spot | 9.954 | $4.2200 | $5.4200 |

Total conservative estimate: **$5.4200**. Remaining: **$24.5800**. T4 failed
attempts use their authorized caps because billing detail is unavailable. L4
uses the committed 9.953966 GPU-hours at $0.423956/hour. New launches must
pass `scripts/check_gcp_budget.py` with a 15% uncertainty margin.

## 6. Failures

- The first T4 attempt failed before useful training.
- The second T4 run reached only 20 steps of one candidate; candidates 2--4
  were not reached, so scores and ranking are invalid.
- The L4 run completed its pipeline objective but is non-selective because it
  used synthetic data and one candidate.
- Real multimodal local validation remains subject to the existing 1.3 GiB RSS
  gate; that gate was not lowered.

## 7. Evidence boundaries

Kestrel can currently claim a passing local correctness suite, a tested
reference attention/mHC/cache implementation, a deterministic governed-corpus
contract, a working recovery-metric instrument, and a real Nemotron single-option
pipeline demonstration. It cannot claim a causal architecture winner, 95%
teacher retention, language-quality improvement, public benchmark wins,
multimodal quality, or 1M-token training/inference performance.

## 8. Exact next step

Provide or configure licensed text resolvers for all five source categories,
materialize the frozen architecture/recovery manifests locally, and run the
local real-data teacher-relative preflight. Only if that passes should Stage A
of `configs/ablations/nemotron_architecture_screen.yaml` be launched.

## 9. Ready-to-run command

After the real content gate passes and a fresh ledger check permits it:

```powershell
python scripts/check_gcp_budget.py --hours 3 --hourly-rate 0.423956
python scripts/run_real_ablations.py --device cuda --corpus-backend real --corpus-config configs/data/real_corpus.yaml --total-tokens 1000000 --candidate-index 0
```

This command is intentionally not run in this phase because the configured
production sources currently fail closed before training. The GCP launcher also
enforces the cumulative ledger on the VM.
