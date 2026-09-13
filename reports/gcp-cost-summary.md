# Batch 1+2 — GCP cost summary

Retrieved: 2026-09-12/13 (UTC). 7 L4 smoke VMs launched (6 FAIL, 1 PASS); feature work CPU-only. Billing-confirmed: n/a.

## Pricing table (us-central1, Linux, full-node USD/hr)

| Config | GPU | On-demand | Spot (surveyed) | Source |
|---|---|---|---|---|
| `a2-highgpu-1g` | 1x A100 40GB | $3.67 | variable daily, ≤ on-demand; ~$1.10 (ThunderCompute) | [GCP GPU pricing](https://cloud.google.com/compute/gpus-pricing) + [survey 2026-09-01](https://www.thundercompute.com/blog/google-cloud-gpu-instances) |
| `a2-ultragpu-1g` | 1x A100 80GB | $5.03 | same policy | same |
| **`g2-standard-4` (CHEAPEST adequate)** | **1x L4 24GB** | **$0.7068** | **≤ on-demand by construction; surveyed ~$0.31 (DevZero) / $0.6481 (Holori 2026-09-06)** | [Holori](https://calculator.holori.com/gcp/vm/g2-standard-4) + [DevZero](https://www.devzero.io/instances/gpu/l4) + [usage.ai 2026-09-07](https://www.usage.ai/blogs/gcp/compute-engine/) |
| `g2-standard-8` (fallback) | 1x L4 24GB | $0.8536 | ~$0.7362 (Holori) | Holori 2026-09-02 |
| N1 + T4 (REJECTED) | 1x T4 16GB | ~$0.54 total | — | ThunderCompute — rejected: T4 has no BF16, and 16GB is tight even for sequential int8 loads |

Conclusion (2026-09-13): cheapest GCP option that can run the BF16 smoke test is
**spot `g2-standard-4` (1×L4 24GB, 4 vCPU, 16GB RAM) in us-central1-a** —
~5× cheaper than the cheapest A100. Fit: sequential load (2B ≈4GB validate→unload,
then 9B ≈18GB bf16 + short-prompt activations < 24GB); 100GB boot PD for weight
staging via the repo's streaming shard reader. `g2-standard-8` is the named
fallback only if 16GB host RAM OOMs (requires a fresh approval). Quota verified
2026-09-13: `PREEMPTIBLE_NVIDIA_L4_GPUS` 1.0/0.0 + `NVIDIA_L4_GPUS` 1.0/0.0 in
us-central1, project `project-ba289c9c-3c25-4e15-9dc`. G2 available in
us-central1-a per [GPU locations](https://docs.cloud.google.com/compute/docs/regions-zones/gpu-regions-zones).

## Smoke-test cost estimate (proposed, NOT launched)

Plan: download pinned revs (~22GB ingress free, ~10–20 min) → checksums →
sequential load → `validate_sources` → short-prompt forward + zero-gate logit
compare vs standalone 9B → delete VM. Realistic 30–45 min; hard cap 1h.

| Component | Math | USD |
|---|---|---|
| VM (spot g2-standard-4, conservative ceiling = on-demand) | $0.7068 × 1h | $0.71 |
| 100GB balanced PD | $0.04/GB-mo ÷ 730 × 1h | $0.006 |
| Egress | none (results via serial log) | $0.00 |
| Uncertainty (repo `conservative_cost`, 15%) | ×1.15 | — |
| **Projected MAX** | | **$0.82** |
| **Expected TRUE cost** (spot ~$0.31–0.65/h × ~0.5–0.75h) | | **~$0.16–0.49** |

$0.10 gate: cannot be met honestly — even spot L4 needs ≥30 min max for a 22GB
download + two model loads. Hence approval is required. True billed cost will be
reconciled into `gcp-cost-ledger.jsonl` (and issues #3/#9) after the run, labeled
`billing-confirmed` only if the Billing API is reachable, else `estimated`.

## Spend to date (updated 2026-09-13, Batch 2 close)

- Estimated actual: **≈$0.92** — L4 smoke attempts 1–7 in us-central1-a
  (`g2-standard-4`, 1×L4 24GB): 0.01 + 0.01 + 0.06 + 0.16 + 0.26 + 0.12 +
  0.30. Attempts 1–6 FAIL (fast guest-terminate ×2, missing accelerate,
  vocab proof, torchaudio ABI ×2); attempt 7 PASS (wall 703 s, pinned
  2B+9B revs bf16, zero-gate 0.0, Paris generation; evidence
  `reports/gcp/ced_l4_smoke_pass.json`). Batch-2 feature work itself
  (CSA2, indexer, mHC, stages, combined tests): **$0.00** (local CPU).
- Billing-confirmed: **n/a** (no billing export configured).
- Ledger: `reports/gcp-cost-ledger.jsonl`, 11 entries, append-only
  (repaired 2026-09-13: first line held two concatenated objects; split,
  no data changed).

## Policy reminder

- **Batch 1 is CPU-only. Zero paid runs** unless a **<$0.10 smoke test** is explicitly justified and approved.
- Any future GPU launch MUST use auto-terminate flags: `--max-run-duration=<DURATION>` + `--instance-termination-action=DELETE` (min 30s, max 120d). Confirmed supported per [Limit the run time of a VM](https://docs.cloud.google.com/compute/docs/instances/limit-vm-runtime).
- gcloud: available locally (Google Cloud SDK 582.0.0); no components installed by this worker.
