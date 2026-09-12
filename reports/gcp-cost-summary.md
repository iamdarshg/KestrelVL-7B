# Batch 1 — GCP cost summary (cost-governance worker)

Retrieved: 2026-09-12 (UTC). No VM launched. No spend. Billing-confirmed: n/a.

## Pricing table (US, cheapest region us-central1)

| Config | GPU | On-demand (USD/hr, full node) | Spot | Source |
|---|---|---|---|---|
| `a2-highgpu-1g` | 1x A100 40GB | $3.67 (range $3.67–$4.33 across 10 US regions) | Variable daily; Spot VMs up to 91% off on-demand incl. GPUs (exact rate on Spot VMs pricing page) | [GCP GPU pricing](https://cloud.google.com/compute/gpus-pricing) (A2 taxonomy) + [GCP GPU instances survey, rev. 2026-09-01](https://www.thundercompute.com/blog/google-cloud-gpu-instances) + [A100 pricing, rev. 2026-09-04](https://www.thundercompute.com/blog/nvidia-a100-pricing) |
| `a2-ultragpu-1g` | 1x A100 80GB | $5.03 (range $5.03–$6.04 across 10 US regions) | Same spot policy as above | Same sources |

Conclusion: cheapest adequate A100 config on GCP US is **`a2-highgpu-1g` (1x A100 40GB) at $3.67/hr on-demand in us-central1** (80GB only if VRAM requires it: `a2-ultragpu-1g` at $5.03/hr). Spot is cheaper but preemptible and rate varies — quote the Spot VMs pricing page at launch time.

## Batch-1 spend

- Estimated: **$0.00**. No VM launched by this worker; no paid runs authorized.
- Billing-confirmed: **n/a** (quote-only; pricing assumptions are not an invoice).
- `gcloud compute instances list` (2026-09-12): one instance, `kestrel-checkpoint-transfer` (e2-micro, us-central1-a), status **TERMINATED** — zero running compute. (Terminated VM's residual persistent disk, if any, is a console-billing check, not a Batch-1 charge.)

## Project totals

- Batch 1 total: $0.00 estimated / billing-confirmed n/a.
- Ledger: `reports/gcp-cost-ledger.jsonl` (append-only; this batch added the `batch1-pricing-survey` entry).

## Policy reminder

- **Batch 1 is CPU-only. Zero paid runs** unless a **<$0.10 smoke test** is explicitly justified and approved.
- Any future GPU launch MUST use auto-terminate flags: `--max-run-duration=<DURATION>` + `--instance-termination-action=DELETE` (min 30s, max 120d). Confirmed supported per [Limit the run time of a VM](https://docs.cloud.google.com/compute/docs/instances/limit-vm-runtime).
- gcloud: available locally (Google Cloud SDK 582.0.0); no components installed by this worker.
