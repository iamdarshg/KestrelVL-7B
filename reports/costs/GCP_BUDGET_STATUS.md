# GCP budget status

The hard cumulative budget for this architecture-selection phase is **$30.00**.
The ledger counts all prior experiments that belong to the same causal
architecture program, including failed or non-selective runs.

| Experiment | GPU | GPU-hours | Conservative estimate | Status |
|---|---:|---:|---:|---|
| T4 startup attempt | Tesla T4 | unavailable | $0.20 | failed before training |
| T4 partial screen | Tesla T4 | unavailable | $1.00 | 20 steps; ranking invalid |
| L4 single option | L4 | 9.954 | $4.2200 | completed pipeline demo, non-selective |
| **Cumulative** |  |  | **$5.4200** |  |

The conservative remaining envelope is **$24.57996**. The T4 entries use their
authorized caps because exact billing is not available. The L4 entry uses the
recorded 9.953966 GPU-hours and the committed reference rate of $0.423956/hour.
New runs must pass `scripts/check_gcp_budget.py` with a projected cost plus a
15% uncertainty margin before launch. A stopped VM or billing delay is not
treated as free compute.

The earlier L4 run cannot select an architecture: it evaluated one candidate and
used the synthetic `generic-code-v2` stream. It remains valuable only as a
pipeline/throughput record.
