#!/usr/bin/env bash
set -euo pipefail
python scripts/run_stage.py --stage attention_initialized --config configs/hardware/gcp_dual_l4.yaml "$@"

