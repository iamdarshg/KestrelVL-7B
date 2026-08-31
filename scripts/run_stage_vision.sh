#!/usr/bin/env bash
set -euo pipefail
python scripts/run_stage.py --stage vision_projector --config configs/hardware/gcp_dual_l4.yaml "$@"

