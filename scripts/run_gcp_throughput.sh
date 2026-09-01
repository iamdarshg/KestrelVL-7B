#!/usr/bin/env bash
set -euo pipefail

# Quick, real-model GCP measurement.  The default is a 10-step train-mode
# workload; pass --mode forward for a prefill-only measurement.  The launcher
# does not create a VM or upload data.  Run it after bootstrap_gcp.sh on the
# selected guest with the model artifacts already available locally/GCS.
PROFILE="configs/hardware/gcp_single_l4.yaml"
WORLD_SIZE="1"

while (($#)); do
  case "$1" in
    --profile)
      PROFILE="$2"
      shift 2
      ;;
    --world-size)
      WORLD_SIZE="$2"
      shift 2
      ;;
    *)
      break
      ;;
  esac
done

if [[ "$WORLD_SIZE" == "2" ]]; then
  exec torchrun --standalone --nproc_per_node=2 scripts/benchmark_gcp_throughput.py \
    --profile "$PROFILE" "$@"
fi
exec python scripts/benchmark_gcp_throughput.py --profile "$PROFILE" "$@"
