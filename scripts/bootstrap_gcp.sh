#!/usr/bin/env bash
set -euo pipefail

# Explicit preflight only. VM creation is intentionally separate and requires
# a user-supplied project/zone and --confirm-budget in the caller.
MAX_BUDGET_USD="${MAX_BUDGET_USD:-90}"
if [[ "${MAX_BUDGET_USD}" == *.* ]]; then
  budget_int="${MAX_BUDGET_USD%.*}"
else
  budget_int="${MAX_BUDGET_USD}"
fi
if (( budget_int >= 100 )); then
  echo "Refusing budget ${MAX_BUDGET_USD}: must stay below 100 USD" >&2
  exit 2
fi
echo "GCP preflight passed; no VM created. budget=${MAX_BUDGET_USD} USD"

