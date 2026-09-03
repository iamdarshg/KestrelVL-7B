#!/usr/bin/env bash
set -Eeuo pipefail

LOG=/var/log/kestrel-l4-real-option.log
exec > >(tee -a "$LOG") 2>&1

PROJECT_DIR=/opt/kestrel
REPO_URL="https://github.com/iamdarshg/KestrelVL-7B.git"
REPO_COMMIT="10fdd7a"
DEADLINE_EPOCH="${KESTREL_DEADLINE_EPOCH:-}"
BUDGET_USD="10.00"
SPOT_RATE_USD_PER_HOUR="0.423956"
PROJECTED_HOURS="${KESTREL_PROJECTED_HOURS:-12}"

timestamp() { date --iso-8601=seconds; }
echo "[$(timestamp)] startup begin host=$(hostname) deadline=${DEADLINE_EPOCH:-unset} budget=${BUDGET_USD}"

if [[ -z "$DEADLINE_EPOCH" ]]; then
  DEADLINE_EPOCH="$(curl -fsS -H 'Metadata-Flavor: Google' \
    http://metadata.google.internal/computeMetadata/v1/instance/attributes/KESTREL_DEADLINE_EPOCH \
    2>/dev/null || true)"
fi
if [[ -z "$DEADLINE_EPOCH" ]]; then
  # Leave a safety margin below the $10 compute cap at the current reference
  # rate. Spot prices are variable, so this is an accounting guard, not a bill.
  DEADLINE_EPOCH=$(( $(date +%s) + 82800 ))
fi

if [[ "$(date +%s)" -ge "$DEADLINE_EPOCH" ]]; then
  echo "[$(timestamp)] deadline reached before startup work"
  exit 124
fi

rm -rf "$PROJECT_DIR"
git clone --filter=blob:none "$REPO_URL" "$PROJECT_DIR"
git -C "$PROJECT_DIR" checkout "$REPO_COMMIT"
cd "$PROJECT_DIR"
python3 scripts/check_gcp_budget.py \
  --hours "$PROJECTED_HOURS" \
  --hourly-rate "$SPOT_RATE_USD_PER_HOUR" \
  --budget 30.00
python3 -m pip install --no-cache-dir \
  'transformers>=4.47' 'safetensors>=0.4' 'pyyaml>=6.0' \
  'pillow>=10.0' 'numpy>=1.26' 'tqdm>=4.66' 'psutil>=5.9' \
  'bitsandbytes>=0.43' 'accelerate>=0.34'
python3 -m pip uninstall -y torchaudio || true

mkdir -p reports/ablations checkpoints/gcp_l4_real_option
cat > reports/ablations/gcp_l4_runtime.json <<EOF
{
  "started_at": "$(timestamp)",
  "host": "$(hostname)",
  "gpu": "$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true)",
  "commit": "$(git rev-parse HEAD)",
  "global_token_budget": 10000000,
  "candidate": "proxy_winner_kv2_hca128_topk64_no_mhc",
  "candidate_token_budget": 2500000,
  "budget_usd": ${BUDGET_USD},
  "spot_rate_reference_usd_per_hour": ${SPOT_RATE_USD_PER_HOUR},
  "deadline_epoch": ${DEADLINE_EPOCH}
}
EOF

export HF_HOME=/opt/huggingface
export TRANSFORMERS_CACHE=/opt/huggingface/transformers
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

NEMOTRON_MODEL_ID="nvidia/OpenReasoning-Nemotron-7B"
NEMOTRON_SNAPSHOT="$(find "$HF_HOME/hub/models--nvidia--OpenReasoning-Nemotron-7B/snapshots" \
  -mindepth 1 -maxdepth 1 -type d 2>/dev/null | head -n 1 || true)"
if [[ -n "$NEMOTRON_SNAPSHOT" && -f "$NEMOTRON_SNAPSHOT/model.safetensors.index.json" ]]; then
  NEMOTRON_MODEL_ID="$NEMOTRON_SNAPSHOT"
fi

echo "[$(timestamp)] gpu diagnostics"
nvidia-smi || true
python3 - <<'PY'
import torch
print({"torch": torch.__version__, "cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available()})
if torch.cuda.is_available():
    print({"device": torch.cuda.get_device_name(0), "capability": torch.cuda.get_device_capability(0)})
PY

REMAINING=$((DEADLINE_EPOCH - $(date +%s)))
if (( REMAINING < 300 )); then
  echo "[$(timestamp)] insufficient time remains before deadline: ${REMAINING}s"
  exit 124
fi
RUN_SECONDS=$((REMAINING - 120))
echo "[$(timestamp)] starting one real Nemotron candidate with ${RUN_SECONDS}s remaining"
set +e
timeout --signal=TERM --kill-after=30s "${RUN_SECONDS}s" \
  python3 -u scripts/run_real_ablations.py \
    --model-id "$NEMOTRON_MODEL_ID" \
    --device cuda \
    --corpus-backend real \
    --corpus-config configs/data/real_corpus.yaml \
    --total-tokens 10000000 \
    --evaluate-proxy-winner \
    --sequence-length 1024 \
    --max-rss-gib 0 \
    --checkpoint-interval 20 \
    --max-checkpoints 1 \
    --gradient-checkpointing \
    --enable-mhc \
    --output reports/ablations/gcp_l4_real_option_results.json \
    --checkpoint-root checkpoints/gcp_l4_real_option \
    --local-gpu-hourly-cost "$SPOT_RATE_USD_PER_HOUR"
RUN_STATUS=$?
set -e
echo "[$(timestamp)] candidate exit=${RUN_STATUS}"

python3 - <<PY
import json
from datetime import datetime
from pathlib import Path
p = Path("reports/ablations/gcp_l4_runtime.json")
data = json.loads(p.read_text())
data.update({"finished_at": datetime.now().astimezone().isoformat(), "exit_code": ${RUN_STATUS}})
p.write_text(json.dumps(data, indent=2) + "\n")
PY
exit "$RUN_STATUS"
