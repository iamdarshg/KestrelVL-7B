#!/usr/bin/env bash
set -Eeuo pipefail

LOG=/var/log/kestrel-t4-ablation.log
exec > >(tee -a "$LOG") 2>&1

PROJECT_DIR=/opt/kestrel
REPO_URL="https://github.com/iamdarshg/KestrelVL-7B.git"
REPO_COMMIT="28a4ede"
DEADLINE_EPOCH="${KESTREL_DEADLINE_EPOCH:-}"
if [[ -z "$DEADLINE_EPOCH" ]]; then
  DEADLINE_EPOCH="$(curl -fsS -H 'Metadata-Flavor: Google' \
    http://metadata.google.internal/computeMetadata/v1/instance/attributes/KESTREL_DEADLINE_EPOCH \
    2>/dev/null || true)"
fi
DEADLINE_EPOCH="${DEADLINE_EPOCH:-0}"
REPORT_DIR="$PROJECT_DIR/reports/ablations"

timestamp() { date --iso-8601=seconds; }
echo "[$(timestamp)] startup begin host=$(hostname) deadline=${DEADLINE_EPOCH}"

if [[ "$DEADLINE_EPOCH" != 0 && "$(date +%s)" -ge "$DEADLINE_EPOCH" ]]; then
  echo "[$(timestamp)] deadline reached before startup work"
  exit 124
fi

rm -rf "$PROJECT_DIR"
git clone --filter=blob:none "$REPO_URL" "$PROJECT_DIR"
git -C "$PROJECT_DIR" checkout "$REPO_COMMIT"
cd "$PROJECT_DIR"
python3 -m pip install --no-cache-dir -e .

mkdir -p "$REPORT_DIR"
cat > "$REPORT_DIR/gcp_t4_runtime.json" <<EOF
{
  "started_at": "$(timestamp)",
  "host": "$(hostname)",
  "gpu": "$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true)",
  "commit": "$(git rev-parse HEAD)",
  "global_token_budget": 10000000,
  "candidate_count": 4,
  "budget_usd": 0.20,
  "deadline_epoch": ${DEADLINE_EPOCH:-0}
}
EOF

if [[ "$DEADLINE_EPOCH" != 0 ]]; then
  REMAINING=$((DEADLINE_EPOCH - $(date +%s)))
else
  REMAINING=720
fi
if (( REMAINING < 60 )); then
  echo "[$(timestamp)] insufficient time remains before deadline: ${REMAINING}s"
  exit 124
fi

export HF_HOME=/opt/huggingface
export TRANSFORMERS_CACHE=/opt/huggingface/transformers
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

echo "[$(timestamp)] gpu diagnostics"
nvidia-smi || true
python3 - <<'PY'
import torch
print({"torch": torch.__version__, "cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available()})
if torch.cuda.is_available():
    print({"device": torch.cuda.get_device_name(0), "capability": torch.cuda.get_device_capability(0)})
PY

# The outer timeout is deliberately shorter than the VM lifetime.  It prevents
# a stalled HF download or model construction from consuming the whole cap.
RUN_SECONDS=$((REMAINING - 90))
if (( RUN_SECONDS < 60 )); then RUN_SECONDS=60; fi
echo "[$(timestamp)] starting real Nemotron ablation with ${RUN_SECONDS}s run budget"
set +e
timeout --signal=TERM --kill-after=20s "${RUN_SECONDS}s" \
  python3 scripts/run_real_ablations.py \
    --model-id nvidia/OpenReasoning-Nemotron-7B \
    --device cuda \
    --total-tokens 10000000 \
    --sequence-length 1024 \
    --max-rss-gib 0 \
    --checkpoint-interval 1000 \
    --max-checkpoints 1 \
    --gradient-checkpointing \
    --output "$REPORT_DIR/gcp_t4_ablation_results.json" \
    --checkpoint-root checkpoints/gcp_t4_real_ablations \
    --local-gpu-hourly-cost 0.35
RUN_STATUS=$?
set -e
echo "[$(timestamp)] ablation exit=${RUN_STATUS}"

python3 - <<PY
import json
from pathlib import Path
p = Path("$REPORT_DIR/gcp_t4_runtime.json")
data = json.loads(p.read_text())
data.update({"finished_at": __import__("datetime").datetime.now().astimezone().isoformat(), "exit_code": $RUN_STATUS})
p.write_text(json.dumps(data, indent=2) + "\n")
PY
exit "$RUN_STATUS"
