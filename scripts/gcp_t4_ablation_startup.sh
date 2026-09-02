#!/usr/bin/env bash
set -Eeuo pipefail

LOG=/var/log/kestrel-t4-ablation.log
exec > >(tee -a "$LOG") 2>&1

PROJECT_DIR=/opt/kestrel
REPO_URL="https://github.com/iamdarshg/KestrelVL-7B.git"
REPO_COMMIT="4e6b011"
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
python3 -m pip install --no-cache-dir \
  'transformers>=4.47' 'safetensors>=0.4' 'pyyaml>=6.0' \
  'pillow>=10.0' 'numpy>=1.26' 'tqdm>=4.66' 'psutil>=5.9' \
  'bitsandbytes>=0.43' 'accelerate>=0.34'
# The image ships a torchaudio binary compiled against a different torch
# release.  Kestrel does not use audio; remove that incompatible optional
# package so Transformers can import the Qwen2 model cleanly.
python3 -m pip uninstall -y torchaudio || true

mkdir -p "$REPORT_DIR"
cat > "$REPORT_DIR/gcp_t4_runtime.json" <<EOF
{
  "started_at": "$(timestamp)",
  "host": "$(hostname)",
  "gpu": "$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true)",
  "commit": "$(git rev-parse HEAD)",
  "global_token_budget": 10000000,
  "candidate_count": 4,
  "budget_usd": 1.00,
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

# Prefer the fully materialized HF snapshot when the image cache already has
# it.  This selects Kestrel's bounded streaming loader and avoids rebuilding
# dense Q/K/V/O workspaces for every candidate.  Fall back to the Hub ID on a
# cold VM so the bootstrap remains self-contained.
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

# The outer timeout is deliberately shorter than the VM lifetime.  It prevents
# a stalled HF download or model construction from consuming the whole cap.
RUN_SECONDS=$((REMAINING - 90))
if (( RUN_SECONDS < 60 )); then RUN_SECONDS=60; fi
echo "[$(timestamp)] starting real Nemotron ablation with ${RUN_SECONDS}s run budget"
set +e
timeout --signal=TERM --kill-after=20s "${RUN_SECONDS}s" \
  python3 scripts/run_real_ablations.py \
    --model-id "$NEMOTRON_MODEL_ID" \
    --device cuda \
    --total-tokens 10000000 \
    --sequence-length 1024 \
    --max-rss-gib 0 \
    --checkpoint-interval 1000 \
    --max-checkpoints 1 \
    --gradient-checkpointing \
    --training-logit-stride 16 \
    --trainable-layer-start 24 \
    --muon-ns-steps 1 \
    --output "$REPORT_DIR/gcp_t4_ablation_results.json" \
          --checkpoint-root checkpoints/gcp_t4_real_ablations \
          --init-cache checkpoints/gcp_t4_real_ablations/svd_init.pt \
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
