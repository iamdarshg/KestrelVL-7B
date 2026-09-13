#!/bin/bash
# GCP L4 micro-validation startup (Batch 3): real-9B numerics for #5/#6/#7.
# Unattended; always ends with VM shutdown (GCP --max-run-duration +
# --instance-termination-action=DELETE is the hard guard).
# Payload = this repo @ main HEAD (repo is public; exact SHA recorded into
# evidence). Launch ONLY after the Batch-3 prep commits are pushed.
echo "=== ced L4 micro-validate start $(date -u +%FT%TZ) ==="

finish() {
  echo "=== shutdown rc=$? $(date -u +%FT%TZ) ==="
  shutdown -h now || poweroff || true
}
trap finish EXIT

export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export TOKENIZERS_PARALLELISM=false

echo "--- env markers ---"
whoami; pwd
command -v python3; command -v python; command -v pip; command -v git
nvidia-smi --query-gpu=name,memory.total --format=csv 2>&1 | head -3

PY=""
if command -v python3 >/dev/null 2>&1; then PY=python3
elif command -v python >/dev/null 2>&1; then PY=python
else echo "FATAL: no python interpreter"; exit 10
fi
echo "using interpreter: $PY"
$PY --version
$PY -c "import torch; print('torch', torch.__version__, 'cuda=', torch.cuda.is_available())"

echo "--- pip (NO -U: never touch the image torch stack) ---"
$PY -m pip install -q transformers hf_transfer safetensors huggingface_hub accelerate 2>&1 | tail -2
$PY -c "import transformers; print('transformers', transformers.__version__)"
echo "--- remove broken torchaudio (image ships torch 2.9.1 + torchaudio 2.11 ABI mismatch; text-only run does not need it) ---"
$PY -m pip uninstall -y torchaudio 2>&1 | tail -1 || true
echo "--- preflight: torch stack must import cleanly before any download ---"
$PY -c "import torch, transformers, accelerate; assert torch.cuda.is_available(); print('preflight OK', torch.__version__)" || { echo "FATAL: preflight failed"; exit 11; }

echo "--- clone repo @ main (exact SHA recorded to evidence) ---"
cd /tmp
rm -rf KestrelVL-7B
git clone --depth 1 --branch main https://github.com/iamdarshg/KestrelVL-7B.git || { echo "FATAL: clone failed"; exit 12; }
cd KestrelVL-7B
REPO_SHA=$(git rev-parse HEAD)
echo "repo_sha=$REPO_SHA"
export PYTHONPATH=/tmp/KestrelVL-7B/src
$PY -c "import model.csa2, model.sparse_indexer, model.attention.mhc_singlepass; print('repo imports OK')" || { echo "FATAL: repo imports failed"; exit 13; }

echo "--- running micro validation (real 9B, S=4096) ---"
if timeout 3000 $PY scripts/ced_l4_micro_validate.py --mode real --out /var/log/micro-results.json; then
  echo "micro exit: 0 PASS repo_sha=$REPO_SHA"
else
  echo "micro exit: $? FAIL repo_sha=$REPO_SHA -- holding VM 10 min for log collection, then shutdown"
  dmesg 2>/dev/null | tail -20 || true
  $PY -m pip list 2>/dev/null | grep -i -E "torch|transformers|safetensors|huggingface|numpy" || true
  sleep 600
fi
