#!/bin/bash
# GCP L4 CED smoke-test startup script (attempt 2: robust interpreter discovery).
# Unattended; always ends with VM shutdown (GCP --max-run-duration +
# --instance-termination-action=DELETE is the hard guard).
echo "=== ced L4 smoke start $(date -u +%FT%TZ) ==="

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
command -v python3; command -v python; command -v pip; command -v pip3
nvidia-smi --query-gpu=name,memory.total --format=csv 2>&1 | head -3

PY=""
if command -v python3 >/dev/null 2>&1; then PY=python3
elif command -v python >/dev/null 2>&1; then PY=python
else echo "FATAL: no python interpreter"; exit 10
fi
echo "using interpreter: $PY"
$PY --version
$PY -c "import torch; print('torch', torch.__version__, 'cuda=', torch.cuda.is_available())"

echo "--- pip upgrades ---"
$PY -m pip install -q -U transformers hf_transfer safetensors huggingface_hub accelerate 2>&1 | tail -2
$PY -c "import transformers; print('transformers', transformers.__version__)"

cat > /tmp/ced_l4_smoke.py <<PYEOF
PLACEHOLDER
PYEOF
echo "--- running smoke ---"
if timeout 3000 $PY /tmp/ced_l4_smoke.py; then
  echo "smoke exit: 0 PASS"
else
  echo "smoke exit: $? FAIL -- holding VM 10 min for log collection, then shutdown"
  dmesg 2>/dev/null | tail -20 || true
  $PY -m pip list 2>/dev/null | grep -i -E "torch|transformers|safetensors|huggingface|numpy" || true
  sleep 600
fi
