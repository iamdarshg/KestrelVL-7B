# KestrelVL-7B

KestrelVL is an approximately 7–8B-class multimodal coding/reasoning
research model. It starts from `nvidia/OpenReasoning-Nemotron-7B`, keeps the
dense Qwen body and tokenizer, and adds an InternViT developer-vision graft
plus a reference-first, V4-Flash-inspired attention transplant. The
repository is licensed under the GNU Affero General Public License v3.0; see
`LICENSE` and `NOTICE` for the important third-party model/data boundaries.

## Status

This repository contains the executable bootstrap and reference
implementation. It includes the attention conversion, mHC ablation, cache,
data governance, resumable stage runner, local 4060 profile, GCP cost guard,
and tests. Expensive training stages are opt-in and must be backed by measured
checkpoints and held-out evaluations. No paid GCP job is started by install or
testing.

The issue-#1 memory/release foundation is now implemented in the reference
path: local K/V is bounded, CSA/HCA state is append-only and chunked, the
Lightning Indexer never needs a dense `Q x M` score tensor, long-context
execution has explicit `full_recompute` and `stateful_truncated` modes, and
production resume/release artifacts have safetensors-only paths. A fused CUDA
kernel, full real-Nemotron 1M forward/backward measurement, and 4060 Q4
quality result remain validation work; the repository does not claim them.

The current first gate is attention correctness and representation recovery.
SFT/RL launchers refuse to proceed unless the recovery gate is present and
passing. Local ablations are architecture/recovery proxies, not public
benchmark claims.

## One-command local smoke inference

```powershell
python -m pip install -e ".[dev]"
python scripts/run_local.py --prompt "Explain this function and suggest a test." --max-new-tokens 64
```

With a local Hugging Face checkpoint, use `--model-path`. A screenshot can be
passed with `--image`; the fallback vision encoder keeps the API usable for
tests, while production InternViT loading uses the immutable reference config.

```powershell
python scripts/run_local.py --model-path .\checkpoints\18_q4_release `
  --prompt "Find the visual bug and propose a minimal patch." --image .\sample.png
```

## Architecture

The initial schedule is `[sliding, sliding, csa, hca, ...]`. CSA uses ratio 4
compression, HCA uses ratio 128, one shared K/V representation, a configurable
Lightning Indexer (`top-k` 64/128/256/512), partial RoPE, sinks, and grouped
low-rank output. mHC is implemented as a Sinkhorn-projected doubly-stochastic
two-stream residual connection and is selected or rejected by the local
ablation artifact rather than by assertion.

The reference path never creates a full million-token attention mask. Query
blocks score compressed keys in bounded chunks. Compressed history is paired
with an append-only Lightning Index state: INT8 projected keys plus BF16 scales
by default, or BF16 projected keys for the quality mode. This proves the
memory contract but still needs a fused kernel before claiming production
1M-token throughput. The cache keeps the final 128 local tokens and append-only
compressed BF16 chunks; `reports/DIMENSION_MAPPING.md` records the dimension
transplant and the distinction between compact local candidate offsets and
gather-safe absolute indices.

```text
tokens ──> Qwen/Nemotron embedding ──> decoder layers ──> LM head
                                      │
                       local sliding or CSA/HCA branch
                                      │
                  shared compressed K=V + Lightning Indexer

image ──> dynamic 448px tiles ──> InternViT-300M ──> adaptive projector
                                                     │
                              visual tokens inserted into the text stream
```

## Local ablation matrix

```powershell
python scripts/run_real_ablations.py --device cuda --total-tokens 10000000
```

The real-weight screen runs four deliberately orthogonal candidates with a
global 10M-token GPU budget: each candidate receives the same 2.5M-token
composition-locked stream, including 2.4M training tokens and 100K held-out
tokens. The candidates cover the intended 1-KV/2-KV, HCA-64/HCA-128,
top-k-128/256/512, and mHC-on/off decisions. The actual Nemotron body is
loaded NF4 and frozen; only the new attention and enabled mHC parameters are
trained. Muon handles matrix parameters; AdamW is limited to vector/scalar
parameters. Checkpoints resume from the exact corpus fingerprint in
`checkpoints/real_ablations/`, while validation loss/perplexity alone selects
`reports/ablations/final_architecture.json`. The result is a local architecture
screen, not a claim about full-scale model quality.

The older `scripts/run_ablations.py` remains a fast tiny correctness proxy and
must not be used for the real architecture choice.

### BERT-targeted ablation

To compare against a real BERT target, run:

```powershell
python scripts/run_bert_ablations.py --device cuda --total-tokens 10000000
```

This is a separate target-model protocol. It downloads/loads the real
`bert-base-uncased` encoder, uses deterministic masked-language-model
corruption, and keeps the same final-corpus source mixture and global 10M
token budget. Each of the four candidates receives 2.5M processed corpus
tokens (2.4M train plus 100K validation). The candidates are full versus
bidirectional local-128 attention, crossed with mHC enabled versus disabled.
The no-mHC candidate is algebraically equivalent to the original BERT
residual computation; mHC is inserted before each attention and feed-forward
LayerNorm. Results are written to
`reports/ablations/bert_ablation_results.json` and the selected candidate to
`reports/ablations/bert_target_selection.json`.

The bounded local profile trains the top four BERT encoder layers by default
(`--trainable-layer-start 8`) and freezes the original lexical embedding/MLM
decoder. This keeps the comparison within the RTX 4060 memory/time envelope
while preserving the real pretrained BERT geometry; pass
`--trainable-layer-start 0` for a slower full-encoder adaptation.

BERT is encoder-only and has a 512-position pretrained geometry, so these MLM
results are not numerically comparable to the causal Nemotron/V4-Flash
ablation. The original real-Nemotron screen remains available through
`scripts/run_real_ablations.py`.

## Required gates and cost limit

1. `pytest -q` must pass causal/sliding masking, compression, indexer, RoPE,
   gradient, and cache tests.
2. `python scripts/benchmark_hardware.py --profile local_4060_8gb` records
   peak VRAM and fails above the configured 7.5 GiB target.
3. A reconstruction checkpoint must report at least 95% frozen-teacher
   retention before SFT/RL is allowed.
4. Every data record carries source/license/provenance and contamination
   fields.
5. Every cloud launch uses the explicit budget guard. The prototype cap is
   $90, below the requested $100 GCP credit ceiling.

## Reproducible stages

`scripts/run_stage.py` implements the milestone names from the specification.
Each run writes configuration, dataset-manifest hash, RNG state, optimizer and
scheduler state when present, progress, hardware telemetry, and a durable-sync
manifest. Real-Nemotron runs use `SafeCheckpointManager`, which stores model,
optimizer, and RNG tensors in safetensors and metadata in JSON. Preemptible
shutdown is handled by signal/checkpoint hooks.

```powershell
python scripts/run_stage.py --stage attention_initialized --config configs/hardware/local_4060_8gb.yaml
python scripts/run_stage.py --stage attention_reconstructed --resume checkpoints/02_attention_reconstructed
```

GCP launchers require `--confirm-budget` and a cap below `$100`. They use
guest-local checkpoints first and optionally sync to a user-selected GCS URI.
Do not put credentials in YAML or source control.

To create a pickle-free release from a standard safetensors state export:

```powershell
python scripts/export_release.py `
  --state-safetensors .\weights\model.safetensors `
  --config-json .\weights\config.json `
  --output .\releases\kestrel-q4 `
  --force
```

The exporter writes groupwise Q4 tensors, native-precision sensitive tensors,
checksums, configuration, and a clean-process load validation. The local
loader uses `load_q4_runtime`: packed linear weights stay packed on the target
device and only the active linear is dequantized for its GEMM. This is a
reference memory-bounded runtime; a fused CUDA dequantize-GEMM kernel is still
needed before claiming production throughput or a measured 4060 VRAM gate.
`load_q4_bundle` remains available as an explicit dense/dequantized inspection
utility and should not be used for full-model deployment.

## References and honest evidence boundary

Immutable, fetched-at-bootstrap model configs are under `references/`; source
URLs, revisions, and SHA-256 digests are recorded in `references/manifest.json`.
Current public GPT-5.6 Luna comparison values are stored in
`benchmarks/luna_targets.json` and are targets only. This repository does not
claim completed full-scale training, benchmark parity, 1M-token throughput, or
a Q4 artifact until those are independently measured on held-out data.
