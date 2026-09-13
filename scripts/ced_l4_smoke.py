"""L4 real-checkpoint smoke test (pinned Qwen3.5 revs). CPU-free, GPU-only, unattended.

Steps: verify arch/tokenizer contract from HF metadata -> sequential-load
2B-Base (bf16) -> short forward, no-NaN, param accounting -> unload ->
load 9B (bf16) -> short forward + greedy 8-token generate + peak VRAM ->
zero-gate identity check on REAL 4096-d hidden states -> write JSON results.

Any failure raises loudly; the caller (startup script) traps output so partial
evidence is still recorded. This is a smoke test, NOT distillation (#9).
"""

from __future__ import annotations

import hashlib
import json
import sys
import time

ENCODER_ID = "Qwen/Qwen3.5-2B-Base"
ENCODER_REV = "b1485b2fa6dfa1287294f269f5fb618e03d52d7c"
DECODER_ID = "Qwen/Qwen3.5-9B"
DECODER_REV = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"

EXPECTED = {
    # TOKENIZER_VOCAB (len(tokenizer)) != EMBEDDING_VOCAB (padded LM-head rows).
    "encoder": {"hidden": 2048, "layers": 24, "tok_vocab": 248077, "emb_vocab": 248320},
    "decoder": {"hidden": 4096, "layers": 32, "tok_vocab": 248077, "emb_vocab": 248320},
}
SPECIAL_IDS = {"endoftext": 248044, "im_start": 248045, "im_end": 248046}
PROMPT = "The capital of France is"
N_GENERATE = 8


def _sha_sample(tensor, n: int = 1 << 20) -> str:
    flat = tensor.detach().cpu().float().reshape(-1)[:n]
    return hashlib.sha256(flat.numpy().tobytes()).hexdigest()[:16]


def check_tokenizer(tok, side: str, exp_tok_vocab: int) -> dict:
    vocab = len(tok)
    if vocab != exp_tok_vocab:
        raise ValueError(f"{side}: tokenizer vocab {vocab} != {exp_tok_vocab}")
    dec = tok.added_tokens_decoder
    for name, tid in SPECIAL_IDS.items():
        entry = dec.get(tid)
        content = entry.content if hasattr(entry, "content") else entry
        if content is None or name not in str(content):
            raise ValueError(f"{side}: special id {tid} is not {name!r} (got {content!r})")
    found = {tok.convert_ids_to_tokens(i) for i in SPECIAL_IDS.values()}
    return {"vocab_size": vocab, "special_ids_ok": True, "special_tokens": sorted(found)}


def smoke_model(model_id: str, rev: str, kind: str, do_generate: bool) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    exp = EXPECTED[kind]
    tok = AutoTokenizer.from_pretrained(model_id, revision=rev, trust_remote_code=False)
    out: dict = {"tokenizer": check_tokenizer(tok, kind, exp["tok_vocab"])}
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=rev, dtype=torch.bfloat16,
        device_map="cuda:0", trust_remote_code=False,
    ).eval()
    cfg = model.config
    text_cfg = getattr(cfg, "text_config", cfg) or cfg
    hidden = int(getattr(text_cfg, "hidden_size"))
    layers = int(getattr(text_cfg, "num_hidden_layers"))
    emb_vocab = int(getattr(text_cfg, "vocab_size"))
    if hidden != exp["hidden"] or layers != exp["layers"]:
        raise ValueError(f"{kind}: arch {hidden}/{layers} != {exp}")
    if emb_vocab != exp["emb_vocab"]:
        raise ValueError(f"{kind}: embedding vocab {emb_vocab} != {exp['emb_vocab']}")
    out["emb_vocab"] = emb_vocab
    total = sum(p.numel() for p in model.parameters())
    out.update({"hidden": hidden, "layers": layers, "params": total,
                "load_s": round(time.time() - t0, 1)})
    ids = tok(PROMPT, return_tensors="pt").input_ids.cuda()
    t1 = time.time()
    with torch.no_grad():
        logits = model(ids).logits.float()
    if not torch.isfinite(logits).all():
        raise ValueError(f"{kind}: non-finite logits")
    out["forward_s"] = round(time.time() - t1, 2)
    out["logits_shape"] = list(logits.shape)
    out["logit_sample"] = [round(float(v), 4) for v in logits[0, -1, :5]]
    out["lm_head_sha"] = _sha_sample(model.get_output_embeddings().weight)
    gen_text, gen_ids, tok_s = "", [], 0.0
    if do_generate:
        torch.cuda.reset_peak_memory_stats()
        t2 = time.time()
        with torch.no_grad():
            gen = model.generate(ids, max_new_tokens=N_GENERATE, do_sample=False,
                                 eos_token_id=248046, pad_token_id=248044)
        tok_s = time.time() - t2
        gen_ids = gen[0, ids.shape[1]:].tolist()
        gen_text = tok.decode(gen[0, ids.shape[1]:])
    out["generated_ids"] = gen_ids
    out["generated_text"] = gen_text
    out["gen_s"] = round(tok_s, 2)
    out["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    # Zero-gate identity on REAL hidden states: h + 0.0 * mem == h exactly.
    with torch.no_grad():
        h = model.model(ids)[0].float() if hasattr(model, "model") else logits
        mem = torch.randn_like(h)
        gated = h + 0.0 * mem
        out["zero_gate_max_abs_diff"] = float((gated - h).abs().max().item())
    if out["zero_gate_max_abs_diff"] != 0.0:
        raise ValueError("zero-gate identity violated on real hidden states")
    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return out


def main() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA on smoke VM")
    results: dict = {"gpu": torch.cuda.get_device_name(0), "steps": {}}
    t0 = time.time()
    results["steps"]["encoder_2b"] = smoke_model(ENCODER_ID, ENCODER_REV, "encoder", False)
    results["steps"]["decoder_9b"] = smoke_model(DECODER_ID, DECODER_REV, "decoder", True)
    results["wall_s"] = round(time.time() - t0, 1)
    results["status"] = "PASS"
    print("SMOKE_RESULT_JSON=" + json.dumps(results))
    with open("/var/log/smoke-results.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=1, sort_keys=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # fail loudly but leave evidence
        print("SMOKE_RESULT_JSON=" + json.dumps({"status": "FAIL", "error": str(exc)[:500]}))
        sys.exit(1)
