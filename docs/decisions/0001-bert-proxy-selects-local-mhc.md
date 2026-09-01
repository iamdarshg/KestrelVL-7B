# Decision 0001: transfer local attention plus mHC into the Nemotron graft

Status: accepted for the next Nemotron graft stage.

The completed local BERT proxy screen used the real `bert-base-uncased`
weights and exactly 10M processed tokens globally. Four candidates received
the same composition-locked stream: full versus bidirectional local-128
attention crossed with mHC enabled versus disabled. The winner was
`bert_local128_mhc`, with held-out MLM validation loss `0.0347777526` and
perplexity `1.03538957`.

This result is a transfer signal, not a Nemotron benchmark result. BERT is an
encoder with bidirectional masking and a different representation geometry.
It cannot select the causal model's KV-head count, HCA ratio, compressed
retrieval top-k, or long-context positional behavior.

## Consequence

The causal Kestrel configuration keeps mHC enabled around attention and MLP
residual updates and retains a bounded local branch. The V4-style causal
schedule is therefore:

1. layers 0 and 1 use causal sliding attention of width 128;
2. later layers alternate CSA and HCA;
3. CSA uses ratio 4 and HCA uses ratio 128;
4. one shared compressed K/V representation and `index_topk=256` remain the
   current deployment defaults pending the real Nemotron screen.

The full table, exact JSON, checkpoint/restart caveat, and comparison
arithmetic are in
[`reports/ablations/BERT_ABLATION_REPORT.md`](../../reports/ablations/BERT_ABLATION_REPORT.md).

## Rejection criteria

This decision must be revisited if the real Nemotron attention-recovery gate
shows less than 95% teacher retention, if mHC causes non-finite or unstable
training, or if the local branch increases long-context task error after the
causal CSA/HCA branch is trained. A BERT result alone cannot override those
gates.
