# DeepSeek V4-Flash to Nemotron-7B mapping

This mapping is the executable V1 contract. It preserves the Nemotron dense
body and changes only the self-attention mixer; it does not import DeepSeek's
hidden size, FFN, vocabulary, or layer count.

| Component | DeepSeek V4-Flash reference | Kestrel/Nemotron V1 | Rationale |
| --- | ---: | ---: | --- |
| decoder layers | reference-dependent | 28 | Preserve the real Nemotron body |
| hidden size | 4096 in saved reference config | 3584 | Preserve Qwen/Nemotron residual geometry |
| query heads | reference-specific | 28 | Matches Nemotron attention head count |
| KV heads | shared compressed K=V path | 1 target; 2 ablation | Lowest state cost, with a recovery fallback |
| head dimension | 512 in reference config | 128 | Matches Nemotron Q/K/V projection geometry |
| Q low-rank path | latent Q projection | 512 | Initial SVD rank scaled to 3584 hidden size |
| partial RoPE | 64/512 reference ratio | 16/128 | Preserves approximately 1/8 rotary fraction |
| compressed RoPE theta | 160000 | 160000 | Keep reference positional frequency |
| position target | 1M+ | 1,048,576 | Principal validation target |
| local window | 128 | 128 | Explicit bounded local cache |
| CSA ratio | 4 | 4 | V1 reference constant |
| HCA ratio | 128 | 128; 64 ablation | Aggressive long-context state compression |
| index dimension | 128 | 64; 128 ablation | Lower retrieval state at Nemotron scale |
| index top-k | 512 | 256 default; 64/128/512 ablations | Quality/memory trade-off |
| candidate chunk | implementation-defined | 64 | Bounds temporary score tensor |
| retrieval/query block | implementation-defined | 512 | Avoids dense `Q x M` allocation |
| grouped output | grouped low-rank output | 4 groups, rank 384 | Divisor of 28 is not required by padded implementation |
| mHC | later stabilization option | enabled in V1, ablated | Sinkhorn residual mixing is isolated from attention |

The indexer returns gather-safe `int64` indices because PyTorch gather requires
that index type. `index_dtype: int8` in deployment configuration describes the
local candidate/delta representation budget; it must not be interpreted as an
int8 absolute index for a 1M-token stream. The optimized kernel can store
chunk-local offsets in int8/uint8 and widen only at the final gather boundary.

The reference implementation uses append-only compressed chunks and scores
candidate subchunks of 64. A dense compressed history may be materialized only
by an explicit inspection/compatibility method, never by the long-context
attention path.
