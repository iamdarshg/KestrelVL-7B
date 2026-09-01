"""Configuration shared by the reference model and training scripts."""

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class KestrelConfig:
    """Nemotron-sized configuration with explicitly tunable V4-Flash knobs.

    The defaults are intentionally small enough for unit tests. Production
    configs can be loaded from the immutable reference and overridden without
    changing the language body's dimensions.
    """

    vocab_size: int = 152064
    hidden_size: int = 3584
    intermediate_size: int = 18944
    num_hidden_layers: int = 28
    num_attention_heads: int = 28
    num_key_value_heads: int = 1
    head_dim: int = 128
    q_lora_rank: int = 512
    index_head_dim: int = 64
    index_topk: int = 256
    index_dtype: str = "int8"
    candidate_chunk_size: int = 64
    retrieval_chunk_size: int = 512
    attention_query_block: int = 512
    compressed_kv_dtype: str = "bfloat16"
    sliding_window: int = 128
    csa_compression_ratio: int = 4
    hca_compression_ratio: int = 128
    partial_rotary_fraction: float = 1 / 8
    rope_theta: float = 1_000_000.0
    compress_rope_theta: float = 160_000.0
    max_position_embeddings: int = 1_048_576
    yarn_factor: float = 16.0
    original_max_position_embeddings: int = 65_536
    output_groups: int = 4
    output_rank: int = 384
    mhc_enabled: bool = True
    mhc_streams: int = 2
    mhc_sinkhorn_iters: int = 6
    attention_output_scale_init: float = 0.01
    dropout: float = 0.0
    rms_norm_eps: float = 1e-6
    layer_schedule: list[str] = field(default_factory=list)
    use_vision: bool = True
    vision_hidden_size: int = 1024
    vision_token_budget: int = 1024
    vision_offload_threshold: int = 262_144
    vision_cache_encoded: bool = True
    vision_freeze_long_context: bool = True
    vision_budget_ordinary: int = 512
    vision_budget_document: int = 1024
    vision_budget_ide: int = 2048
    vision_budget_high_resolution: int = 4096

    def __post_init__(self) -> None:
        if not self.layer_schedule:
            self.layer_schedule = ["sliding", "sliding"] + ["csa", "hca"] * 13
        if len(self.layer_schedule) != self.num_hidden_layers:
            raise ValueError("layer_schedule must have one entry per decoder layer")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if self.num_attention_heads * self.head_dim <= 0:
            raise ValueError("attention dimensions must be positive")
        if self.mhc_streams < 1:
            raise ValueError("mhc_streams must be positive")
        if self.attention_output_scale_init < 0:
            raise ValueError("attention_output_scale_init must be non-negative")
        if self.index_topk < 1 or self.candidate_chunk_size < 1 or self.retrieval_chunk_size < 1:
            raise ValueError("retrieval sizes must be positive")
        if self.index_dtype not in {"int8", "int16", "int32", "int64", "bfloat16"}:
            raise ValueError("index_dtype must be bfloat16 or an explicitly supported integer dtype")

    @property
    def rotary_dim(self) -> int:
        return max(2, int(self.head_dim * self.partial_rotary_fraction) // 2 * 2)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def tiny(cls, **overrides: Any) -> "KestrelConfig":
        values: dict[str, Any] = dict(
            vocab_size=257,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=1,
            head_dim=16,
            q_lora_rank=32,
            index_head_dim=8,
            index_topk=8,
            sliding_window=8,
            csa_compression_ratio=4,
            hca_compression_ratio=8,
            output_groups=2,
            output_rank=8,
            mhc_streams=2,
            max_position_embeddings=4096,
            vision_hidden_size=32,
            vision_token_budget=16,
        )
        values.update(overrides)
        if "layer_schedule" not in values:
            values["layer_schedule"] = ["sliding", "sliding", "csa", "hca"]
        return cls(**values)
