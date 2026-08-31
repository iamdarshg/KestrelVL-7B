from dataclasses import dataclass


@dataclass
class SFTConfig:
    sequence_length: int = 16384
    learning_rate: float = 1e-5
    frontier_fraction: float = 0.20
    executable_fraction: float = 0.25

