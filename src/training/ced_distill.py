"""Batch-1 measurement skeleton for CED mechanistic distillation (GitHub issue #9).

Instrumentation only: pure-tensor loss measurement, numerical recovery gates,
bridge memory statistics, freeze policies, compute-budget tracking and
deterministic evidence reports. NO training loops, NO GPU requirements,
NO network access.

This module deliberately does NOT import ``src/model/ced.py``,
``src/model/encoder_bridge.py`` or ``src/model/ced_config.py`` (owned by
other agents). It operates on plain tensors and dicts (duck-typing); any
model integration must use optional lazy imports outside this module.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
import torch.nn.functional as F

__all__ = [
    "DistillLossConfig",
    "compute_losses",
    "measure_recovery",
    "bridge_memory_stats",
    "parameter_accounting",
    "FreezePolicy",
    "BudgetTracker",
    "write_evidence_report",
]

_DEFAULT_SELECTED_LAYERS = (4, 8, 12, 16, 20)


@dataclass
class DistillLossConfig:
    """Per-term distillation loss weights and matching scope.

    Defaults (documented): ``w_ce=1.0``, ``w_kl=1.0``, ``w_hidden=0.5``,
    ``w_attn=0.5``, ``w_kv=0.25``, ``w_index=0.25``, ``temperature=1.0``.
    ``selected_layers`` is a small subset of layers used for hidden/attention
    matching -- never full retention by default.
    """

    w_ce: float = 1.0
    w_kl: float = 1.0
    w_hidden: float = 0.5
    w_attn: float = 0.5
    w_kv: float = 0.25
    w_index: float = 0.25
    temperature: float = 1.0
    selected_layers: tuple | list = field(default_factory=lambda: _DEFAULT_SELECTED_LAYERS)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Raise ValueError on negative weights, bad temperature or bad layers."""
        for name in ("w_ce", "w_kl", "w_hidden", "w_attn", "w_kv", "w_index"):
            if float(getattr(self, name)) < 0:
                raise ValueError(f"distill weight {name} must be >= 0")
        if float(self.temperature) <= 0:
            raise ValueError("temperature must be > 0")
        layers = list(self.selected_layers)
        if not layers:
            raise ValueError("selected_layers must be non-empty")
        if any(not isinstance(i, int) or i < 0 for i in layers):
            raise ValueError("selected_layers must be ints >= 0")


def _mean_mse(matched: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[float, bool]:
    if not matched:
        return 0.0, False
    total = 0.0
    for a, b in matched:
        total += float(F.mse_loss(a.detach().float(), b.detach().float()).item())
    return total / len(matched), True


def _match_layers(
    teacher: Mapping[Any, torch.Tensor] | None,
    student: Mapping[Any, torch.Tensor] | None,
    selected: Iterable[int],
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    if not teacher or not student:
        return []
    matched = []
    for idx in selected:
        if idx in teacher and idx in student:
            matched.append((teacher[idx], student[idx]))
    return matched


_KV_GLOBAL_PAIRS = (("teacher_kv", "student_kv"), ("kv_teacher", "kv_student"))
_INDEX_GLOBAL_PAIRS = (("teacher_index", "student_index"), ("index_teacher", "index_student"))


def _stats_pair_mse(
    memory_stats: Mapping[str, Any] | None,
    selected: Iterable[int],
    kind: str,
) -> tuple[float, bool]:
    """MSE from memory_stats for ``kind`` in {"kv", "index"}, selected-keys only."""
    if not memory_stats:
        return 0.0, False
    per_layer: list[tuple[torch.Tensor, torch.Tensor]] = []
    for idx in selected:
        # Canonical per-layer keys, e.g. teacher_kv_4 / student_kv_4.
        for tk, sk in ((f"teacher_{kind}_{idx}", f"student_{kind}_{idx}"),
                       (f"{kind}_teacher_{idx}", f"{kind}_student_{idx}")):
            if tk in memory_stats and sk in memory_stats:
                per_layer.append((memory_stats[tk], memory_stats[sk]))
                break
    if per_layer:
        pairs = [(torch.as_tensor(a, dtype=torch.float32),
                  torch.as_tensor(b, dtype=torch.float32)) for a, b in per_layer]
        return _mean_mse(pairs)
    pairs = _KV_GLOBAL_PAIRS if kind == "kv" else _INDEX_GLOBAL_PAIRS
    for tk, sk in pairs:
        if tk in memory_stats and sk in memory_stats:
            a = torch.as_tensor(memory_stats[tk], dtype=torch.float32)
            b = torch.as_tensor(memory_stats[sk], dtype=torch.float32)
            return _mean_mse([(a, b)])
    return 0.0, False


def compute_losses(
    teacher_logits: torch.Tensor | None = None,
    student_logits: torch.Tensor | None = None,
    teacher_hiddens: Mapping[Any, torch.Tensor] | None = None,
    student_hiddens: Mapping[Any, torch.Tensor] | None = None,
    teacher_attn: Mapping[Any, torch.Tensor] | None = None,
    student_attn: Mapping[Any, torch.Tensor] | None = None,
    memory_stats: Mapping[str, Any] | None = None,
    labels: torch.Tensor | None = None,
    config: DistillLossConfig | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Compute per-term distillation losses; return ``(losses, log_dict)``.

    ``losses`` maps {ce, kl, hidden, attn, kv, index, total} to floats where
    ``total`` is the weight-weighted sum. ``log_dict`` repeats every term
    (``loss/<term>`` raw, ``loss/<term>_weighted``), per-term ``<term>_active``
    flags (False + 0.0 when the optional input is missing), plus temperature
    and selected layers. Pure function of plain tensors/dicts.
    """
    cfg = config or DistillLossConfig()
    cfg.validate()
    selected = list(cfg.selected_layers)
    t = float(cfg.temperature)

    with torch.no_grad():
        if labels is not None and student_logits is not None:
            ce = float(F.cross_entropy(
                student_logits.detach().float().reshape(-1, student_logits.shape[-1]),
                labels.detach().reshape(-1).long(),
            ).item())
            ce_active = True
        else:
            ce, ce_active = 0.0, False

        if teacher_logits is not None and student_logits is not None:
            log_p_s = F.log_softmax(student_logits.detach().float() / t, dim=-1)
            p_t = F.softmax(teacher_logits.detach().float() / t, dim=-1)
            kl = max(0.0, float(F.kl_div(log_p_s, p_t, reduction="batchmean").item())) * (t ** 2)
            kl_active = True
        else:
            kl, kl_active = 0.0, False

        hidden, hidden_active = _mean_mse(
            _match_layers(teacher_hiddens, student_hiddens, selected))
        attn, attn_active = _mean_mse(_match_layers(teacher_attn, student_attn, selected))
        kv, kv_active = _stats_pair_mse(memory_stats, selected, "kv")
        index, index_active = _stats_pair_mse(memory_stats, selected, "index")

    total = (cfg.w_ce * ce + cfg.w_kl * kl + cfg.w_hidden * hidden
             + cfg.w_attn * attn + cfg.w_kv * kv + cfg.w_index * index)
    losses = {"ce": ce, "kl": kl, "hidden": hidden, "attn": attn, "kv": kv,
              "index": index, "total": float(total)}
    log_dict: dict[str, Any] = {
        "loss/ce": ce, "loss/kl": kl, "loss/hidden": hidden, "loss/attn": attn,
        "loss/kv": kv, "loss/index": index, "loss/total": float(total),
        "loss/ce_weighted": cfg.w_ce * ce, "loss/kl_weighted": cfg.w_kl * kl,
        "loss/hidden_weighted": cfg.w_hidden * hidden,
        "loss/attn_weighted": cfg.w_attn * attn, "loss/kv_weighted": cfg.w_kv * kv,
        "loss/index_weighted": cfg.w_index * index,
        "ce_active": ce_active, "kl_active": kl_active, "hidden_active": hidden_active,
        "attn_active": attn_active, "kv_active": kv_active, "index_active": index_active,
        "temperature": t, "selected_layers": list(selected),
    }
    return losses, log_dict


def measure_recovery(
    teacher_logits: torch.Tensor,
    student_logits_no_memory: torch.Tensor,
    tol: float = 1e-5,
) -> dict[str, Any]:
    """Gate-disabled numerical recovery check: can the student reproduce the teacher.

    Returns {max_abs_diff, mean_abs_diff, kl, within_tol} where ``within_tol``
    is True iff max abs diff <= ``tol``.
    """
    with torch.no_grad():
        t = teacher_logits.detach().float()
        s = student_logits_no_memory.detach().float()
        diff = (t - s).abs()
        max_abs = float(diff.max().item())
        mean_abs = float(diff.mean().item())
        kl = max(0.0, float(F.kl_div(F.log_softmax(s, dim=-1), F.softmax(t, dim=-1),
                            reduction="batchmean").item()))
    return {"max_abs_diff": max_abs, "mean_abs_diff": mean_abs, "kl": kl,
            "within_tol": bool(max_abs <= tol)}


def bridge_memory_stats(
    global_k: torch.Tensor,
    global_v: torch.Tensor,
) -> dict[str, Any]:
    """Small-tensor-safe stats for bridge global KV with deterministic fingerprint."""
    k = torch.as_tensor(global_k).detach().cpu().float()
    v = torch.as_tensor(global_v).detach().cpu().float()
    m2d = k.reshape(-1, k.shape[-1]) if k.ndim >= 2 else k.reshape(1, -1)
    try:
        sv = torch.linalg.svdvals(m2d)
        denom = float(sv.sum().item())
        rank_proxy = float((sv[0] / sv.sum()).item()) if denom > 0 else 0.0
    except Exception:
        rank_proxy = 0.0
    canonical = json.dumps(
        {"k_shape": list(k.shape), "k": k.flatten().tolist(),
         "v_shape": list(v.shape), "v": v.flatten().tolist()},
        sort_keys=True,
    )
    fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return {
        "norm_k": float(k.norm().item()), "norm_v": float(v.norm().item()),
        "mean_k": float(k.mean().item()), "std_k": float(k.std().item()) if k.numel() > 1 else 0.0,
        "mean_v": float(v.mean().item()), "std_v": float(v.std().item()) if v.numel() > 1 else 0.0,
        "rank_proxy": rank_proxy, "fingerprint": fingerprint,
    }


def parameter_accounting(
    named_params: Iterable[tuple[str, bool, int]],
) -> dict[str, Any]:
    """Freeze-policy accounting over ``(name, requires_grad, numel)`` triples."""
    trainable = 0
    frozen = 0
    by_prefix: dict[str, dict[str, int]] = {}
    for name, requires_grad, numel in named_params:
        n = int(numel)
        entry = by_prefix.setdefault(name.split(".")[0], {"trainable": 0, "frozen": 0, "total": 0})
        if bool(requires_grad):
            trainable += n
            entry["trainable"] += n
        else:
            frozen += n
            entry["frozen"] += n
        entry["total"] += n
    return {"trainable": trainable, "frozen": frozen, "total": trainable + frozen,
            "by_prefix": by_prefix}


class FreezePolicy:
    """Stage -> freeze-map helper for the CED distillation schedule."""

    STAGES = ("bridge_only", "bridge_global_kv", "lora_recovery", "selective_unfreeze")

    _POLICIES: dict[str, dict[str, bool]] = {
        "bridge_only": {"encoder": True, "decoder": True, "bridge_trainable": True,
                        "lora": False},
        "bridge_global_kv": {"encoder": True, "decoder": True, "bridge_trainable": True,
                             "lora": False},
        "lora_recovery": {"encoder": True, "decoder": True, "bridge_trainable": True,
                          "lora": True},
        "selective_unfreeze": {"encoder": True, "decoder": False, "bridge_trainable": True,
                               "lora": True},
    }

    @staticmethod
    def stage_policy(stage: str) -> dict[str, bool]:
        """Return {encoder, decoder, bridge_trainable, lora} for ``stage``.

        ``encoder``/``decoder`` are frozen flags. Unknown stages raise ValueError.
        """
        if stage not in FreezePolicy._POLICIES:
            raise ValueError(f"unknown freeze stage {stage!r}; expected one of "
                             f"{list(FreezePolicy.STAGES)}")
        return dict(FreezePolicy._POLICIES[stage])


class BudgetTracker:
    """Accumulating GPU-budget ledger with hard-stop gate and JSON persistence."""

    def __init__(self, max_gpu_hours: float = 50.0, gpu_class: str = "a100_80gb",
                 hard_stop: bool = True) -> None:
        if float(max_gpu_hours) < 0:
            raise ValueError("max_gpu_hours must be >= 0")
        self.max_gpu_hours = float(max_gpu_hours)
        self.gpu_class = str(gpu_class)
        self.hard_stop = bool(hard_stop)
        self._stages: list[dict[str, Any]] = []

    @property
    def consumed_gpu_hours(self) -> float:
        return float(sum(s["gpu_hours"] for s in self._stages))

    @property
    def consumed_tokens(self) -> int:
        return int(sum(s["tokens"] for s in self._stages))

    def record_stage(self, stage: str, gpu_hours: float, tokens: int,
                     peak_vram_gb: float) -> dict[str, Any]:
        """Append one stage row and enforce the hard stop; returns the row."""
        if float(gpu_hours) < 0 or int(tokens) < 0 or float(peak_vram_gb) < 0:
            raise ValueError("stage metrics must be non-negative")
        row = {"stage": str(stage), "gpu_hours": float(gpu_hours), "tokens": int(tokens),
               "peak_vram_gb": float(peak_vram_gb)}
        self._stages.append(row)
        self.check()
        return row

    def check(self) -> None:
        """Raise RuntimeError when hard_stop is set and budget is exceeded."""
        if self.hard_stop and self.consumed_gpu_hours > self.max_gpu_hours:
            raise RuntimeError(
                f"compute budget exceeded: consumed {self.consumed_gpu_hours:.3f} > "
                f"max {self.max_gpu_hours:.3f} gpu-hours ({self.gpu_class})")

    def summary(self) -> dict[str, Any]:
        peak = max((s["peak_vram_gb"] for s in self._stages), default=0.0)
        return {
            "gpu_class": self.gpu_class, "max_gpu_hours": self.max_gpu_hours,
            "hard_stop": self.hard_stop, "consumed_gpu_hours": self.consumed_gpu_hours,
            "consumed_tokens": self.consumed_tokens,
            "remaining_gpu_hours": self.max_gpu_hours - self.consumed_gpu_hours,
            "peak_vram_gb_max": float(peak), "num_stages": len(self._stages),
            "stages": [dict(s) for s in self._stages],
        }

    def to_dict(self) -> dict[str, Any]:
        return {"max_gpu_hours": self.max_gpu_hours, "gpu_class": self.gpu_class,
                "hard_stop": self.hard_stop, "stages": [dict(s) for s in self._stages],
                "consumed_gpu_hours": self.consumed_gpu_hours,
                "consumed_tokens": self.consumed_tokens}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BudgetTracker":
        tracker = cls(max_gpu_hours=float(payload.get("max_gpu_hours", 50.0)),
                      gpu_class=str(payload.get("gpu_class", "a100_80gb")),
                      hard_stop=bool(payload.get("hard_stop", True)))
        for row in payload.get("stages", []):
            tracker._stages.append({"stage": str(row["stage"]),
                                    "gpu_hours": float(row["gpu_hours"]),
                                    "tokens": int(row["tokens"]),
                                    "peak_vram_gb": float(row["peak_vram_gb"])})
        return tracker

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
                       encoding="utf-8")
        return out

    @classmethod
    def load(cls, path: str | Path) -> "BudgetTracker":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def write_evidence_report(path: str | Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Write ``payload`` as deterministic JSON (sorted keys) plus a sha field."""
    if not isinstance(payload, Mapping):
        raise ValueError("payload must be a mapping")
    canonical = json.dumps(dict(payload), sort_keys=True, indent=2)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    report = dict(payload)
    report["sha256"] = digest
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return report


def _optional_ced_imports() -> dict[str, Any]:
    """Lazy, optional access to other agents' CED model modules (never at top level)."""
    loaded: dict[str, Any] = {}
    import importlib
    for dotted in ("model.ced", "model.encoder_bridge", "model.ced_config"):
        try:
            loaded[dotted] = importlib.import_module(dotted)
        except Exception:
            loaded[dotted] = None
    return loaded
