"""Atomic, resumable checkpoint state including RNG and dataset cursor."""

import json
import os
import random
import signal
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


class CheckpointManager:
    def __init__(
        self,
        directory: str | Path,
        interval_steps: int = 100,
        max_checkpoints: int | None = None,
    ) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.interval_steps = interval_steps
        self.max_checkpoints = max_checkpoints
        if max_checkpoints is not None and max_checkpoints < 1:
            raise ValueError("max_checkpoints must be positive or None")
        self.stop_requested = False
        signal.signal(signal.SIGTERM, self._signal)
        if hasattr(signal, "SIGBREAK"):
            signal.signal(signal.SIGBREAK, self._signal)
        # A process can die during torch.save (including a full-disk failure)
        # after creating a temporary checkpoint.  It is never a valid resume
        # point because ``latest`` is advanced only after the atomic replace.
        # Remove only those manager-owned temporary directories on startup.
        for temporary in self.directory.glob(".step_*.tmp"):
            if temporary.is_dir():
                shutil.rmtree(temporary, ignore_errors=True)
            elif temporary.exists():
                temporary.unlink()

    def _signal(self, *_args: object) -> None:
        self.stop_requested = True

    def should_save(self, step: int) -> bool:
        return step == 0 or step % self.interval_steps == 0 or self.stop_requested

    def save(self, step: int, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None = None, scheduler: Any = None, dataset_state: dict[str, Any] | None = None, metrics: dict[str, Any] | None = None) -> Path:
        target = self.directory / f"step_{step:08d}"
        temporary = self.directory / f".step_{step:08d}.tmp"
        if temporary.exists():
            if temporary.is_dir():
                shutil.rmtree(temporary)
            else:
                temporary.unlink()
        temporary.mkdir()
        # Real Nemotron candidates keep the frozen 4-bit backbone in the
        # immutable HF cache.  Their checkpoints contain only trainable
        # transplant parameters; tiny test models still save a normal state
        # dict.  This prevents each periodic checkpoint from duplicating GBs
        # of frozen weights.
        state_dict_fn = getattr(model, "trainable_state_dict", None)
        model_state = state_dict_fn() if state_dict_fn is not None else model.state_dict()
        try:
            torch.save(model_state, temporary / "model.pt")
            if optimizer is not None:
                torch.save(optimizer.state_dict(), temporary / "optimizer.pt")
            if scheduler is not None:
                torch.save(scheduler.state_dict(), temporary / "scheduler.pt")
            torch.save(capture_rng_state(), temporary / "rng.pt")
            (temporary / "state.json").write_text(json.dumps({"step": step, "dataset": dataset_state or {}, "metrics": metrics or {}, "saved_at": time.time()}, indent=2, default=str), encoding="utf-8")
        except BaseException:
            # Do not strand a partial model/optimizer snapshot after a failed
            # write.  The previous ``latest`` checkpoint remains intact.
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        if target.exists():
            for child in target.iterdir():
                child.unlink()
            target.rmdir()
        os.replace(temporary, target)
        (self.directory / "latest").write_text(target.name, encoding="utf-8")
        if self.max_checkpoints is not None:
            checkpoints = sorted(
                (path for path in self.directory.glob("step_*" ) if path.is_dir()),
                key=lambda path: path.name,
            )
            for stale in checkpoints[:-self.max_checkpoints]:
                for child in stale.iterdir():
                    if child.is_file():
                        child.unlink()
                stale.rmdir()
        return target

    def latest(self) -> Path | None:
        pointer = self.directory / "latest"
        if not pointer.exists():
            return None
        path = self.directory / pointer.read_text(encoding="utf-8").strip()
        return path if path.exists() else None

    def load(self, path: str | Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None = None, scheduler: Any = None) -> dict[str, Any]:
        path = Path(path)
        state = torch.load(path / "model.pt", map_location="cpu", weights_only=False)
        load_trainable = getattr(model, "load_trainable_state_dict", None)
        if load_trainable is not None:
            load_trainable(state)
        else:
            model.load_state_dict(state)
        if optimizer is not None and (path / "optimizer.pt").exists():
            optimizer.load_state_dict(torch.load(path / "optimizer.pt", map_location="cpu"))
        if scheduler is not None and (path / "scheduler.pt").exists():
            scheduler.load_state_dict(torch.load(path / "scheduler.pt", map_location="cpu"))
        if (path / "rng.pt").exists():
            restore_rng_state(torch.load(path / "rng.pt", map_location="cpu", weights_only=False))
        return json.loads((path / "state.json").read_text(encoding="utf-8"))
