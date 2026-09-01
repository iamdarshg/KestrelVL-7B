"""Atomic, resumable checkpoint state including RNG and dataset cursor."""

import json
import hashlib
import os
import random
import signal
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file, save_file


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


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {"__ndarray__": value.tolist(), "dtype": str(value.dtype)}
    if isinstance(value, tuple):
        return {"__tuple__": [_json_safe(item) for item in value]}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"value is not JSON serializable: {type(value)!r}")


def _restore_json_safe(value: Any) -> Any:
    if isinstance(value, list):
        return [_restore_json_safe(item) for item in value]
    if isinstance(value, dict) and "__tuple__" in value:
        return tuple(_restore_json_safe(item) for item in value["__tuple__"])
    if isinstance(value, dict) and "__ndarray__" in value:
        return np.asarray(value["__ndarray__"], dtype=value["dtype"])
    if isinstance(value, dict):
        return {key: _restore_json_safe(item) for key, item in value.items()}
    return value


def _safe_tensor_name(prefix: str, *parts: object) -> str:
    return prefix + "__" + "__".join(str(part).replace(".", "_") for part in parts)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_checkpoint_checksums(path: Path) -> None:
    checksum_path = path / "checksums.json"
    if not checksum_path.exists():
        return
    checksums = json.loads(checksum_path.read_text(encoding="utf-8"))
    for name, expected in checksums.items():
        artifact = path / name
        if not artifact.is_file() or _file_sha256(artifact) != expected:
            raise ValueError(f"checkpoint checksum mismatch: {name}")


class SafeCheckpointManager(CheckpointManager):
    """Safetensors/JSON resume checkpoints for production training.

    ``CheckpointManager`` is retained for legacy tiny/ablation jobs.  Real
    Nemotron training should use this manager: model and optimizer tensors are
    safetensors, scheduler/dataset metadata is JSON, and RNG state is split
    between JSON-compatible Python/NumPy state and a safetensors tensor file.
    No release or resume path needs pickle deserialization.
    """

    def __init__(
        self,
        *args: Any,
        durable_uri: str | None = None,
        checkpoint_metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.durable_uri = durable_uri
        self.checkpoint_metadata = dict(checkpoint_metadata or {})

    def save(
        self,
        step: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any = None,
        dataset_state: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> Path:
        target = self.directory / f"step_{step:08d}"
        temporary = self.directory / f".step_{step:08d}.tmp"
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
        temporary.mkdir()
        try:
            state_fn = getattr(model, "trainable_state_dict", None)
            model_state = state_fn() if state_fn is not None else model.state_dict()
            model_tensors: dict[str, torch.Tensor] = {}
            model_names: dict[str, str] = {}
            for name, tensor in model_state.items():
                safe = _safe_tensor_name("model", name)
                model_tensors[safe] = tensor.detach().to(device="cpu").contiguous()
                model_names[name] = safe
            save_file(model_tensors, str(temporary / "model.safetensors"), metadata={"format": "kestrel-resume-v1"})

            optimizer_manifest: dict[str, Any] | None = None
            if optimizer is not None:
                optimizer_state = optimizer.state_dict()
                optimizer_tensors: dict[str, torch.Tensor] = {}
                optimizer_values: dict[str, dict[str, Any]] = {}
                for parameter_id, values in optimizer_state["state"].items():
                    json_values: dict[str, Any] = {}
                    for name, value in values.items():
                        key = str(parameter_id)
                        if torch.is_tensor(value):
                            safe = _safe_tensor_name("optimizer", parameter_id, name)
                            optimizer_tensors[safe] = value.detach().to(device="cpu").contiguous()
                            json_values[name] = {"tensor": safe}
                        else:
                            json_values[name] = _json_safe(value)
                    optimizer_values[key] = json_values
                if optimizer_tensors:
                    save_file(optimizer_tensors, str(temporary / "optimizer.safetensors"), metadata={"format": "kestrel-resume-v1"})
                optimizer_manifest = {
                    "state": optimizer_values,
                    "param_groups": _json_safe(optimizer_state["param_groups"]),
                }
                (temporary / "optimizer.json").write_text(
                    json.dumps(optimizer_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
            if scheduler is not None:
                (temporary / "scheduler.json").write_text(
                    json.dumps(_json_safe(scheduler.state_dict()), indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
            rng = capture_rng_state()
            rng_tensors = {"torch": rng["torch"].contiguous()}
            if "cuda" in rng:
                rng_tensors.update({f"cuda_{idx}": value.contiguous() for idx, value in enumerate(rng["cuda"])})
            save_file(rng_tensors, str(temporary / "rng.safetensors"), metadata={"format": "kestrel-rng-v1"})
            rng_json = {
                "python": _json_safe(rng["python"]),
                "numpy": _json_safe(rng["numpy"]),
            }
            (temporary / "rng.json").write_text(json.dumps(rng_json, indent=2) + "\n", encoding="utf-8")
            (temporary / "state.json").write_text(
                json.dumps(
                    {
                        "step": step,
                        "dataset": dataset_state or {},
                        "metrics": metrics or {},
                        "model_names": model_names,
                        "saved_at": time.time(),
                        "format": "kestrel-resume-v1",
                    },
                    indent=2,
                    default=str,
                )
                + "\n",
                encoding="utf-8",
            )
            metadata = _json_safe(self.checkpoint_metadata)
            if "git_commit" not in metadata:
                try:
                    metadata["git_commit"] = subprocess.check_output(
                        ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
                    ).strip()
                except (OSError, subprocess.CalledProcessError):
                    metadata["git_commit"] = None
            config = metadata.get("config", {})
            if not isinstance(config, dict):
                raise TypeError("checkpoint metadata config must be a JSON object")
            (temporary / "config.json").write_text(
                json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            artifact_names = sorted(
                file.name
                for file in temporary.iterdir()
                if file.is_file() and file.name not in {"checksums.json", "manifest.json"}
            )
            checksums = {name: _file_sha256(temporary / name) for name in artifact_names}
            (temporary / "checksums.json").write_text(
                json.dumps(checksums, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            (temporary / "manifest.json").write_text(
                json.dumps(
                    {
                        "format": "kestrel-resume-v1",
                        "step": step,
                        "files": artifact_names + ["checksums.json", "manifest.json"],
                        "metadata": metadata,
                        "dataset_state": dataset_state or {},
                        "metrics": metrics or {},
                    },
                    indent=2,
                    sort_keys=True,
                    default=str,
                )
                + "\n",
                encoding="utf-8",
            )
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        if target.exists():
            shutil.rmtree(target)
        os.replace(temporary, target)
        (self.directory / "latest").write_text(target.name, encoding="utf-8")
        if self.max_checkpoints is not None:
            checkpoints = sorted(
                (path for path in self.directory.glob("step_*") if path.is_dir()),
                key=lambda path: path.name,
            )
            for stale in checkpoints[:-self.max_checkpoints]:
                shutil.rmtree(stale)
        if self.durable_uri:
            if shutil.which("gcloud") is None:
                raise RuntimeError("durable checkpoint URI configured but gcloud is unavailable")
            subprocess.run(
                ["gcloud", "storage", "cp", "--recursive", str(target), f"{self.durable_uri.rstrip('/')}/{target.name}"],
                check=True,
            )
        return target

    def load(
        self,
        path: str | Path,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any = None,
    ) -> dict[str, Any]:
        path = Path(path)
        _verify_checkpoint_checksums(path)
        state = json.loads((path / "state.json").read_text(encoding="utf-8"))
        model_tensors = load_file(str(path / "model.safetensors"), device="cpu")
        model_state = {name: model_tensors[safe] for name, safe in state["model_names"].items()}
        load_trainable = getattr(model, "load_trainable_state_dict", None)
        if load_trainable is not None:
            load_trainable(model_state)
        else:
            model.load_state_dict(model_state)
        if optimizer is not None and (path / "optimizer.json").exists():
            manifest = json.loads((path / "optimizer.json").read_text(encoding="utf-8"))
            optimizer_tensors = load_file(str(path / "optimizer.safetensors"), device="cpu") if (path / "optimizer.safetensors").exists() else {}
            optimizer_state = {int(key): {
                name: optimizer_tensors[value["tensor"]] if isinstance(value, dict) and "tensor" in value else _restore_json_safe(value)
                for name, value in values.items()
            } for key, values in manifest["state"].items()}
            optimizer.load_state_dict({"state": optimizer_state, "param_groups": _restore_json_safe(manifest["param_groups"])})
        if scheduler is not None and (path / "scheduler.json").exists():
            scheduler.load_state_dict(_restore_json_safe(json.loads((path / "scheduler.json").read_text(encoding="utf-8"))))
        rng_tensors = load_file(str(path / "rng.safetensors"), device="cpu")
        rng_json = json.loads((path / "rng.json").read_text(encoding="utf-8"))
        restored_rng = {
            "python": _restore_json_safe(rng_json["python"]),
            "numpy": _restore_json_safe(rng_json["numpy"]),
            "torch": rng_tensors["torch"],
        }
        if torch.cuda.is_available():
            restored_rng["cuda"] = [
                rng_tensors[key] for key in sorted(rng_tensors) if key.startswith("cuda_")
            ]
        restore_rng_state(restored_rng)
        return state
