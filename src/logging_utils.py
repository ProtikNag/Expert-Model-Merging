"""Structured logging and run manifest for reproducibility.

Every script creates a ``RunLogger`` which (a) tees stdout to a timestamped
file under ``results/logs/``, (b) writes a run manifest capturing package
versions, git state, config hash, and host info, and (c) exposes a tiny
metric-recording API. All artifacts carry the same run-id suffix so a single
experiment is easy to reassemble after the fact.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import torch


@dataclass
class RunManifest:
    """Per-run provenance record."""
    run_id: str
    script: str
    started_at: str
    host: str
    platform: str
    python: str
    torch: str
    cuda: Optional[str]
    config_path: Optional[str]
    config_sha256: Optional[str]
    cli_args: list = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


def _hash_file(path: str | Path) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_head() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
            cwd=Path(__file__).resolve().parent)
        return out.decode().strip()
    except Exception:
        return None


def _pkg_versions() -> Dict[str, str]:
    """Collect key package versions for the manifest."""
    out: Dict[str, str] = {}
    for pkg in ["numpy", "transformers", "datasets", "torchvision",
                "sklearn", "matplotlib"]:
        try:
            mod = __import__(pkg.replace("-", "_"))
            out[pkg] = getattr(mod, "__version__", "?")
        except Exception:
            out[pkg] = "missing"
    return out


class _Tee:
    """File+stdout duplicator used by ``RunLogger``.

    Proxies any attribute access we don't explicitly override to the
    wrapped stream, so third-party libraries that query ``stdout.isatty()``,
    ``stdout.fileno()``, etc. continue to work.
    """

    def __init__(self, stream, file_obj) -> None:
        self._stream = stream
        self._file = file_obj

    def write(self, data: str) -> int:
        self._file.write(data)
        self._file.flush()
        return self._stream.write(data)

    def flush(self) -> None:
        self._file.flush()
        self._stream.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


class RunLogger:
    """Tees stdout/stderr to a timestamped log file and tracks metrics."""

    def __init__(self,
                 experiment: str,
                 log_root: str | Path = "./results/logs",
                 config_path: Optional[str] = None,
                 extra: Optional[Dict[str, Any]] = None) -> None:
        self.run_id = time.strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}"
        self.experiment = experiment
        self.log_dir = Path(log_root)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / f"{experiment}_{self.run_id}.log"
        self.manifest_path = self.log_dir / f"{experiment}_{self.run_id}.manifest.json"
        self.metrics_path = self.log_dir / f"{experiment}_{self.run_id}.metrics.jsonl"
        self._file = open(self.log_path, "w", buffering=1)
        self._tee_out = _Tee(sys.stdout, self._file)
        self._tee_err = _Tee(sys.stderr, self._file)
        sys.stdout = self._tee_out
        sys.stderr = self._tee_err

        cuda_ver = torch.version.cuda if torch.cuda.is_available() else None
        self.manifest = RunManifest(
            run_id=self.run_id,
            script=experiment,
            started_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            host=socket.gethostname(),
            platform=platform.platform(),
            python=sys.version.split()[0],
            torch=torch.__version__,
            cuda=cuda_ver,
            config_path=config_path,
            config_sha256=_hash_file(config_path) if config_path else None,
            cli_args=list(sys.argv),
            env={"pkg": _pkg_versions(),
                 "git_head": _git_head() or "none",
                 "extra": extra or {}},
        )
        with open(self.manifest_path, "w") as f:
            json.dump(self.manifest.to_dict(), f, indent=2)
        self.log_header()

    def log_header(self) -> None:
        m = self.manifest
        print("=" * 72)
        print(f"  {self.experiment}  [run_id={self.run_id}]")
        print(f"  host: {m.host}  |  {m.platform}")
        print(f"  python {m.python}  torch {m.torch}"
              + (f"  cuda {m.cuda}" if m.cuda else "  (cpu)"))
        print(f"  started: {m.started_at}")
        if m.config_path:
            print(f"  config: {m.config_path}  (sha256 {m.config_sha256[:12]})")
        print("=" * 72)

    def record(self, **fields: Any) -> None:
        """Append a JSON line to the metrics log."""
        fields = {"t": time.time(), **fields}
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(fields, default=_json_default) + "\n")

    def close(self) -> None:
        sys.stdout = self._tee_out._stream
        sys.stderr = self._tee_err._stream
        self._file.close()


def _json_default(o: Any) -> Any:
    if isinstance(o, torch.Tensor):
        if o.numel() == 1:
            return o.item()
        return o.detach().cpu().tolist()
    if isinstance(o, Path):
        return str(o)
    try:
        return float(o)
    except Exception:
        return str(o)


@contextmanager
def run_context(experiment: str, **kwargs: Any):
    """Context manager wrapper around :class:`RunLogger`."""
    logger = RunLogger(experiment=experiment, **kwargs)
    try:
        yield logger
    finally:
        logger.close()
