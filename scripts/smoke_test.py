"""End-to-end smoke test: tiny BERT + 2 tasks + tiny subsets.

Runs the full pipeline (train experts -> collect statistics -> merge with
every method -> evaluate -> generate figures) using the ``glue_smoke.yaml``
config. The goal is to catch integration bugs before a long GPU run. Every
scalar should land in the structured logs and every figure should render.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "glue_smoke.yaml"
PYTHON = sys.executable


def _run(cmd: list) -> None:
    print("\n$ " + " ".join(str(c) for c in cmd))
    res = subprocess.run(cmd, cwd=ROOT)
    if res.returncode != 0:
        raise SystemExit(f"step failed with exit code {res.returncode}")


def main() -> None:
    _run([PYTHON, "scripts/lm_train_experts.py", "--config", str(CONFIG),
          "--device", "cpu"])
    _run([PYTHON, "scripts/lm_run_merging.py", "--config", str(CONFIG),
          "--device", "cpu"])
    _run([PYTHON, "scripts/lm_make_figures.py", "--config", str(CONFIG)])
    print("\n[smoke test] all stages completed successfully.")


if __name__ == "__main__":
    main()
