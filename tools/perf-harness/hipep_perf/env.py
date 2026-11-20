#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Environment resolution and the per-run environment discipline.

Everything resolves from environment variables so the same commands run on any
machine. Timed runs always start from CI's environment: the profiling switches
below are cleared, and anything else is opted into per run and restored after.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

# Variables that change what a timed run measures. HIPDNN_EP_PERF alone costs
# about 4%; HIPDNN_EP_TRACE_FILE and HIPDNN_EP_PERF_OPS turn the same profiler
# on silently; HIPDNN_EP_ROCTX builds a label per op; HIPDNN_EP_HOST_TIMELINE
# writes files. CI sets none of them, nor the two tuning switches.
PROFILING_VARS = (
    "HIPDNN_EP_PERF",
    "HIPDNN_EP_PERF_OPS",
    "HIPDNN_EP_DEBUG",
    "HIPDNN_EP_TRACE_FILE",
    "HIPDNN_EP_HOST_TIMELINE",
    "HIPDNN_EP_ROCTX",
    "HIPDNN_EP_ROCTX_LIB",
    "HIPDNN_EP_ROCTX_VERBOSE",
    "HIPDNN_EP_AUTOTUNE",
    "HIPDNN_EP_MATMUL_CUSTOM_WMMA",
)

# Trace-shim and capture controls: an armed fence idles the GPU mid-run.
CAPTURE_VARS = (
    "HIPEP_TRACE",
    "HIPEP_TRACE_API",
    "HIPEP_TRACE_FENCE",
    "HIPEP_TRACE_FENCE_SKIP",
    "HIPEP_TRACE_FENCE_MS",
    "HIPEP_TRACE_FENCE_AFTER_RUNS",
    "HIPEP_TRACE_FENCE_GRAPH",
    "HIPEP_TRACE_SQTT_MARKER",
    "RGP_FENCE",
    "RGP_FENCE_SKIP",
    "RGP_FENCE_MS",
    "RGP_FENCE_AFTER_INFERENCES",
)


def _existing(name: str) -> Path | None:
    v = os.environ.get(name)
    return Path(v).resolve() if v and Path(v).exists() else None


@dataclass
class HarnessEnv:
    bin: Path | None
    model: Path | None
    python: str
    out_root: Path
    vlm_bench: Path | None
    image: Path | None
    prompt_file: Path | None
    path_extra: list[str]
    rgp_cli: Path | None

    @classmethod
    def resolve(cls) -> HarnessEnv:
        out = os.environ.get("HIPEP_OUT") or str(
            Path(tempfile.gettempdir()) / "hipep-perf"
        )
        extra = [p for p in os.environ.get("HIPEP_PATH_EXTRA", "").split(";") if p]
        return cls(
            bin=_existing("HIPEP_BIN"),
            model=_existing("HIPEP_MODEL"),
            python=os.environ.get("HIPEP_PY") or shutil.which("python") or "python",
            out_root=Path(out),
            vlm_bench=_existing("HIPEP_VLM_BENCH"),
            image=_existing("HIPEP_IMAGE"),
            prompt_file=_existing("HIPEP_PROMPT_FILE"),
            path_extra=extra,
            rgp_cli=find_rgp_cli(),
        )

    def require(self, *fields: str) -> None:
        names = {
            "bin": "HIPEP_BIN (model_benchmark.exe + EP DLLs)",
            "model": "HIPEP_MODEL (directory with genai_config.json)",
            "vlm_bench": "HIPEP_VLM_BENCH (onnxruntime-genai vlm_benchmark.py)",
            "image": "HIPEP_IMAGE (benchmark image)",
            "rgp_cli": "RGP_DIR or PATH (RadeonDeveloperPanelCLI.exe)",
        }
        missing = [names.get(f, f) for f in fields if getattr(self, f) is None]
        if missing:
            raise SystemExit("missing: " + "; ".join(missing))

    def run_env(self, base: dict[str, str] | None = None) -> dict[str, str]:
        """Child environment for a timed run: CI's, with the bin dir on PATH."""
        env = dict(os.environ if base is None else base)
        for k in PROFILING_VARS + CAPTURE_VARS:
            env.pop(k, None)
        parts = [str(self.bin)] if self.bin else []
        parts += self.path_extra
        env["PATH"] = os.pathsep.join(parts + [env.get("PATH", "")])
        return env


def find_rgp_cli() -> Path | None:
    exe = "RadeonDeveloperPanelCLI.exe"
    on_path = shutil.which(exe)
    if on_path:
        return Path(on_path)
    d = os.environ.get("RGP_DIR")
    if d and (Path(d) / exe).exists():
        return Path(d) / exe
    for root in (Path("C:/tools"), Path.home() / "workspace"):
        if not root.exists():
            continue
        for suite in sorted(root.glob("RadeonDeveloperToolSuite*"), reverse=True):
            hits = sorted(suite.rglob(exe))
            if hits:
                return hits[0]
    return None


def parse_pairs(pairs: list[str] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for kv in pairs or []:
        k, sep, v = kv.partition("=")
        if not k or not sep:
            raise SystemExit(f"--set-env expects K=V, got {kv!r}")
        out[k] = v
    return out


@contextlib.contextmanager
def scoped_environ(updates: dict[str, str | None]):
    """Apply updates to os.environ and restore the previous values afterwards.

    os.environ is process-wide; a key left behind by one arm silently applies to
    every later run, which once turned a whole config sweep into noise.
    """
    saved = {k: os.environ.get(k) for k in updates}
    try:
        for k, v in updates.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
