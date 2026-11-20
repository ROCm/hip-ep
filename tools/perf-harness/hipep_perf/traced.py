#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Run a workload under the trace shim and post-process the result.

Needs an EP built with -DHIPDNN_EP_ENABLE_TRACE_SHIM=ON in HIPEP_BIN. Both ROCTx
bridges are pointed at that build's hipgpu.dll, which exports the shim's
roctxRangePushA/Pop/MarkA, so op and subgraph ranges land in the same ring as
the GPU commands. Traced runs are for structure and attribution only: the
shim and the per-op labels cost time, so never read throughput from them.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from . import bench, trace
from .collector import Collector
from .env import HarnessEnv
from .proc import Harness


@dataclass
class FenceSpec:
    op: str
    skip: int = 0
    after_runs: int = 0
    sleep_ms: int = 200
    graph_filter: str = ""
    sqtt_marker: bool = False

    def env(self) -> dict[str, str]:
        e = {
            "HIPEP_TRACE_FENCE": self.op,
            "HIPEP_TRACE_FENCE_SKIP": str(self.skip),
            "HIPEP_TRACE_FENCE_AFTER_RUNS": str(self.after_runs),
            "HIPEP_TRACE_FENCE_MS": str(self.sleep_ms),
        }
        if self.graph_filter:
            e["HIPEP_TRACE_FENCE_GRAPH"] = self.graph_filter
        if self.sqtt_marker:
            e["HIPEP_TRACE_SQTT_MARKER"] = "1"
        return e


def shim_env(env: HarnessEnv) -> dict[str, str]:
    lib = env.bin / "hipgpu.dll"
    return {"HIPDNN_EP_ROCTX": "1", "HIPDNN_EP_ROCTX_LIB": str(lib)}


def run_command(
    env: HarnessEnv,
    cmd: list[str],
    out_dir: Path,
    extra_env: dict[str, str] | None = None,
    api: bool = True,
    capacity: int = 1 << 21,
    cwd: Path | None = None,
) -> dict:
    """Run an arbitrary command under the collector; returns the collector meta."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    harness = Harness(env.out_root)
    with harness.lock():
        harness.stop_competitors()
        col = Collector(out_dir, capacity=capacity).start()
        child = env.run_env()
        child.update(shim_env(env))
        child.update(col.env())
        if not api:
            child["HIPEP_TRACE_API"] = "0"
        child.update(extra_env or {})
        t0 = time.time()
        with (out_dir / "target.log").open(
            "w", encoding="utf-8", errors="replace"
        ) as log:
            p = subprocess.Popen(
                cmd,
                cwd=str(cwd or env.bin),
                env=child,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            rc = p.wait()
        meta = col.stop(
            {
                "command": cmd,
                "exit_code": rc,
                "seconds": round(time.time() - t0, 3),
                "env": {
                    k: v
                    for k, v in child.items()
                    if k.startswith(("HIPEP_", "HIPDNN_"))
                },
            }
        )
    if not meta.get("attached"):
        print(
            "WARNING: the shim never attached. Is HIPEP_BIN a build with "
            "-DHIPDNN_EP_ENABLE_TRACE_SHIM=ON? See target.log."
        )
    return meta


def run_bench(
    env: HarnessEnv,
    spec: bench.BenchSpec,
    out_dir: Path,
    fence: FenceSpec | None = None,
    api: bool = True,
) -> dict:
    env.require("bin", "model")
    bench._defaults(spec)
    cmd = bench._command(env, spec, Path(out_dir) / "bench.json")
    extra = dict(spec.set_env)
    if fence:
        extra.update(fence.env())
    return run_command(env, cmd, out_dir, extra, api=api)


def postprocess(
    out_dir: Path, chrome: bool = True, pftrace: bool = True, api_slices: bool = False
) -> Path:
    out_dir = Path(out_dir)
    tr = trace.load(out_dir)
    trace.attribute(tr)
    rep = trace.write_report(tr, out_dir)
    if chrome:
        trace.write_chrome(tr, out_dir / "trace_chrome.json", api=api_slices)
    if pftrace:
        trace.write_pftrace(tr, out_dir / "trace.pftrace")
    trace.write_rocprof(tr, out_dir)
    (out_dir / "trace_validation.json").write_text(
        json.dumps(trace.validate(tr), indent=2)
    )
    return rep
