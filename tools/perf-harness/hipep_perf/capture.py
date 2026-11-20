#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""One RGP capture positioned on an op by the trace shim's fence, then decoded.

Positioning by delay or dispatch index does not survive run-to-run jitter, so
the shim (profiling build) drains the GPU and idles right before the chosen
op's N-th instance after Run ``--fence-after-runs`` and prints
[RGP_FENCE_ARMED]; this triggers the Radeon Developer Panel inside that window.
The panel must be armed before the HIP context exists, so it starts first.

RGP streams the trace out of the live process: a late fence leaves too little
runtime for a large dump, which stalls and writes nothing. More reps (or
generated tokens after the fence) fix that, not a different position.

Two alternatives:
* ``--trigger dispatch:<delay>:<count>``: RGP's own blind trigger, for builds
  without the shim.
* ``--inproc-sqtt``: no panel; ktrace's SPM-patched HIP runtime records the
  window itself, keyed on the shim's marker kernel (KTRACE_SQTT_KERNEL). The
  patched amdhip64_7.dll must be the one the target loads (beside the exe).
"""

from __future__ import annotations

import math
import re
import subprocess
import threading
import time
from pathlib import Path

from . import bench
from .env import HarnessEnv
from .proc import Harness
from .rgp import build as rgp_build
from .rgp import formats as rgp_formats
from .traced import FenceSpec, shim_env

MARKER_KERNEL = "hipep_trace_sqtt_marker"


def _pump(stream, sink: list[str], log: Path) -> None:
    with log.open("a", encoding="utf-8", errors="replace") as f:
        for line in iter(stream.readline, ""):
            sink.append(line)
            f.write(line)
            f.flush()


class Panel:
    """RadeonDeveloperPanelCLI in dispatch-capture mode, armed before the target starts."""

    def __init__(
        self,
        cli: Path,
        rgp: Path,
        process: str,
        log: Path,
        buf: str = "minimum",
        op_count: int = 3000,
        counters: bool = False,
        instructions: bool = False,
        trigger: str | None = None,
    ):
        self.rgp, self.lines = rgp, []
        args = [
            str(cli),
            "-m",
            "profiling",
            "-o",
            str(rgp),
            "--rgp-capture-mode",
            "dispatch",
            "--rgp-sqtt-buffer-size",
            buf,
            "--rgp-render-op-count",
            str(op_count),
        ]
        if counters:
            args.append("--rgp-counter-collection")
        if instructions:
            args.append("--rgp-instruction-tracing")
        if trigger:
            args += ["--rgp-auto-capture", trigger]
        args += ["-p", process, "--verbose"]
        self.proc = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
        )
        threading.Thread(
            target=_pump, args=(self.proc.stdout, self.lines, log), daemon=True
        ).start()
        self._wait_armed(timeout=90, settle=10)

    def _wait_armed(self, timeout: float, settle: float) -> None:
        # A target that starts before the panel is listening gets attached to
        # late, and the panel then fails with "Failed to enable Profiling (RGP)
        # feature". The panel block-buffers its "Waiting for GPU application"
        # line on a pipe, so the last line it flushes in time is "Capture API
        # initialized"; enabling RGP still takes several seconds after that.
        deadline = time.time() + timeout
        while time.time() < deadline:
            text = "".join(self.lines)
            if "Failed to enable" in text or self.proc.poll() is not None:
                break
            if "Capture API initialized" in text:
                time.sleep(settle)
                if self.proc.poll() is None:
                    return
                break
            time.sleep(0.25)
        self.proc.kill()
        raise SystemExit(
            "Radeon Developer Panel did not arm (see the panel log); "
            "no target was started"
        )

    def trigger(self) -> None:
        self.proc.stdin.write("capture\n")
        self.proc.stdin.flush()

    def finish(self, dwell: int) -> None:
        """Let the dump complete, then ask the panel to quit and wait for the file.

        The panel writes the .rgp as it shuts down, so it must exit on its own.
        """
        time.sleep(dwell)
        try:
            self.proc.stdin.write("quit\n")
            self.proc.stdin.flush()
        except OSError:
            pass
        try:
            # Closes the pipe even when the flush failed (the panel had already
            # exited), so the finalizer does not retry it.
            self.proc.stdin.close()
        except OSError:
            pass
        size = -1
        for _ in range(100):
            if self.proc.poll() is not None:
                break
            time.sleep(3)
            if self.rgp.exists():
                now = self.rgp.stat().st_size
                if now > 0 and now == size:
                    break
                size = now

    def stall_note(self) -> str:
        prog = [
            m
            for m in (re.search(r"([\d.]+) / ([\d.]+) MB", ln) for ln in self.lines)
            if m
        ]
        if not prog:
            return ""
        got, tot = float(prog[-1].group(1)), float(prog[-1].group(2))
        return (
            f"dump stalled at {got:.1f} / {tot:.1f} MB: the target exited before RGP "
            "finished streaming; give it more runtime after the window."
        )


def main(env: HarnessEnv, a, spec: bench.BenchSpec) -> int:
    env.require("bin", "model")
    if a.decode_step:
        if spec.prompt_file:
            raise SystemExit(
                "--decode-step needs --seqlen (random ids): a prompt file's token "
                "count is unknown here. Use --fence-after-runs instead."
            )
        chunks = math.ceil((spec.seqlen or 128) / a.prefill_chunk)
        a.fence_after_runs = chunks + a.decode_step - 1
        print(
            f"decode step {a.decode_step} at seqlen {spec.seqlen} = {chunks} prefill "
            f"Runs -> --fence-after-runs {a.fence_after_runs} (check the chunking "
            "with a trace)"
        )
    trigger = getattr(a, "trigger", None)
    inproc = getattr(a, "inproc_sqtt", False)
    if not a.fence_op and not trigger:
        raise SystemExit("capture needs --fence-op (profiling build) or --trigger")
    if not inproc:
        env.require("rgp_cli")
        if a.sqtt_marker:
            # Seen on gfx1151: the target spins forever and the panel's trace
            # stays at 0%. The fence alone opens the panel's window at the op.
            raise SystemExit(
                "--sqtt-marker hangs panel captures; it is only for --inproc-sqtt"
            )
    bench._defaults(spec)
    spec.warmup = 0
    out_dir = spec.out_dir or env.out_root / "captures"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = spec.tag
    rgp = out_dir / f"{tag}.rgp"
    bench_log, panel_log = out_dir / f"bench_{tag}.log", out_dir / f"panel_{tag}.log"
    for p in (rgp, bench_log, panel_log):
        p.unlink(missing_ok=True)

    harness = Harness(env.out_root)
    with harness.lock():
        harness.stop_competitors()
        panel = None
        if not inproc:
            process = "python" if spec.driver == "vlm" else "model_benchmark"
            panel = Panel(
                env.rgp_cli,
                rgp,
                process,
                panel_log,
                a.buf,
                a.op_count,
                a.counters,
                a.instructions,
                trigger,
            )
            print(
                f"panel pid={panel.proc.pid} buf={a.buf} ops={a.op_count} "
                f"counters={a.counters}"
            )
        child = env.run_env()
        child.update(spec.set_env)
        if a.fence_op:
            fence = FenceSpec(
                a.fence_op,
                a.fence_skip,
                a.fence_after_runs,
                a.fence_ms,
                a.fence_graph,
                sqtt_marker=inproc,
            )
            child.update(shim_env(env))
            child.update(fence.env())
        if inproc:
            child.update(
                {
                    "KTRACE_SQTT_OUTPUT": str(rgp),
                    "KTRACE_SQTT_KERNEL": MARKER_KERNEL,
                    "KTRACE_SQTT_COUNT": str(a.op_count),
                }
            )
            if a.instructions:
                child["KTRACE_SQTT_INSTRUCTIONS"] = "1"
        cmd = bench._command(env, spec, out_dir / f"bench_{tag}.json")
        with bench_log.open("w", encoding="utf-8", errors="replace") as f:
            target = subprocess.Popen(
                cmd, cwd=str(env.bin), env=child, stdout=f, stderr=subprocess.STDOUT
            )
        if spec.driver == "vlm":
            harness.register_driver(target.pid)
        print(f"target pid={target.pid}")

        triggered = panel is None or trigger is not None
        armed = ""
        deadline = time.time() + 1800
        while target.poll() is None and time.time() < deadline:
            if not armed:
                m = re.search(
                    r"^\[RGP_FENCE_ARMED\].*$",
                    bench_log.read_text(errors="replace"),
                    re.MULTILINE,
                )
                if m:
                    armed = m.group(0)
                    print(f">>> {armed}")
                    if not triggered:
                        time.sleep(0.7)
                        panel.trigger()
                        triggered = True
            time.sleep(0.25)
        print(
            f"target exited={target.poll() is not None} triggered={triggered}; "
            f"dwell {a.dwell}s"
        )
        if panel is not None:
            panel.finish(a.dwell)
        harness.stop_competitors()

    if not rgp.exists():
        print(f"RESULT: {rgp} NOT created")
        note = panel.stall_note() if panel else ""
        if note:
            print(f"!! {note} Raise --reps / --gen or lower --op-count.")
        if not armed and a.fence_op:
            print(
                "!! the fence never armed: wrong op name, Run index past the end, or "
                "not a profiling build (see bench log)."
            )
        return 1
    ok, detail = rgp_build.verify(rgp, need_spm=a.counters and not inproc)
    print(("capture OK: " if ok else "capture UNUSABLE: ") + detail)
    if not ok:
        return 1
    cap = rgp_build.build(rgp)
    for p in rgp_formats.write_all(cap, str(out_dir / tag)):
        print(f"wrote {p}")
    return 0
