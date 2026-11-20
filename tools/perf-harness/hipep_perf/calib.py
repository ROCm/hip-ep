#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Ground-truth calibration: build the workload, trace it, score the tracer.

The workload (calib/hipep_calib.hip) runs kernels whose cost is known by
construction. ``check`` runs it under the trace shim and scores:

* spin kernels: traced duration vs the exact wall-clock duration;
* every kernel: traced duration vs its own hipEvent pair (independent source);
* memory kernels: achieved GB/s from exact byte counts;
* attribution: each kernel must land in the ROCTx phase that launched it;
* clocks: no command may start before the HIP call that issued it.
"""

from __future__ import annotations

import json
import os
import re
import statistics as st
import subprocess
from pathlib import Path

from . import trace, traced
from .env import HarnessEnv
from .proc import Harness

SRC = Path(__file__).resolve().parent.parent / "calib" / "hipep_calib.hip"

# Pass bars. Spin truth is exact, so the tracer may only add launch-edge
# overhead: within 2 us or 2%, whichever is larger, for 95% of launches.
SPIN_ABS_US = 2.0
SPIN_REL = 0.02
SPIN_QUANTILE = 0.95
SPM_REL = 0.05  # SPM bytes per kernel against bytes moved by construction


def _therock() -> Path:
    for k in ("THEROCK_DIST", "HIPEP_THEROCK"):
        v = os.environ.get(k)
        if v and (Path(v) / "bin" / "hipcc.exe").exists():
            return Path(v)
    raise SystemExit(
        "set THEROCK_DIST to the TheRock SDK (needs bin/hipcc.exe and bin/clang.exe)"
    )


def _vsdevcmd() -> Path:
    vswhere = (
        Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)"))
        / "Microsoft Visual Studio/Installer/vswhere.exe"
    )
    if vswhere.exists():
        root = subprocess.run(
            [
                str(vswhere),
                "-latest",
                "-products",
                "*",
                "-property",
                "installationPath",
            ],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        cand = Path(root) / "Common7/Tools/VsDevCmd.bat"
        if cand.exists():
            return cand
    raise SystemExit("Visual Studio not found (needed for the HIP host link)")


def exe_path(env: HarnessEnv) -> Path:
    return env.out_root / "calib" / "hipep_calib.exe"


def build(env: HarnessEnv, arch: str = "") -> Path:
    rock = _therock()
    if not arch:
        arch = subprocess.run(
            [str(rock / "bin" / "amdgpu-arch.exe")],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()[0]
    out = exe_path(env)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = (
        f'call "{_vsdevcmd()}" -arch=amd64 -host_arch=amd64 >nul && '
        f'set "HIP_CLANG_PATH={rock / "bin"}" && '
        f'"{rock / "bin" / "hipcc.exe"}" --offload-arch={arch} -O2 -o "{out}" "{SRC}"'
    )
    # A raw string keeps cmd's own quoting; a list would escape the quotes.
    r = subprocess.run(
        f'cmd /s /c "{cmd}"', capture_output=True, text=True, check=False
    )
    if r.returncode != 0 or not out.exists():
        raise SystemExit(f"calib build failed:\n{r.stdout}\n{r.stderr}")
    print(f"built {out} for {arch}")
    return out


def _workload_json(log: Path) -> dict:
    raw = Path(log).read_bytes()
    # A PowerShell redirect writes UTF-16 with CRLF.
    enc = "utf-16" if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else "utf-8"
    text = raw.decode(enc, errors="replace")
    m = re.search(r'\{\s*"device"', text)
    end = text.rfind("}")
    if not m or end < 0:
        raise SystemExit(f"workload produced no JSON; see {log}")
    return json.loads(text[m.start() : end + 1])


def score(tr: trace.Trace, wl: dict) -> dict:
    trace.attribute(tr)
    names = [tr.name(o["aux"]) if o["op"] == trace.OP_DISPATCH else "" for o in tr.ops]
    calib_idx = [i for i, n in enumerate(names) if n.startswith("calib_")]
    launches = wl["launches"]
    # Warmup launches come first; the scored launches are the trailing ones.
    if len(calib_idx) < len(launches):
        return {
            "error": f"traced {len(calib_idx)} calib dispatches, expected >= {len(launches)}"
        }
    scored = calib_idx[-len(launches) :]
    rows = []
    for li, oi in zip(launches, scored):
        o = tr.ops[oi]
        if names[oi] != li["kernel"]:
            return {
                "error": f"order mismatch: traced {names[oi]} where {li['kernel']} ran"
            }
        dur_us = (int(o["end"]) - int(o["begin"])) / 1e3
        ri = int(tr.op_range[oi])
        rows.append(
            {
                "kernel": li["kernel"],
                "phase": li["phase"],
                "trace_us": dur_us,
                "event_us": li["event_ms"] * 1e3,
                "expect_us": li["expect_us"],
                "bytes": li["expect_bytes"],
                "range": tr.ranges[ri].name if ri >= 0 else "",
            }
        )
    out: dict = {
        "device": wl["device"],
        "arch": wl["arch"],
        "launches": len(rows),
        "validation": trace.validate(tr),
    }
    spin = [r for r in rows if r["kernel"] == "calib_spin"]
    by_phase: dict[str, list] = {}
    for r in spin:
        by_phase.setdefault(r["phase"], []).append(r)
    out["spin"] = {}
    ok = 0
    for ph, rs in by_phase.items():
        exp = rs[0]["expect_us"]
        err = [r["trace_us"] - exp for r in rs]
        eerr = [r["event_us"] - exp for r in rs]
        tol = max(SPIN_ABS_US, SPIN_REL * exp)
        ok += sum(abs(e) <= tol for e in err)
        out["spin"][ph] = {
            "expect_us": exp,
            "trace_err_us_median": st.median(err),
            "trace_err_us_max": max(err, key=abs),
            "event_err_us_median": st.median(eerr),
            "event_err_us_max": max(eerr, key=abs),
            "tolerance_us": tol,
        }
    out["spin_within_tolerance"] = ok / len(spin) if spin else 0.0
    out["memory"] = {}
    for k in ("calib_read", "calib_write", "calib_copy"):
        rs = [r for r in rows if r["kernel"] == k]
        if rs:
            tus = st.median(r["trace_us"] for r in rs)
            eus = st.median(r["event_us"] for r in rs)
            out["memory"][k] = {
                "bytes": rs[0]["bytes"],
                "trace_us_median": tus,
                "event_us_median": eus,
                "trace_gbps": rs[0]["bytes"] / tus / 1e3,
                "trace_vs_event_pct": 100 * (tus - eus) / eus,
            }
    attributed = sum(r["range"] == r["phase"] for r in rows)
    out["attribution_correct"] = attributed / len(rows)
    v = out["validation"]
    out["gates"] = {
        "spin_accuracy": out["spin_within_tolerance"] >= SPIN_QUANTILE,
        "attribution": out["attribution_correct"] == 1.0,
        "clock_causality": v.get("begin_before_call", 1) == 0,
        "no_drops": v["dropped"] == 0,
    }
    out["pass"] = all(out["gates"].values())
    return out


def check(env: HarnessEnv, out_dir: Path, mb: int = 256, iters: int = 20) -> dict:
    env.require("bin")
    exe = exe_path(env)
    if not exe.exists():
        build(env)
    shim = env.bin / "hipgpu.dll"
    cmd = [str(exe), "--mb", str(mb), "--iters", str(iters), "--shim", str(shim)]
    meta = traced.run_command(env, cmd, out_dir, cwd=env.bin)
    wl = _workload_json(Path(out_dir) / "target.log")
    tr = trace.load(out_dir)
    res = score(tr, wl)
    res["collector"] = {
        k: meta.get(k) for k in ("records", "dropped", "api", "api_detail")
    }
    Path(out_dir, "calib_score.json").write_text(json.dumps(res, indent=2))
    return res


def score_kernel_csv(kernel_trace: Path, workload_json: Path) -> dict:
    """Score any rocprofv3-shaped kernel_trace.csv (ktrace's, or ours via
    trace.write_rocprof) against the workload's truth, so tracers are compared
    on identical terms."""
    import csv

    wl = _workload_json(workload_json)
    with open(kernel_trace, newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r["Kernel_Name"].startswith("calib_")]
    rows.sort(key=lambda r: int(r["Start_Timestamp"]))
    launches = wl["launches"]
    rows = rows[-len(launches) :]
    if len(rows) != len(launches):
        return {"error": f"{len(rows)} calib kernels traced, {len(launches)} launched"}
    out: dict = {"spin": {}}
    ok = n = 0
    by_phase: dict[str, list[float]] = {}
    for r, li in zip(rows, launches):
        if not r["Kernel_Name"].startswith(li["kernel"]):
            return {"error": f"order mismatch: {r['Kernel_Name']} vs {li['kernel']}"}
        if li["kernel"] != "calib_spin":
            continue
        dur = (int(r["End_Timestamp"]) - int(r["Start_Timestamp"])) / 1e3
        e = dur - li["expect_us"]
        by_phase.setdefault(li["phase"], []).append(e)
        n += 1
        ok += abs(e) <= max(SPIN_ABS_US, SPIN_REL * li["expect_us"])
    for ph, errs in by_phase.items():
        out["spin"][ph] = {
            "err_us_median": st.median(errs),
            "err_us_max": max(errs, key=abs),
        }
    out["spin_within_tolerance"] = ok / n if n else 0.0
    return out


def rgp_check(
    env: HarnessEnv,
    out_dir: Path,
    iters: int = 20,
    mb: int = 64,
    counters: bool = False,
) -> dict:
    """Score the RGP decoder against the workload: capture every dispatch through
    the Radeon Developer Panel, decode, and compare spin durations with truth.

    With counters, SPM is checked too: the memory traffic SPM attributes to each
    read, write and copy kernel must be within SPM_REL of the bytes it moves by
    construction."""
    from .capture import Panel
    from .rgp import build as rgp_build
    from .rgp import formats as rgp_formats

    env.require("rgp_cli")
    exe = exe_path(env)
    if not exe.exists():
        build(env)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rgp = out_dir / "calib.rgp"
    rgp.unlink(missing_ok=True)
    # The window must close before the process exits or RGP fails the capture,
    # and it closes on a dispatch after its last one, so it stops two short of
    # the workload's own launches (9 warmup + 7 phases x iters). The last copies
    # fall outside it; every spin launch stays inside.
    window = 9 + 7 * iters - 2
    log = out_dir / "target.log"
    harness = Harness(env.out_root)
    with harness.lock():
        harness.stop_competitors()  # a stale panel or service refuses to arm
        panel = Panel(
            env.rgp_cli,
            rgp,
            exe.stem,
            out_dir / "panel.log",
            "minimum",
            window,
            counters=counters,
            trigger=f"dispatch:1:{window}",
        )
        child = env.run_env()
        with log.open("w", encoding="utf-8", errors="replace") as f:
            rc = subprocess.run(
                [
                    str(exe),
                    "--mb",
                    str(mb),
                    "--iters",
                    str(iters),
                    "--hold-ms",
                    "45000",
                ],
                env=child,
                stdout=f,
                stderr=subprocess.STDOUT,
                check=False,
            ).returncode
        panel.finish(5)
        harness.stop_competitors()
    if rc != 0 or not rgp.exists():
        raise SystemExit(f"no capture (exit {rc}); {panel.stall_note()} see {out_dir}")
    wl = _workload_json(log)
    cap = rgp_build.build(rgp)
    rgp_formats.write_all(cap, str(out_dir / "calib"))
    spins = sorted(
        (d for d in cap.dispatches if d.name == "calib_spin"), key=lambda d: d.ts_us
    )
    truth = [x for x in wl["launches"] if x["kernel"] == "calib_spin"]
    res: dict = {
        "capture": str(rgp),
        "decoded_spins": len(spins),
        "launched_spins": len(truth),
    }
    if len(spins) != len(truth):
        res["error"] = "decoded spin count differs from launches; widen the window"
        return res
    phases: dict[str, list] = {}
    for d, t in zip(spins, truth):
        phases.setdefault(t["phase"], []).append((d, t))
    res["spin"] = {}
    for ph, pairs in phases.items():
        exp = pairs[0][1]["expect_us"]
        err = [d.dur_us - exp for d, _ in pairs]
        span = [d.wave_span_us - exp for d, _ in pairs]
        res["spin"][ph] = {
            "expect_us": exp,
            "dur_err_us_median": st.median(err),
            "dur_err_us_max": max(err, key=abs),
            "wave_span_err_us_median": st.median(span),
            "within_tol": sum(abs(e) <= max(SPIN_ABS_US, SPIN_REL * exp) for e in err)
            / len(err),
        }
    allerr = [d.dur_us - t["expect_us"] for d, t in zip(spins, truth)]
    tol = [max(SPIN_ABS_US, SPIN_REL * t["expect_us"]) for t in truth]
    res["within_tolerance"] = sum(abs(e) <= m for e, m in zip(allerr, tol)) / len(
        allerr
    )
    res["pass"] = res["within_tolerance"] >= SPIN_QUANTILE
    if counters:
        res["spm"] = {}
        for kernel in ("calib_read", "calib_write", "calib_copy"):
            moved = [
                d.mem_bytes for d in cap.dispatches if d.name == kernel and d.mem_bytes
            ]
            expect = next(
                (x["expect_bytes"] for x in wl["launches"] if x["kernel"] == kernel), 0
            )
            res["spm"][kernel] = {
                "dispatches_with_spm": len(moved),
                "expect_bytes": expect,
                "spm_bytes_median": st.median(moved) if moved else 0,
                "ratio_median": st.median(moved) / expect if moved and expect else 0,
            }
        res["spm_pass"] = all(
            v["dispatches_with_spm"] and abs(v["ratio_median"] - 1.0) <= SPM_REL
            for v in res["spm"].values()
        )
        res["pass"] = res["pass"] and res["spm_pass"]
    Path(out_dir, "rgp_score.json").write_text(json.dumps(res, indent=2))
    return res


def dispatch(env: HarnessEnv, a) -> int:
    if a.kcmd == "build":
        build(env, a.arch)
        return 0
    if a.kcmd == "score":
        res = score_kernel_csv(a.kernel_trace, a.workload)
        print(json.dumps(res, indent=2))
        return 0 if res.get("spin_within_tolerance", 0) >= SPIN_QUANTILE else 1
    if a.kcmd == "rgp":
        res = rgp_check(env, a.out, a.iters, counters=a.counters)
        print(json.dumps(res, indent=2))
        return 0 if res.get("pass") else 1
    res = check(env, a.out, a.mb, a.iters)
    print(json.dumps(res, indent=2))
    return 0 if res.get("pass") else 1
