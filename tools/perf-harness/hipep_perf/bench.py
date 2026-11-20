#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""One clean TTFT or decode-throughput measurement, appended to a CSV.

These are the only numbers that decide anything. No profiling variable is ever
set for them (see env.PROFILING_VARS); warmup is mandatory because the first
iteration of a fresh process pays autotune and JIT cost.

TTFT wants ``-g 1`` so nothing but the prefill is timed; TPS wants enough
generated tokens for the steady state to dominate. Defaults mirror the Jenkins
genai case (``-g 128 -r 3 -w 1 -b 1 -ml 0`` with a real prompt).
"""

from __future__ import annotations

import csv
import datetime as _dt
import json
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from .env import HarnessEnv
from .proc import Harness

_NUM = r"([\d.eE+\-]+)"
_FIELDS = {
    "avg_us": re.compile(r"avg \(us\):\s+" + _NUM),
    "tps": re.compile(r"avg \(tokens/s\):\s+" + _NUM),
    "p50_us": re.compile(r"p50 \(us\):\s+" + _NUM),
    "stddev_us": re.compile(r"stddev \(us\):\s+" + _NUM),
    "n": re.compile(r"n:\s+(\d+)"),
}
PREFILL_HEADER = "Prompt processing (time to first token)"
DECODE_HEADER = "Token generation"
_PROMPT_TOKENS = re.compile(r"prompt tokens:\s*(\d+)")


def parse_block(lines: list[str], header: str) -> dict[str, float] | None:
    """Fields of one model_benchmark timing block.

    The blocks are adjacent and share field names, so reading stops at the next
    unindented line; a fixed-size window once let the sampler's 14 us overwrite
    token generation's 16,600 us.
    """
    for i, line in enumerate(lines):
        if not line.startswith(header):
            continue
        out: dict[str, float] = {}
        for nxt in lines[i + 1 :]:
            if not nxt[:1].isspace():
                break
            for key, rx in _FIELDS.items():
                m = rx.search(nxt)
                if m:
                    out[key] = int(m.group(1)) if key == "n" else float(m.group(1))
        return out
    return None


def parse_prompt_tokens(lines: list[str]) -> int | None:
    for line in lines:
        m = _PROMPT_TOKENS.search(line)
        if m:
            return int(m.group(1))
    return None


@dataclass
class BenchSpec:
    tag: str
    metric: str  # "ttft" | "tps"
    driver: str = "model_benchmark"  # | "vlm"
    seqlen: int | None = None
    prompt_file: Path | None = None
    gen: int | None = None
    reps: int | None = None
    warmup: int = 1
    max_tokens: int = 4  # vlm ttft only
    max_length: int | None = None  # vlm only
    execution_provider: str = "follow_config"
    set_env: dict[str, str] = field(default_factory=dict)
    out_dir: Path | None = None
    extra_env: dict[str, str] = field(default_factory=dict)  # not recorded; e.g. TEMP


def _defaults(spec: BenchSpec) -> None:
    if spec.metric == "ttft":
        spec.reps = spec.reps or 4
        spec.gen = 1
        if spec.seqlen is None and spec.prompt_file is None:
            spec.seqlen = 16384
    else:
        spec.reps = spec.reps or 3
        spec.gen = spec.gen or 128
        if spec.seqlen is None and spec.prompt_file is None:
            spec.seqlen = 128
    if (
        spec.driver == "model_benchmark"
        and spec.prompt_file
        and spec.seqlen
        and spec.metric == "ttft"
    ):
        raise SystemExit(
            "seqlen and prompt-file are mutually exclusive on the text TTFT "
            "path: the prompt file fixes the length."
        )


def _command(env: HarnessEnv, spec: BenchSpec, json_out: Path) -> list[str]:
    if spec.driver == "vlm":
        env.require("vlm_bench", "image")
        if not spec.prompt_file:
            raise SystemExit(
                "driver vlm needs a prompt file (--prompt-file or HIPEP_PROMPT_FILE)"
            )
        if spec.metric == "ttft":
            max_tokens = spec.max_tokens
            max_length = spec.max_length or (spec.seqlen or 16384) + 128
        else:
            max_tokens = spec.gen
            max_length = spec.max_length or (spec.seqlen or 128) + spec.gen
        driver = Path(__file__).with_name("vlm_driver.py")
        return [
            env.python,
            "-u",
            str(driver),
            "--bin",
            str(env.bin),
            "--prompt-file",
            str(spec.prompt_file),
            str(env.vlm_bench),
            "-m",
            str(env.model),
            "-i",
            str(env.image),
            "--max_tokens",
            str(max_tokens),
            "--max_length",
            str(max_length),
            "-e",
            spec.execution_provider,
            "-n",
            str(spec.reps),
            "-w",
            str(spec.warmup),
            "-o",
            str(json_out),
        ]
    cmd = [
        str(env.bin / "model_benchmark.exe"),
        "-i",
        str(env.model),
        "-g",
        str(spec.gen),
        "-r",
        str(spec.reps),
        "-w",
        str(spec.warmup),
        "-b",
        "1",
        "-ml",
        "0",
        "-v",
    ]
    if spec.prompt_file:
        cmd += ["--prompt_file", str(spec.prompt_file)]
    else:
        cmd += ["-l", str(spec.seqlen), "--use_random_tokens"]
    return cmd


def run_driver(
    env: HarnessEnv,
    harness: Harness,
    spec: BenchSpec,
    log: Path,
    json_out: Path,
    child_env: dict[str, str],
) -> int:
    harness.stop_competitors()
    json_out.unlink(missing_ok=True)
    cmd = _command(env, spec, json_out)
    with log.open("w", encoding="utf-8", errors="replace") as f:
        p = subprocess.Popen(
            cmd, cwd=str(env.bin), env=child_env, stdout=f, stderr=subprocess.STDOUT
        )
        if spec.driver == "vlm":
            harness.register_driver(p.pid)
        return _wait(p, log if spec.driver != "vlm" else None)


# model_benchmark's last line. The process has occasionally stayed in GPU driver
# teardown after it, where even TerminateProcess cannot finish it and waiting
# on its handle blocks forever.
_DONE_LINE = "Total GPU memory usage"


def _wait(p: subprocess.Popen, log: Path | None, grace: float = 60) -> int:
    done_at = None
    while True:
        try:
            return p.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass
        if log is None:
            continue
        if done_at is None:
            if _DONE_LINE in log.read_text(errors="replace"):
                done_at = time.time()
        elif time.time() - done_at > grace:
            print(
                f"    !! pid {p.pid} still alive {grace:.0f}s after its output ended; killing"
            )
            p.kill()
            try:
                return p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                print(
                    f"    !! pid {p.pid} did not terminate (stuck in the driver); moving on"
                )
                return -1


def _artifacts(env: HarnessEnv) -> list[str]:
    out = []
    for pattern in ("custom_kernels_*.dll", "hipgpu.dll"):
        for p in sorted(env.bin.glob(pattern)):
            st = p.stat()
            ts = _dt.datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds")
            out.append(f"{ts}  {st.st_size / 2**20:.2f} MB  {p.name}")
    return out


def _append_csv(path: Path, row: dict) -> None:
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def run(env: HarnessEnv, spec: BenchSpec) -> dict | None:
    """Run one measurement; returns the CSV row, or None if parsing failed."""
    env.require("bin", "model")
    _defaults(spec)
    if spec.prompt_file is None and spec.driver == "vlm":
        spec.prompt_file = env.prompt_file
    harness = Harness(env.out_root)
    out_dir = spec.out_dir or env.out_root / spec.metric
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / f"{spec.metric}_{spec.tag}.log"
    json_out = out_dir / f"{spec.metric}_{spec.tag}.json"
    csv_path = out_dir / f"{spec.metric}_summary.csv"

    with harness.lock():
        child = env.run_env()
        child.update(spec.set_env)
        child.update(spec.extra_env)
        workload = (
            f"prompt={spec.prompt_file.name}"
            if spec.prompt_file
            else f"seqlen={spec.seqlen} (random ids)"
        )
        print(
            f">>> {spec.metric.upper()} [{spec.tag}] driver={spec.driver} {workload} "
            f"-g {spec.gen} -r {spec.reps} -w {spec.warmup}"
        )
        for k, v in spec.set_env.items():
            print(f"    env {k}={v}")
        for line in _artifacts(env):
            print(f"      {line}")
        rc = run_driver(env, harness, spec, log, json_out, child)
        row = _parse(spec, log, json_out)
        if row is None:
            print(
                f"\n=== {spec.metric.upper()} [{spec.tag}] PARSE FAILED (exit={rc}) -- inspect {log}"
            )
            print("\n".join(log.read_text(errors="replace").splitlines()[-25:]))
            return None
        row["when"] = _dt.datetime.now().isoformat(timespec="seconds")
        _append_csv(csv_path, row)
        _echo(spec, row, rc)
        print(f"    appended -> {csv_path}")
        return row


def run_perf_ops(env: HarnessEnv, spec: BenchSpec, ops: list[str]) -> list[Path]:
    """Per-family GPU ms from event pairs (HIPDNN_EP_PERF_OPS), one process per op.

    Event pairs around one family would perturb another's, hence one process
    each. The throughput of these runs is not valid and is never appended to a
    summary CSV; the logs feed analysis.calibrate_sqtt.
    """
    env.require("bin", "model")
    spec.metric = "tps"
    _defaults(spec)
    harness = Harness(env.out_root)
    out_dir = spec.out_dir or env.out_root / "tps"
    out_dir.mkdir(parents=True, exist_ok=True)
    logs = []
    with harness.lock():
        for op in ops:
            child = env.run_env()
            child.update(spec.set_env)
            child.update(spec.extra_env)
            child["HIPDNN_EP_PERF_OPS"] = op
            log = out_dir / f"tps_{spec.tag}.perfops_{op}.log"
            print(f">>> PERF_OPS [{spec.tag}] op={op} (throughput not valid)")
            rc = run_driver(
                env,
                harness,
                spec,
                log,
                out_dir / f"tps_{spec.tag}.perfops_{op}.json",
                child,
            )
            print(f"    exit={rc} -> {log}")
            logs.append(log)
    return logs


def _parse(spec: BenchSpec, log: Path, json_out: Path) -> dict | None:
    if spec.driver == "vlm":
        if not json_out.exists():
            return None
        s = json.loads(json_out.read_text())["summary"]
        tb = s.get("token_breakdown", {})
        prompt = tb.get("prompt_tokens", spec.seqlen)
        pre = {
            "avg_us": s["ttft_ms"]["avg"] * 1000,
            "p50_us": s["ttft_ms"]["p50"] * 1000,
            "stddev_us": s["ttft_ms"]["std"] * 1000,
        }
        tg = s.get("token_generation_ms")
        dec = (
            None
            if tg is None
            else {
                "avg_us": tg["avg"] * 1000,
                "tps": tg["tps_avg"],
                "p50_us": tg["p50"] * 1000,
                "stddev_us": tg["std"] * 1000,
                "n": s.get("total_runs"),
            }
        )
    else:
        lines = log.read_text(errors="replace").splitlines()
        prompt = parse_prompt_tokens(lines) or spec.seqlen
        pre = parse_block(lines, PREFILL_HEADER)
        dec = parse_block(lines, DECODE_HEADER)
    if spec.metric == "ttft":
        if not pre or "avg_us" not in pre:
            return None
        return {
            "tag": spec.tag,
            "ttft_ms": round(pre["avg_us"] / 1000, 1),
            "p50_ms": round(pre.get("p50_us", 0) / 1000, 1),
            "stddev_ms": round(pre.get("stddev_us", 0) / 1000, 1),
            "seqlen": prompt,
            "reps": spec.reps,
        }
    if not dec or not dec.get("tps"):
        return None
    return {
        "tag": spec.tag,
        "prompt_tokens": prompt,
        "tps": round(dec["tps"], 2),
        "ms_per_token": round(dec["avg_us"] / 1000, 3),
        "p50_ms": round(dec.get("p50_us", 0) / 1000, 3),
        "stddev_ms": round(dec.get("stddev_us", 0) / 1000, 3),
        "ttft_ms": round(pre["avg_us"] / 1000, 1) if pre and "avg_us" in pre else "",
        "gen": spec.gen,
        "reps": spec.reps,
        "n": dec.get("n", ""),
    }


def _echo(spec: BenchSpec, row: dict, rc: int) -> None:
    if spec.metric == "ttft":
        print(
            f"\n=== TTFT [{spec.tag}] = {row['ttft_ms']:,.0f} ms   p50 {row['p50_ms']:,.0f} ms   "
            f"stddev {row['stddev_ms']:,.0f} ms   (exit={rc})"
        )
    else:
        print(
            f"\n=== TPS [{spec.tag}] = {row['tps']:.2f} tok/s   ({row['ms_per_token']:.2f} ms/token, "
            f"p50 {row['p50_ms']:.2f}, stddev {row['stddev_ms']:.2f})   n={row['n']} (exit={rc})"
        )
