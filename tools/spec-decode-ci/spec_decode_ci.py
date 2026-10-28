#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

r"""
Speculative-decoding end-to-end check for CI.

Runs the Spec-Bench harness that ships with a hip-ep speculator model repo
(e.g. amd/Qwen3.8-27B-DFlash2-Quark-hip-ep, scripts/qwen38_spec_bench.py) on a fixed
prompt set, then gates the pooled result against a baseline file:

  * decode tok/s of every arm         >= reference * (1 - tolerance.tok_s)
  * speedup of every drafter arm      >= reference * (1 - tolerance.speedup)
  * tokens per verify call (accepted) >= reference * (1 - tolerance.tokens_per_verify)

Plain decode and the drafter arms run in one process on the same prompts, so
the speedup is a paired ratio and does not depend on board-to-board noise.

The baseline file names the harness, its arguments and its output format, so
one driver serves every speculator repo. Harness arguments may use the
placeholders {model} (model repo checkout), {ep_dir} (folder with hipgpu.dll),
{therock_bin} and {n_prompts}; the driver appends --arms, --questions, -n and
--json itself.

Contract with the CI scenario runner: called as
``<python> spec_decode_ci.py -m <model_dir> -o <output.json> [options]``, writes
the verdict to <output.json> and exits non-zero on any failed check.

Usage:
    # run + gate (default baseline: baselines/qwen38_27b_dflash2.json)
    python tools/spec-decode-ci/spec_decode_ci.py -m <model_dir> -o out.json \
        --ep-dir <dir with hipgpu.dll> --therock-bin <TheRock bin>

    # re-gate an existing harness JSON without a GPU
    python tools/spec-decode-ci/spec_decode_ci.py -o out.json --evaluate harness.json

    # record the board's own numbers as the new reference
    python tools/spec-decode-ci/spec_decode_ci.py -m <model_dir> -o out.json \
        --calibrate baselines/qwen38_27b_dflash2.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BASELINE = SCRIPT_DIR / "baselines" / "qwen38_27b_dflash2.json"

# (session name or None to take it from group 1, pattern whose last group is seconds)
LOAD_PATTERNS = [
    (None, re.compile(r"^\[session\] (\S+) .* load ([\d.]+)s", re.MULTILINE)),
    ("target", re.compile(r"^target\s+.* loaded in ([\d.]+) s", re.MULTILINE)),
    (None, re.compile(r"^draft\s+(\S+) loaded in ([\d.]+) s", re.MULTILINE)),
    ("warmup", re.compile(r"^\s+done in ([\d.]+) s", re.MULTILINE)),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_baseline(path: Path) -> dict:
    if not path.is_absolute():
        path = SCRIPT_DIR / path
    cfg = json.loads(path.read_text(encoding="utf-8"))
    cfg["_path"] = path
    cfg["_prompts"] = (path.parent / cfg["prompts"]).resolve()
    return cfg


def resolve_ep_dir(arg: str | None) -> Path:
    candidates = [arg, os.environ.get("HIPEP_EP_BIN"), os.getcwd()]
    for c in candidates:
        if c and (Path(c) / "hipgpu.dll").is_file():
            return Path(c).resolve()
    raise FileNotFoundError(
        "hipgpu.dll not found; pass --ep-dir, set HIPEP_EP_BIN or run from the "
        "gpu-test-package bin directory"
    )


def resolve_therock_bin(arg: str | None) -> Path:
    if arg:
        return Path(arg).resolve()
    dist = os.environ.get("THEROCK_DIST")
    if dist and (Path(dist) / "bin").is_dir():
        return (Path(dist) / "bin").resolve()
    for d in filter(None, os.environ.get("PATH", "").split(os.pathsep)):
        if (Path(d) / "amdhip64_7.dll").is_file():
            return Path(d).resolve()
    raise FileNotFoundError("TheRock bin not found; pass --therock-bin, set THEROCK_DIST "
                            "or put the folder with amdhip64_7.dll on PATH")


def harness_values(cfg: dict, a: argparse.Namespace) -> dict:
    used = json.dumps(cfg["harness"])
    values = {"model": str(Path(a.model_dir).resolve()), "n_prompts": str(len(cfg["question_ids"]))}
    if "{ep_dir}" in used:
        values["ep_dir"] = str(resolve_ep_dir(a.ep_dir))
    if "{therock_bin}" in used:
        values["therock_bin"] = str(resolve_therock_bin(a.therock_bin))
    return values


def run_harness(cfg: dict, a: argparse.Namespace, harness_json: Path, log: Path) -> tuple[int, float]:
    h = cfg["harness"]
    values = harness_values(cfg, a)
    cmd = [
        a.python or sys.executable,
        "-u",
        str(Path(values["model"]) / h["script"]),
        *(arg.format(**values) for arg in h.get("args", [])),
        "--arms", ",".join(cfg["arms"]),
        "--questions", str(cfg["_prompts"]),
        "-n", str(cfg["max_new"]),
        "--json", str(harness_json),
    ]
    env = dict(os.environ)
    env.update(cfg.get("env", {}))
    env["PYTHONIOENCODING"] = "utf-8"
    prepend = [p.format(**values) for p in h.get("path_prepend", [])]
    if prepend:
        env["PATH"] = os.pathsep.join(prepend + [env.get("PATH", "")])
    print("$ " + " ".join(cmd), flush=True)
    t0 = time.perf_counter()
    with open(log, "w", encoding="utf-8") as lf:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                env=env, text=True, encoding="utf-8", errors="replace")
        for line in proc.stdout:
            sys.stdout.write(line)
            lf.write(line)
        try:
            rc = proc.wait(timeout=a.timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            rc = -1
    return rc, time.perf_counter() - t0


def load_times(text: str) -> dict:
    out = {}
    for name, pattern in LOAD_PATTERNS:
        for m in pattern.finditer(text):
            out[name or m.group(1)] = float(m.groups()[-1])
    return out


def prompts_spec_bench(result: dict) -> list[dict]:
    """scripts/spec_bench.py (EAGLE-4 repos) already writes the common form."""
    return result.get("prompts", [])


def prompts_qwen38_spec_bench(result: dict) -> list[dict]:
    """scripts/qwen38_spec_bench.py (MTP / DSpark / DFlash2 repos).

    Each prompt's first token comes out of prefill, so it is left out of both the
    decode tokens and the tokens per verify call, as the harness's own summary does.
    """
    out = []
    for rec in result.get("records", []):
        arms = {}
        base = rec.get("arms", {}).get("none")
        for name, r in rec.get("arms", {}).items():
            arms[name] = {
                "new_tokens": r["tokens"] - 1,
                "decode_s": r["t_decode"],
                "iters": r["iterations"],
                "accepted": sum(r["accepted"]),
                "drafted": sum(r["proposed"]),
                "tok_s": r["tps_decode"],
                "identical_to_none": bool(base) and r["sha1"] == base["sha1"],
            }
        out.append({"question_id": rec["question_id"], "task": rec["subtask"],
                    "prompt_len": rec["prompt_tokens"], "arms": arms,
                    **({"skipped": rec["skipped"]} if "skipped" in rec else {})})
    return out


RESULT_FORMATS = {
    "spec_bench": prompts_spec_bench,
    "qwen38_spec_bench": prompts_qwen38_spec_bench,
}


def pooled(prompts: list[dict], arm: str) -> dict:
    n = sum(p["arms"][arm]["new_tokens"] for p in prompts)
    t = sum(p["arms"][arm]["decode_s"] for p in prompts)
    out = {"tok_s": n / t if t else 0.0, "tokens": n, "decode_s": t}
    if arm != "none":
        iters = sum(p["arms"][arm]["iters"] for p in prompts)
        out["tokens_per_verify"] = n / iters if iters else 0.0
        out["accepted"] = sum(p["arms"][arm]["accepted"] for p in prompts)
        out["drafted"] = sum(p["arms"][arm]["drafted"] for p in prompts)
        out["identical_to_none"] = sum(bool(p["arms"][arm]["identical_to_none"]) for p in prompts)
    return out


def evaluate(cfg: dict, result: dict) -> dict:
    prompts = RESULT_FORMATS[cfg["harness"]["format"]](result)
    base_arm = cfg["baseline_arm"]
    ref = cfg["reference"]["arms"]
    tol = cfg["tolerance"]
    checks: list[dict] = []

    def check(name, value, threshold, ok):
        checks.append({"name": name, "value": value, "threshold": threshold, "pass": bool(ok)})

    got_ids = sorted(int(p["question_id"]) for p in prompts)
    want_ids = sorted(cfg["question_ids"])
    check("prompt set", got_ids, want_ids, got_ids == want_ids)

    arms = {}
    missing = sorted({f"{arm}@q{p['question_id']}" for p in prompts for arm in cfg["arms"]
                      if arm not in p["arms"]})
    check("all arms decoded", missing, [], not missing and bool(prompts))
    if missing or not prompts:
        return {"arms": arms, "checks": checks}

    for arm in cfg["arms"]:
        arms[arm] = pooled(prompts, arm)
    for arm in cfg["arms"]:
        cur, r = arms[arm], ref.get(arm, {})
        if arm != base_arm:
            cur["speedup"] = cur["tok_s"] / arms[base_arm]["tok_s"]
        for key in ("tok_s", "speedup", "tokens_per_verify"):
            if key in cur and key in r:
                floor = r[key] * (1 - tol[key])
                check(f"{arm} {key}", round(cur[key], 3), round(floor, 3), cur[key] >= floor)
                cur[f"{key}_vs_ref"] = cur[key] / r[key] - 1

    per_prompt = []
    for p in sorted(prompts, key=lambda p: p["question_id"]):
        row = {"question_id": p["question_id"], "task": p["task"], "prompt_len": p["prompt_len"]}
        for arm in cfg["arms"]:
            st = p["arms"][arm]
            row[arm] = {"tok_s": st["tok_s"], "new_tokens": st["new_tokens"]}
            if arm != base_arm:
                row[arm]["speedup"] = st["tok_s"] / p["arms"][base_arm]["tok_s"]
        per_prompt.append(row)
    return {"arms": arms, "checks": checks, "per_prompt": per_prompt}


def calibrated_reference(cfg: dict, arms: dict) -> dict:
    ref = {}
    for arm, cur in arms.items():
        ref[arm] = {k: round(cur[k], 3) for k in ("tok_s", "speedup", "tokens_per_verify") if k in cur}
    return ref


def print_report(verdict: dict) -> None:
    print("\n== speculative decoding CI ==")
    for arm, cur in verdict.get("arms", {}).items():
        extra = ""
        if "speedup" in cur:
            extra = (f"  speedup {cur['speedup']:.3f}x  tokens/verify {cur['tokens_per_verify']:.2f}"
                     f"  accepted {cur['accepted']}/{cur['drafted']}"
                     f"  identical {cur['identical_to_none']}")
        print(f"  {arm:18s} {cur['tok_s']:7.2f} tok/s{extra}")
    for c in verdict.get("checks", []):
        bound = "expected" if isinstance(c["threshold"], list) else "min"
        print(f"  [{'PASS' if c['pass'] else 'FAIL'}] {c['name']}: {c['value']} ({bound} {c['threshold']})")
    if verdict.get("session_load_s"):
        print("  load/compile s: " + ", ".join(f"{k} {v:.1f}" for k, v in verdict["session_load_s"].items()))
    if "wall_s" in verdict:
        print(f"  wall: {verdict['wall_s'] / 60:.1f} min")
    print(f"  status: {verdict['status']}", flush=True)


def speed_only_failure(part: dict) -> bool:
    """Timing can be off on a cold run; prompt-set, decode and acceptance failures cannot."""
    failed = [c["name"] for c in part.get("checks", []) if not c["pass"]]
    return "error" not in part and bool(failed) and all(
        name.endswith((" tok_s", " speedup")) for name in failed)


def run_attempt(cfg: dict, a: argparse.Namespace, out: Path, attempt: int) -> dict:
    part: dict = {}
    if a.evaluate:
        harness_json = Path(a.evaluate).resolve()
    else:
        stem = out.stem + (f".retry{attempt}" if attempt else "")
        harness_json = out.with_name(stem + ".harness.json")
        log = out.with_name(stem + ".harness.log")
        try:
            rc, wall = run_harness(cfg, a, harness_json, log)
        except FileNotFoundError as exc:
            return {"status": "FAIL", "error": str(exc)}
        text = log.read_text(encoding="utf-8", errors="replace")
        part.update(harness_returncode=rc, wall_s=round(wall, 1), session_load_s=load_times(text))
        if rc != 0:
            part.update(status="FAIL", error=f"harness exited with {rc} (log: {log})")
            return part
    result = json.loads(harness_json.read_text(encoding="utf-8"))
    part["harness_json"] = str(harness_json)
    part.update(evaluate(cfg, result))
    part["status"] = "PASS" if all(c["pass"] for c in part["checks"]) else "FAIL"
    return part


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("-m", "--model-dir", help="speculator model repo checkout")
    ap.add_argument("-o", "--output", required=True, help="verdict JSON")
    ap.add_argument("--baseline", default=str(DEFAULT_BASELINE),
                    help="baseline JSON; a relative path resolves against this script's directory")
    ap.add_argument("--ep-dir", default=None, help="directory holding hipgpu.dll")
    ap.add_argument("--therock-bin", default=None, help="TheRock bin (default $THEROCK_DIST/bin)")
    ap.add_argument("--python", default=None, help="interpreter for the harness (default: this one)")
    ap.add_argument("--timeout", type=int, default=1800, help="harness timeout in seconds")
    ap.add_argument("--retries", type=int, default=1,
                    help="re-runs when only tok/s or speedup checks fail (a cold first run "
                         "after boot or download is up to ~13%% slower)")
    ap.add_argument("--evaluate", default=None, help="gate an existing harness JSON instead of running")
    ap.add_argument("--calibrate", default=None,
                    help="write the measured numbers as the reference of this baseline file")
    a = ap.parse_args()

    cfg = load_baseline(Path(a.baseline))
    out = Path(a.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    verdict = {"baseline": str(cfg["_path"]), "calibrated": cfg["reference"].get("calibrated", False)}

    prompt_sha = sha256(cfg["_prompts"])
    if prompt_sha != cfg["prompts_sha256"]:
        verdict.update(status="FAIL", error=f"prompt file {cfg['_prompts']} sha256 {prompt_sha} "
                                            f"!= {cfg['prompts_sha256']}")
        out.write_text(json.dumps(verdict, indent=1), encoding="utf-8")
        print(f"[ERROR] {verdict['error']}", flush=True)
        return 1

    if not a.evaluate and not a.model_dir:
        ap.error("-m/--model-dir is required unless --evaluate is given")
    retries = 0 if (a.evaluate or a.calibrate) else a.retries
    attempts = []
    for attempt in range(retries + 1):
        if attempts:
            print(f"[INFO] only speed checks failed; retrying ({attempt}/{retries})", flush=True)
        part = run_attempt(cfg, a, out, attempt)
        attempts.append(part)
        if part["status"] == "PASS" or not speed_only_failure(part):
            break
    verdict.update(attempts[-1])
    if len(attempts) > 1:
        verdict["attempts"] = [
            {"status": p["status"], "wall_s": p.get("wall_s"),
             "failed": [c["name"] for c in p.get("checks", []) if not c["pass"]]}
            for p in attempts
        ]
        verdict["wall_s"] = round(sum(p.get("wall_s", 0) for p in attempts), 1)
    if "error" in verdict:
        out.write_text(json.dumps(verdict, indent=1), encoding="utf-8")
        print(f"[ERROR] {verdict['error']}", flush=True)
        return 1

    if a.calibrate:
        if not verdict.get("arms") or not all(
                c["pass"] for c in verdict["checks"] if c["name"] in ("prompt set", "all arms decoded")):
            verdict["status"] = "FAIL"
        else:
            target = Path(a.calibrate)
            new_cfg = json.loads(target.read_text(encoding="utf-8"))
            new_cfg["reference"] = {
                "calibrated": True,
                "source": f"calibration run {time.strftime('%Y-%m-%d')} ({harness_json.name})",
                "arms": calibrated_reference(cfg, verdict["arms"]),
            }
            target.write_text(json.dumps(new_cfg, indent=2) + "\n", encoding="utf-8")
            verdict["status"] = "CALIBRATED"
            print(f"[INFO] wrote reference to {target}", flush=True)

    out.write_text(json.dumps(verdict, indent=1), encoding="utf-8")
    print_report(verdict)
    return 0 if verdict["status"] in ("PASS", "CALIBRATED") else 1


if __name__ == "__main__":
    raise SystemExit(main())
