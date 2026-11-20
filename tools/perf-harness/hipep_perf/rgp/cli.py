#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""``hipep_perf rgp decode|verify|parity``."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

from . import build, formats

CSV_SUFFIXES = ("_dispatches.csv", "_operators.csv", "_families.csv")


def decode(capture: Path, out_base: Path, legacy_spm: bool = False) -> list[str]:
    cap = build.build(capture, legacy_spm=legacy_spm)
    return formats.write_all(cap, str(out_base))


def _csv_diff(a: Path, b: Path, limit: int = 5) -> list[str]:
    with open(a, newline="") as fa, open(b, newline="") as fb:
        ra, rb = list(csv.reader(fa)), list(csv.reader(fb))
    out = []
    if len(ra) != len(rb):
        out.append(f"row count {len(ra)} vs {len(rb)}")
    hdr = ra[0] if ra else []
    for i, (x, y) in enumerate(zip(ra, rb)):
        if x != y:
            cols = [
                hdr[j] if j < len(hdr) else str(j)
                for j in range(min(len(x), len(y)))
                if x[j] != y[j]
            ]
            out.append(f"row {i}: columns {cols}")
            if len(out) >= limit:
                break
    return out


def _normalize_timeline(path: Path) -> dict:
    doc = json.loads(path.read_text())
    for e in doc["traceEvents"]:
        if e.get("name") == "process_name" and e.get("pid") == 0:
            e["args"]["name"] = "<process>"
    return doc


def parity(corpus: Path, out: Path) -> dict:
    """Decode corpus/rgp/*.rgp and compare with corpus/ref/<name>_* (frozen).

    SPM is decoded the reference decoder's way here, so the comparison covers
    everything else; the corrected SPM handling is checked against the
    calibration workload instead (calib rgp --counters)."""
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    for rgp in sorted((corpus / "rgp").glob("*.rgp")):
        name = rgp.stem
        t0 = time.time()
        decode(rgp, out / name, legacy_spm=True)
        secs = round(time.time() - t0, 1)
        res: dict = {"seconds": secs, "diffs": {}}
        for suf in CSV_SUFFIXES:
            ref, new = corpus / "ref" / (name + suf), out / (name + suf)
            if ref.read_bytes() != new.read_bytes():
                res["diffs"][suf] = _csv_diff(ref, new)
        sref = json.loads((corpus / "ref" / f"{name}_summary.json").read_text())
        snew = json.loads((out / f"{name}_summary.json").read_text())
        for s in (sref, snew):  # the capture's path, wherever the corpus lives
            s["source"] = Path(s.get("source", "")).name
        if sref != snew:
            res["diffs"]["_summary.json"] = sorted(
                k for k in set(sref) | set(snew) if sref.get(k) != snew.get(k)
            )
        if _normalize_timeline(
            corpus / "ref" / f"{name}_trace.json"
        ) != _normalize_timeline(out / f"{name}_trace.json"):
            res["diffs"]["_trace.json"] = ["content differs"]
        res["identical"] = not res["diffs"]
        report[name] = res
        print(
            f"{name}: {'IDENTICAL' if res['identical'] else 'DIFFERS'} ({secs}s)"
            + ("" if res["identical"] else f" {json.dumps(res['diffs'])[:400]}")
        )
    (out / "parity.json").write_text(json.dumps(report, indent=2))
    return report


def dispatch(a) -> int:
    if a.gcmd == "decode":
        for p in decode(a.capture, a.out_base):
            print(f"wrote {p}")
        return 0
    if a.gcmd == "verify":
        ok, detail = build.verify(a.capture, a.need_spm)
        print(("OK " if ok else "UNUSABLE ") + detail)
        return 0 if ok else 1
    rep = parity(a.corpus, a.out)
    return 0 if all(r["identical"] for r in rep.values()) else 1
