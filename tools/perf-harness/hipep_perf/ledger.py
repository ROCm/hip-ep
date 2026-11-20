#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Prediction ledger: every sized candidate, then what the A/B actually measured.

The ranking tools (headroom, decode_model) produce estimates from captures,
which run at pinned clocks and stretch gaps. The only check on whether those
estimates can be trusted is to record each one before the change is built and
compare it with the interleaved A/B afterwards. The report shows the error per
source, so a source that keeps over-promising is visible.

Stored as JSON lines in $HIPEP_OUT/ledger.jsonl (override with HIPEP_LEDGER).
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import statistics as st
from pathlib import Path

from .env import HarnessEnv


def _path(env: HarnessEnv) -> Path:
    p = os.environ.get("HIPEP_LEDGER")
    return Path(p) if p else env.out_root / "ledger.jsonl"


def _load(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                out.setdefault(rec["id"], {}).update(rec)
    return out


def _append(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rec["when"] = _dt.datetime.now().isoformat(timespec="seconds")
    with path.open("a") as f:
        f.write(json.dumps(rec) + "\n")


def predict(
    env: HarnessEnv, cid: str, metric: str, pct: float, source: str, note: str
) -> None:
    _append(
        _path(env),
        {
            "id": cid,
            "metric": metric,
            "predicted_pct": pct,
            "source": source,
            "note": note,
        },
    )


def measure(env: HarnessEnv, cid: str, summary_json: Path, arm: str) -> dict:
    s = json.loads(Path(summary_json).read_text())
    hit = next((r for r in s["arms"] if r["arm"] == arm), None)
    if hit is None:
        raise SystemExit(f"arm {arm!r} not in {summary_json}")
    rec = {
        "id": cid,
        "measured_pct": hit["pct"],
        "measured_low": hit["low"],
        "measured_high": hit["high"],
        "measured_n": hit["n"],
        "measured_spans_zero": hit["spans_zero"],
        "summary": str(summary_json),
    }
    _append(_path(env), rec)
    return rec


def report(env: HarnessEnv) -> list[dict]:
    rows = list(_load(_path(env)).values())
    print(f"{'id':<28}{'metric':>7}{'pred %':>9}{'meas %':>9}{'error':>8}  source")
    errs: dict[str, list[float]] = {}
    for r in rows:
        meas = r.get("measured_pct")
        pred = r.get("predicted_pct")
        err = (meas - pred) if meas is not None and pred is not None else None
        flag = " (spans zero)" if r.get("measured_spans_zero") else ""
        print(
            f"{r['id']:<28}{r.get('metric', ''):>7}{pred if pred is not None else '-':>9}"
            f"{f'{meas:+.2f}' if meas is not None else '-':>9}"
            f"{f'{err:+.2f}' if err is not None else '-':>8}  {r.get('source', '')}{flag}"
        )
        if err is not None:
            errs.setdefault(r.get("source", ""), []).append(err)
    if errs:
        print("\nprediction error by source (measured - predicted, percentage points):")
        for src, e in errs.items():
            med = st.median(e)
            print(
                f"  {src or '(none)':<24} n={len(e):<3} median {med:+.2f}  "
                f"worst {max(e, key=abs):+.2f}"
            )
    return rows


def dispatch(env: HarnessEnv, a) -> int:
    if a.lcmd == "predict":
        predict(env, a.id, a.metric, a.predicted_pct, a.source, a.note)
    elif a.lcmd == "measure":
        print(json.dumps(measure(env, a.id, a.summary_json, a.arm), indent=2))
    else:
        report(env)
    return 0
