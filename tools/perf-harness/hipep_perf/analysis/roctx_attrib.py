#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Op attribution by ROCTx range, and the kernel-family map derived from it.

SQTT carries no op boundaries, so capture analysis assigns kernels to EP ops
through a family -> op map (perfcommon.load_op_map). A shim trace knows the op
directly: each GPU command is charged to the hip_ep.op.<op> range open on the
thread whose HIP call issued it. --emit-map writes the map that evidence gives
(each family to the op that launched most of its dispatches); without it, the
current map is compared with ROCTx on the given traces.

usage: roctx_attrib <trace dir>... [--op-map FILE] [--map OP=FAM,...]
                    [--emit-map FILE] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

from hipep_perf import trace
from hipep_perf.rgp import kernels, symbols

from .perfcommon import op_family_map, op_for_family


def _attributed(tr: trace.Trace):
    """(kernel family, ROCTx op, GPU ns) per dispatch charged to an op range."""
    trace.attribute(tr)
    names = sorted({tr.name(o["aux"]) for o in tr.ops if o["op"] == trace.OP_DISPATCH})
    dem = symbols.demangle(names)
    family_of = {n: kernels.parse(symbols.simplify(dem[n])).family for n in names}
    for i, o in enumerate(tr.ops):
        if o["op"] != trace.OP_DISPATCH:
            continue
        ri = trace._innermost_op_range(tr, int(tr.op_range[i]))
        if ri >= 0:
            yield (
                family_of[tr.name(o["aux"])],
                trace.op_family(tr.ranges[ri].name),
                int(o["end"]) - int(o["begin"]),
            )


def derive_map(traces: list[tuple[str, trace.Trace]]) -> dict:
    """Each kernel family -> the op that launched most of its dispatches."""
    counts: dict[str, Counter] = defaultdict(Counter)
    for _, tr in traces:
        for fam, op, _ in _attributed(tr):
            counts[fam][op] += 1
    fams = {}
    for fam, c in sorted(counts.items()):
        op, n = c.most_common(1)[0]
        fams[fam] = {"op": op, "share": round(n / sum(c.values()), 4), "ops": dict(c)}
    return {"sources": [src for src, _ in traces], "families": fams}


def compare(tr: trace.Trace, fmap: dict) -> dict:
    per_op: dict[str, Counter] = defaultdict(Counter)
    per_family: dict[str, Counter] = defaultdict(Counter)
    agree = total = 0
    gpu_ns = Counter()
    for fam, roctx_op, dur in _attributed(tr):
        map_op = op_for_family(fam, fmap) or "(unmapped)"
        total += 1
        same = map_op == roctx_op
        agree += same
        per_op[roctx_op]["agree" if same else f"map:{map_op}"] += 1
        per_family[fam][roctx_op] += 1
        gpu_ns["agree" if same else "disagree"] += dur
    return {
        "dispatches_compared": total,
        "agreement": agree / total if total else 0.0,
        "gpu_time_agreement": gpu_ns["agree"]
        / max(1, gpu_ns["agree"] + gpu_ns["disagree"]),
        "per_op": {k: dict(v) for k, v in sorted(per_op.items())},
        "families_spanning_ops": {
            f: dict(c) for f, c in per_family.items() if len(c) > 1
        },
        "per_family": {f: dict(c) for f, c in sorted(per_family.items())},
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="roctx_attrib", description=__doc__.split("\n\n")[0]
    )
    ap.add_argument("trace_dirs", nargs="+")
    ap.add_argument("--op-map", help="family map file (default: the checked-in one)")
    ap.add_argument("--map", action="append", help="perfcommon --map override(s)")
    ap.add_argument("--emit-map", help="write the ROCTx-derived family map here")
    ap.add_argument("--json")
    a = ap.parse_args()
    traces = [(d, trace.load(d)) for d in a.trace_dirs]
    if a.emit_map:
        derived = derive_map(traces)
        with open(a.emit_map, "w") as fh:
            json.dump(derived, fh, indent=2)
            fh.write("\n")
        split = {f: d["ops"] for f, d in derived["families"].items() if d["share"] < 1}
        print(f"wrote {a.emit_map}: {len(derived['families'])} families")
        if split:
            print(
                "families launched under more than one op (assigned to the majority):"
            )
            for f, c in split.items():
                print(f"  {f:<36} {c}")
    fmap = op_family_map(a.map, a.emit_map or a.op_map)
    results = {}
    for src, tr in traces:
        res = compare(tr, fmap)
        results[src] = res
        print(
            f"{src}: dispatches {res['dispatches_compared']}  agreement "
            f"{100 * res['agreement']:.1f}%  (by GPU time "
            f"{100 * res['gpu_time_agreement']:.1f}%)"
        )
        for op, c in res["per_op"].items():
            bad = {k: v for k, v in c.items() if k != "agree"}
            if bad:
                print(f"  {op:<28} agree={c.get('agree', 0):>7}  disagree={bad}")
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(results, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
