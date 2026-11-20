#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Turn a collector journal into tables, attribution, reports and timelines.

Attribution follows the correlation id: a GPU command carries the id of the HIP
call that issued it, the call record says which thread made it and when, and
the command is charged to the innermost ROCTx range open on that thread at that
moment. Host time cannot be used directly, because GPU work runs after the host
range that enqueued it has closed.

All timestamps are QPC nanoseconds. ``validate`` checks that assumption on
every run: a command cannot start before the call that issued it.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import ring
from .collector import load_journal

OP_DISPATCH, OP_COPY, OP_BARRIER = 0, 1, 2
OP_NAMES = {OP_DISPATCH: "dispatch", OP_COPY: "copy", OP_BARRIER: "barrier"}


@dataclass
class Range:
    tid: int
    start: int
    end: int
    name: str
    depth: int
    parent: int  # index into Trace.ranges, -1 at the root


@dataclass
class Trace:
    meta: dict
    names: dict[int, str]
    ops: np.ndarray  # structured, see _OPS
    api: np.ndarray  # structured, see _API
    ranges: list[Range] = field(default_factory=list)
    marks: list[tuple[int, int, str]] = field(default_factory=list)  # (ts, tid, name)
    # (ts, tid, "RANGE_PUSH" | "RANGE_POP" | "MARK", message) in program order.
    roctx: list[tuple[int, int, str, str]] = field(default_factory=list)
    fences: list[dict] = field(default_factory=list)
    device: dict = field(default_factory=dict)  # see ring.REC_DEVICE
    # correlation id -> "name=value ..." call arguments; only ktrace spools have
    # them (the shim does not marshal arguments).
    api_args: dict[int, str] = field(default_factory=dict)
    unbalanced_pops: int = 0
    # Filled by attribute(): per op, index of innermost range (-1 if none).
    op_range: np.ndarray | None = None
    op_tid: np.ndarray | None = None
    op_enter: np.ndarray | None = None

    def name(self, nid: int) -> str:
        return self.names.get(int(nid), f"name#{int(nid)}")

    def api_name(self, api_id: int) -> str:
        return self.names.get(ring.API_NAME_BASE | int(api_id), f"api#{int(api_id)}")


_OPS = np.dtype(
    [
        ("corr", "<u8"),
        ("begin", "<u8"),
        ("end", "<u8"),
        ("queue", "<u8"),
        ("kind", "<u4"),
        ("op", "<u1"),
        ("device", "<u4"),
        ("aux", "<u8"),
        ("tid", "<u4"),
        ("seq", "<u8"),  # arrival order from the runtime, before sorting by begin
    ]
)
_API = np.dtype(
    [
        ("corr", "<u8"),
        ("enter", "<u8"),
        ("exit", "<u8"),
        ("api_id", "<u4"),
        ("tid", "<u4"),
    ]
)


def build(recs: np.ndarray, meta: dict) -> Trace:
    names = ring.decode_names(recs)
    t = recs["type"]
    f = recs["f"]

    sel = t == ring.REC_OP
    o = np.empty(int(sel.sum()), dtype=_OPS)
    fo = f[sel]
    o["corr"], o["begin"], o["end"], o["queue"] = fo[:, 0], fo[:, 1], fo[:, 2], fo[:, 3]
    o["kind"] = fo[:, 4] & 0xFFFFFFFF
    o["op"] = (fo[:, 4] >> 32) & 0xFF
    o["device"] = (fo[:, 4] >> 40) & 0xFFFFFF
    o["aux"] = fo[:, 5]
    o["tid"] = recs["tid"][sel]
    o["seq"] = np.arange(len(o))

    sel = t == ring.REC_API
    a = np.empty(int(sel.sum()), dtype=_API)
    fa = f[sel]
    a["corr"], a["enter"], a["exit"], a["api_id"] = (
        fa[:, 0],
        fa[:, 1],
        fa[:, 2],
        fa[:, 3],
    )
    a["tid"] = recs["tid"][sel]

    tr = Trace(meta=meta, names=names, ops=np.sort(o, order="begin"), api=a)
    dev = f[t == ring.REC_DEVICE]
    if len(dev):
        d = [int(x) for x in dev[0]]

        def lo(x: int) -> int:
            return x & 0xFFFFFFFF

        tr.device = {
            "name": names.get(d[0], ""),
            "arch": names.get(d[1], ""),
            "cu_count": lo(d[2]),
            "wave_front_size": d[2] >> 32,
            "threads_per_cu": lo(d[3]),
            "lds_per_cu": d[3] >> 32,
            "clock_khz": lo(d[4]),
            "memory_clock_khz": d[4] >> 32,
            "memory_bus_width": lo(d[5]),
            "device": d[5] >> 32,
        }

    # Ranges: replay push/pop per thread in ring order (ring order is program
    # order within one thread).
    stacks: dict[int, list[tuple[int, int, int]]] = defaultdict(
        list
    )  # tid -> [(start, nid, idx)]
    for r in recs[
        (t == ring.REC_PUSH)
        | (t == ring.REC_POP)
        | (t == ring.REC_MARK)
        | (t == ring.REC_FENCE)
    ]:
        typ, tid, ff = int(r["type"]), int(r["tid"]), r["f"]
        if typ == ring.REC_PUSH:
            st = stacks[tid]
            parent = st[-1][2] if st else -1
            msg = names.get(int(ff[1]), f"name#{int(ff[1])}")
            tr.ranges.append(Range(tid, int(ff[0]), -1, msg, len(st), parent))
            st.append((int(ff[0]), int(ff[1]), len(tr.ranges) - 1))
            tr.roctx.append((int(ff[0]), tid, "RANGE_PUSH", msg))
        elif typ == ring.REC_POP:
            tr.roctx.append((int(ff[0]), tid, "RANGE_POP", ""))
            st = stacks[tid]
            if not st:
                tr.unbalanced_pops += 1
                continue
            _, _, idx = st.pop()
            tr.ranges[idx].end = int(ff[0])
        elif typ == ring.REC_MARK:
            msg = names.get(int(ff[1]), "")
            tr.marks.append((int(ff[0]), tid, msg))
            tr.roctx.append((int(ff[0]), tid, "MARK", msg))
        else:
            tr.fences.append(
                {
                    "ts": int(ff[0]),
                    "tid": tid,
                    "op": names.get(int(ff[1]), ""),
                    "run": int(ff[2]),
                    "instance": int(ff[3]),
                    "marker": bool(ff[4]),
                }
            )
    last = int(
        max(tr.ops["end"].max() if len(tr.ops) else 0, a["exit"].max() if len(a) else 0)
    )
    for rg in tr.ranges:
        if rg.end < 0:
            rg.end = max(last, rg.start)
    return tr


def load(out_dir: Path) -> Trace:
    recs, meta = load_journal(out_dir)
    return build(recs, meta)


def attribute(tr: Trace) -> None:
    """Charge every GPU command to the innermost range open at its HIP call."""
    n = len(tr.ops)
    tr.op_range = np.full(n, -1, dtype=np.int64)
    tr.op_tid = np.zeros(n, dtype=np.uint32)
    tr.op_enter = np.zeros(n, dtype=np.uint64)
    if not len(tr.api) or not n:
        return
    api = np.sort(tr.api, order="corr")
    idx = np.searchsorted(api["corr"], tr.ops["corr"])
    idx = np.clip(idx, 0, len(api) - 1)
    hit = api["corr"][idx] == tr.ops["corr"]
    tr.op_tid[hit] = api["tid"][idx[hit]]
    tr.op_enter[hit] = api["enter"][idx[hit]]

    by_tid: dict[int, list[int]] = defaultdict(list)
    for i, rg in enumerate(tr.ranges):
        by_tid[rg.tid].append(i)
    for tid, idxs in by_tid.items():
        # Sweep the thread's calls in time order against a stack of open
        # ranges; ranges nest per thread, so the stack top is the innermost.
        idxs.sort(key=lambda i: (tr.ranges[i].start, -tr.ranges[i].end))
        ops_here = np.nonzero(hit & (tr.op_tid == tid))[0]
        ops_here = ops_here[np.argsort(tr.op_enter[ops_here], kind="stable")]
        stack: list[int] = []
        k = 0
        for oi in ops_here:
            ts = int(tr.op_enter[oi])
            while k < len(idxs) and tr.ranges[idxs[k]].start <= ts:
                stack.append(idxs[k])
                k += 1
            while stack and tr.ranges[stack[-1]].end < ts:
                stack.pop()
            # A closed range can sit under a still-open sibling only if the
            # push/pop stream was unbalanced; skip those too.
            j = len(stack) - 1
            while j >= 0 and tr.ranges[stack[j]].end < ts:
                j -= 1
            if j >= 0:
                tr.op_range[oi] = stack[j]


def validate(tr: Trace) -> dict:
    """Clock and completeness checks reported with every trace."""
    out = {
        "ops": len(tr.ops),
        "api_calls": len(tr.api),
        "ranges": len(tr.ranges),
        "dropped": int(tr.meta.get("dropped", 0)),
        "unbalanced_pops": tr.unbalanced_pops,
        "uncommitted": int(tr.meta.get("uncommitted", 0)),
    }
    if tr.op_enter is not None and len(tr.ops):
        hit = tr.op_enter > 0
        lead = tr.ops["begin"][hit].astype(np.int64) - tr.op_enter[hit].astype(np.int64)
        out["ops_with_api"] = int(hit.sum())
        if hit.any():
            out["begin_before_call"] = int((lead < 0).sum())
            out["call_to_begin_ns_p50"] = int(np.median(lead))
            out["call_to_begin_ns_min"] = int(lead.min())
        out["ops_attributed"] = int((tr.op_range >= 0).sum())
    bad = tr.ops["end"] < tr.ops["begin"]
    out["negative_durations"] = int(bad.sum())
    return out


def op_family(label: str) -> str:
    """'hip_ep.op.matmul_nbits m=1,...' -> 'matmul_nbits'."""
    if label.startswith("hip_ep.op."):
        return label[10:].split(" ", 1)[0]
    if label.startswith("hip_ep.graph."):
        return label.split(" ", 1)[0]
    return label


def _innermost_op_range(tr: Trace, ri: int) -> int:
    """Nearest enclosing range that is a hip_ep.op range (skipping marks/graphs)."""
    while ri >= 0 and not tr.ranges[ri].name.startswith("hip_ep.op."):
        ri = tr.ranges[ri].parent
    return ri


def _graph_range(tr: Trace, ri: int) -> int:
    while ri >= 0 and not tr.ranges[ri].name.startswith("hip_ep.graph."):
        ri = tr.ranges[ri].parent
    return ri


def summarize(tr: Trace) -> dict:
    if tr.op_range is None:
        attribute(tr)
    dur = (tr.ops["end"].astype(np.int64) - tr.ops["begin"].astype(np.int64)).clip(
        min=0
    )
    kernels: dict[str, list[int]] = defaultdict(list)
    for i in np.nonzero(tr.ops["op"] == OP_DISPATCH)[0]:
        kernels[tr.name(tr.ops["aux"][i])].append(int(dur[i]))
    ops_by_family: dict[str, dict] = defaultdict(
        lambda: {"gpu_ns": 0, "kernels": 0, "calls": set()}
    )
    ops_by_label: dict[str, dict] = defaultdict(
        lambda: {"gpu_ns": 0, "kernels": 0, "calls": set()}
    )
    graphs: dict[str, dict] = defaultdict(lambda: {"gpu_ns": 0, "runs": set()})
    unattributed = 0
    setup = 0  # issued before the first range: session creation (weight upload)
    first_range = min((r.start for r in tr.ranges), default=0)
    for i in range(len(tr.ops)):
        ri = int(tr.op_range[i])
        oi = _innermost_op_range(tr, ri)
        gi = _graph_range(tr, ri)
        if oi < 0 and tr.ranges and 0 < int(tr.op_enter[i]) < first_range:
            setup += int(dur[i])
        elif oi < 0:
            unattributed += int(dur[i])
        else:
            lab = tr.ranges[oi].name
            for d, key in ((ops_by_family, op_family(lab)), (ops_by_label, lab)):
                d[key]["gpu_ns"] += int(dur[i])
                d[key]["kernels"] += 1
                d[key]["calls"].add(oi)
        if gi >= 0:
            g = graphs[op_family(tr.ranges[gi].name)]
            g["gpu_ns"] += int(dur[i])
            g["runs"].add(gi)
    host_by_family: dict[str, int] = defaultdict(int)
    calls_by_family: dict[str, int] = defaultdict(int)
    for rg in tr.ranges:
        if rg.name.startswith("hip_ep.op."):
            host_by_family[op_family(rg.name)] += rg.end - rg.start
            calls_by_family[op_family(rg.name)] += 1
    span = (int(tr.ops["end"].max()) - int(tr.ops["begin"].min())) if len(tr.ops) else 0
    return {
        "validation": validate(tr),
        "gpu_busy_ns": int(dur.sum()),
        "gpu_span_ns": span,
        "unattributed_gpu_ns": unattributed,
        "setup_gpu_ns": setup,
        "kernels": {
            k: {
                "calls": len(v),
                "total_ns": sum(v),
                "avg_ns": sum(v) / len(v),
                "min_ns": min(v),
                "max_ns": max(v),
            }
            for k, v in kernels.items()
        },
        "families": {
            k: {
                "gpu_ns": v["gpu_ns"],
                "kernels": v["kernels"],
                "calls": calls_by_family.get(k, len(v["calls"])),
                "host_ns": host_by_family.get(k, 0),
            }
            for k, v in ops_by_family.items()
        },
        "labels": {
            k: {
                "gpu_ns": v["gpu_ns"],
                "kernels": v["kernels"],
                "calls": len(v["calls"]),
            }
            for k, v in ops_by_label.items()
        },
        "graphs": {
            k: {"gpu_ns": v["gpu_ns"], "runs": len(v["runs"])}
            for k, v in graphs.items()
        },
        "fences": tr.fences,
    }


def write_report(tr: Trace, out_dir: Path, top: int = 30) -> Path:
    s = summarize(tr)
    out_dir = Path(out_dir)
    v = s["validation"]
    lines = ["hip-ep trace report", "=" * 19, ""]
    h = tr.meta
    lines.append(
        f"tracer={h.get('tracer')} api_domain={h.get('api')} ({h.get('api_detail', '')})"
    )
    lines.append("validation: " + ", ".join(f"{k}={v[k]}" for k in v))
    if v.get("begin_before_call"):
        lines.append(
            "WARNING: some GPU commands start before the HIP call that issued them; "
            "host and GPU clocks disagree, so attribution is unreliable."
        )
    if v["dropped"]:
        lines.append(
            f"WARNING: {v['dropped']} records were dropped (ring full); totals are low."
        )
    busy, span = s["gpu_busy_ns"], s["gpu_span_ns"]
    lines.append(
        f"GPU busy {busy / 1e6:.3f} ms over a {span / 1e6:.3f} ms span "
        f"({100 * busy / span if span else 0:.1f}%); "
        f"session setup (before the first range) {s['setup_gpu_ns'] / 1e6:.3f} ms; "
        f"unattributed during the run {s['unattributed_gpu_ns'] / 1e6:.3f} ms"
    )
    lines += [
        "",
        "Op families (by GPU time)",
        f"{'family':<34}{'calls':>8}{'kernels':>9}{'gpu ms':>11}{'gpu %':>8}{'host ms':>10}",
    ]
    for k, d in sorted(s["families"].items(), key=lambda x: -x[1]["gpu_ns"])[:top]:
        pct = 100 * d["gpu_ns"] / busy if busy else 0
        lines.append(
            f"{k:<34}{d['calls']:>8}{d['kernels']:>9}{d['gpu_ns'] / 1e6:>11.3f}"
            f"{pct:>7.1f}%{d['host_ns'] / 1e6:>10.3f}"
        )
    lines += ["", "Graphs (subgraph runs)", f"{'graph':<44}{'runs':>8}{'gpu ms':>11}"]
    for k, d in sorted(s["graphs"].items(), key=lambda x: -x[1]["gpu_ns"]):
        lines.append(f"{k:<44}{d['runs']:>8}{d['gpu_ns'] / 1e6:>11.3f}")
    lines += [
        "",
        "Kernels (by GPU time)",
        f"{'calls':>8}{'total ms':>11}{'avg us':>10}  kernel",
    ]
    for k, d in sorted(s["kernels"].items(), key=lambda x: -x[1]["total_ns"])[:top]:
        lines.append(
            f"{d['calls']:>8}{d['total_ns'] / 1e6:>11.3f}{d['avg_ns'] / 1e3:>10.2f}  {k[:120]}"
        )
    if s["fences"]:
        lines += ["", "Fences"] + [json.dumps(fz) for fz in s["fences"]]
    path = out_dir / "trace_report.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "trace_summary.json").write_text(json.dumps(s, indent=1, default=list))
    with (out_dir / "trace_families.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["family", "calls", "kernels", "gpu_ms", "host_ms"])
        for k, d in sorted(s["families"].items(), key=lambda x: -x[1]["gpu_ns"]):
            w.writerow(
                [
                    k,
                    d["calls"],
                    d["kernels"],
                    f"{d['gpu_ns'] / 1e6:.4f}",
                    f"{d['host_ns'] / 1e6:.4f}",
                ]
            )
    with (out_dir / "trace_kernels.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["kernel", "calls", "total_ms", "avg_us", "min_us", "max_us"])
        for k, d in sorted(s["kernels"].items(), key=lambda x: -x[1]["total_ns"]):
            w.writerow(
                [
                    k,
                    d["calls"],
                    f"{d['total_ns'] / 1e6:.4f}",
                    f"{d['avg_ns'] / 1e3:.3f}",
                    f"{d['min_ns'] / 1e3:.3f}",
                    f"{d['max_ns'] / 1e3:.3f}",
                ]
            )
    write_dispatches(tr, out_dir / "trace_dispatches.csv")
    return path


def write_dispatches(tr: Trace, path: Path) -> None:
    if tr.op_range is None:
        attribute(tr)
    t0 = int(tr.ops["begin"].min()) if len(tr.ops) else 0
    with Path(path).open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "index",
                "op",
                "queue",
                "corr",
                "start_us",
                "dur_us",
                "kernel_or_bytes",
                "tid",
                "op_range",
                "graph",
            ]
        )
        for i, o in enumerate(tr.ops):
            ri = int(tr.op_range[i])
            oi, gi = _innermost_op_range(tr, ri), _graph_range(tr, ri)
            what = tr.name(o["aux"]) if o["op"] == OP_DISPATCH else int(o["aux"])
            w.writerow(
                [
                    i,
                    OP_NAMES.get(int(o["op"]), int(o["op"])),
                    int(o["queue"]),
                    int(o["corr"]),
                    f"{(int(o['begin']) - t0) / 1e3:.3f}",
                    f"{(int(o['end']) - int(o['begin'])) / 1e3:.3f}",
                    what,
                    int(tr.op_tid[i]),
                    tr.ranges[oi].name if oi >= 0 else "",
                    op_family(tr.ranges[gi].name) if gi >= 0 else "",
                ]
            )


# ---- rocprofv3 export ----------------------------------------------------------


def write_rocprof(tr: Trace, out_dir: Path, prefix: str = "trace") -> list[Path]:
    """rocprofv3 file set (see rocprof.py)."""
    from . import rocprof

    return rocprof.write(tr, out_dir, prefix)


# ---- timelines ----------------------------------------------------------------


def _t0(tr: Trace) -> int:
    cands = []
    if len(tr.ops):
        cands.append(int(tr.ops["begin"].min()))
    if len(tr.api):
        cands.append(int(tr.api["enter"].min()))
    if tr.ranges:
        cands.append(min(r.start for r in tr.ranges))
    return min(cands) if cands else 0


def write_chrome(tr: Trace, path: Path, api: bool = False) -> Path:
    """Chrome/Perfetto JSON: GPU queues, ROCTx ranges per thread, optional HIP API."""
    t0 = _t0(tr)
    ev = []
    pid_gpu, pid_host = 1, 2
    ev.append(
        {"ph": "M", "pid": pid_gpu, "name": "process_name", "args": {"name": "GPU"}}
    )
    ev.append(
        {"ph": "M", "pid": pid_host, "name": "process_name", "args": {"name": "Host"}}
    )
    for o in tr.ops:
        name = (
            tr.name(o["aux"])
            if o["op"] == OP_DISPATCH
            else OP_NAMES.get(int(o["op"]), "op")
        )
        ev.append(
            {
                "ph": "X",
                "pid": pid_gpu,
                "tid": int(o["queue"]) & 0xFFFFFFFF,
                "name": name,
                "ts": (int(o["begin"]) - t0) / 1e3,
                "dur": max(0, int(o["end"]) - int(o["begin"])) / 1e3,
                "args": {"corr": int(o["corr"])},
            }
        )
    for rg in tr.ranges:
        ev.append(
            {
                "ph": "X",
                "pid": pid_host,
                "tid": rg.tid,
                "name": rg.name,
                "ts": (rg.start - t0) / 1e3,
                "dur": max(0, rg.end - rg.start) / 1e3,
            }
        )
    for ts, tid, name in tr.marks:
        ev.append(
            {
                "ph": "i",
                "s": "t",
                "pid": pid_host,
                "tid": tid,
                "name": name,
                "ts": (ts - t0) / 1e3,
            }
        )
    if api:
        for a in tr.api:
            ev.append(
                {
                    "ph": "X",
                    "pid": pid_host,
                    "tid": int(a["tid"]),
                    "name": tr.api_name(a["api_id"]),
                    "ts": (int(a["enter"]) - t0) / 1e3,
                    "dur": max(0, int(a["exit"]) - int(a["enter"])) / 1e3,
                    "args": {"corr": int(a["corr"])},
                }
            )
    path = Path(path)
    path.write_text(json.dumps({"traceEvents": ev, "displayTimeUnit": "ns"}))
    return path


def _varint(v: int) -> bytes:
    out = bytearray()
    while True:
        b = v & 0x7F
        v >>= 7
        if v:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _field(num: int, wire: int, payload: bytes | int) -> bytes:
    key = _varint((num << 3) | wire)
    if wire == 0:
        return key + _varint(int(payload))
    return key + _varint(len(payload)) + payload


def write_pftrace(tr: Trace, path: Path) -> Path:
    """Native Perfetto TrackEvent protobuf (Trace{packet}); no dependencies.

    Field numbers are from perfetto/protos: TracePacket.timestamp=8,
    track_event=11, track_descriptor=60, trusted_packet_sequence_id=10;
    TrackEvent.type=9, track_uuid=11, name=23; TrackDescriptor.uuid=1, name=2,
    parent_uuid=5.
    """
    seq = _field(10, 0, 1)
    out = bytearray()

    def packet(body: bytes) -> None:
        out.extend(_field(1, 2, body + seq))

    def track(uuid: int, name: str, parent: int = 0) -> None:
        desc = _field(1, 0, uuid) + _field(2, 2, name.encode())
        if parent:
            desc += _field(5, 0, parent)
        packet(_field(60, 2, desc))

    gpu_root, host_root = 1, 2
    track(gpu_root, "GPU")
    track(host_root, "Host")
    tracks: dict[tuple, int] = {}
    nxt = [100]

    def uuid_for(key: tuple, label: str, parent: int) -> int:
        if key not in tracks:
            nxt[0] += 1
            tracks[key] = nxt[0]
            track(nxt[0], label, parent)
        return tracks[key]

    # GPU commands on one queue can overlap (copies beside kernels); slices on
    # one track must nest, so overlapping commands go to extra lanes.
    lanes: dict[int, list[int]] = defaultdict(list)
    slices = []
    for o in tr.ops:
        b, e, q = int(o["begin"]), max(int(o["end"]), int(o["begin"])), int(o["queue"])
        ends = lanes[q]
        lane = next((i for i, last in enumerate(ends) if last <= b), len(ends))
        if lane == len(ends):
            ends.append(e)
        else:
            ends[lane] = e
        label = f"queue {q:#x}" + (f" lane {lane}" if lane else "")
        u = uuid_for(("q", q, lane), label, gpu_root)
        name = (
            tr.name(o["aux"])
            if o["op"] == OP_DISPATCH
            else OP_NAMES.get(int(o["op"]), "op")
        )
        slices.append((u, b, e, name))
    for rg in tr.ranges:
        u = uuid_for(("t", rg.tid), f"thread {rg.tid}", host_root)
        slices.append((u, rg.start, max(rg.end, rg.start), rg.name))

    # Begin/end events in timestamp order; at equal times ends come first, outer
    # begins before inner ones, and inner ends before outer ones.
    evs = []
    for u, b, e, name in slices:
        evs.append((b, 1, -(e - b), u, name))
        evs.append((e, 0, -b, u, None))
    evs.sort(key=lambda x: (x[0], x[1], x[2]))
    for ts, kind, _, u, name in evs:
        if kind == 1:
            body = _field(9, 0, 1) + _field(11, 0, u) + _field(23, 2, name.encode())
        else:
            body = _field(9, 0, 2) + _field(11, 0, u)
        packet(_field(8, 0, ts) + _field(11, 2, body))
    path = Path(path)
    path.write_bytes(bytes(out))
    return path
