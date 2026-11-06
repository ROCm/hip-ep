#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Convert a ktrace or rocprofv3-shaped capture to Chrome Trace Event JSON.

    python python\\ktrace_cli.py chrome out\\probe --start-ms 0 --span-ms 500
    python python\\ktrace_cli.py chrome out\\probe\\probe_kernel_trace.csv --all

Open the result at https://ui.perfetto.dev (or chrome://tracing). Doing it this
way rather than drawing our own view means zoom, search, flow arrows, slice
aggregation and counter tracks all come for free, and the trace can sit next to
captures from other tools.

Layout: one process per side of the boundary -- GPU commands on hardware queue
tracks, host calls on OS thread tracks. Correlation ids become flow arrows, so a
kernel can be traced back to the launch that issued it. Device memory becomes a
counter track.

Size is the reason for the windowing options. The format is verbose, roughly
150 bytes an event, so large captures can exceed a gigabyte when written
whole. `--all` therefore keeps every GPU command but only the host calls that
block, which is what explains the gaps anyway.
"""

from __future__ import annotations

import argparse
import datetime
import heapq
import json
import os
import platform
import sys
from array import array

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kernel_names import resolve_names  # noqa: E402
from analyze import (
    DEFAULT_PHASE_WORDS,
    PHASE_GAP_S,  # noqa: E402
    phase_of,
    segment_phases,
)
from rocprofv3_reader import ensure_capture  # noqa: E402

GPU_PID, HOST_PID, ROCTX_PID, PHASE_PID, CPU_PID, CPU_SAMPLE_PID = 0, 1, 2, 3, 4, 5
BLOCKING_APIS = (
    "hipStreamSynchronize",
    "hipEventSynchronize",
    "hipDeviceSynchronize",
    "hipStreamWaitEvent",
    "hipMemcpy",
    "hipMalloc",
    "hipFree",
    "hipModuleLoad",
)


def prescan(source):
    """One pass for everything needed before events can be written: the time
    origin, kernel names, phase segments, and compact command timings.

    Phases have to come from the whole capture even when only a window is
    exported, or a window landing mid-phase would show no phase at all.
    """
    lo = None
    raw_names: set[str] = set()
    marks: list[tuple[float, str]] = []
    queues = array("q")
    begins = array("d")
    ends = array("d")
    for r in ensure_capture(source).iter_ops():
        b = float(r["Begin_us"])
        queues.append(int(r["Queue"]))
        begins.append(b)
        ends.append(float(r["End_us"]))
        if lo is None or b < lo:
            lo = b
        if r["Op"] == "dispatch" and r["Kernel_Name"]:
            raw_names.add(r["Kernel_Name"])
            ph = phase_of(r["Kernel_Name"], DEFAULT_PHASE_WORDS)
            if ph:
                marks.append((b, ph))
    timings = queues, begins, ends
    return lo or 0.0, raw_names, segment_phases(marks, PHASE_GAP_S), timings


def plan_lanes(timings, t_from: float, t_to: float):
    """Assign overlap lanes after sorting each queue by timestamp.

    rocprofv3 domain CSV files are not guaranteed to be time ordered. Assigning
    lanes in file order turns harmless timestamp regressions into dozens of
    false tracks, so keep compact timings from the prescan and color intervals
    in queue/start order. The returned array is indexed by original row order.
    """
    queues, begins, ends = timings
    unassigned = (1 << 32) - 1
    lanes = array("I", [unassigned]) * len(queues)
    order = [i for i in range(len(queues)) if ends[i] >= t_from and begins[i] <= t_to]
    order.sort(key=lambda i: (queues[i], begins[i], ends[i], i))

    counts: dict[int, int] = {}
    current_queue = None
    active: list[tuple[float, int]] = []
    available: list[int] = []
    next_lane = 0
    for index in order:
        queue = queues[index]
        if current_queue != queue:
            current_queue = queue
            active.clear()
            available.clear()
            next_lane = 0
        while active and active[0][0] <= begins[index]:
            _, lane = heapq.heappop(active)
            heapq.heappush(available, lane)
        if available:
            lane = heapq.heappop(available)
        else:
            lane = next_lane
            next_lane += 1
        lanes[index] = lane
        counts[queue] = max(counts.get(queue, 0), lane + 1)
        heapq.heappush(active, (max(begins[index], ends[index]), lane))
    return lanes, counts


def earliest_host_enter(source) -> tuple[float | None, set[int]]:
    """Earliest host-call entry, which can precede the first GPU command.

    The trace origin has to cover every track: setup calls like hipMalloc and
    hipModuleLoad run before any kernel does, and anchoring on the first GPU
    command would give them negative timestamps, which trace processor drops.

    The domain file is still streamed; decoded argument sidecars are skipped
    because only timestamps are needed here.
    """
    lo = None
    threads: set[int] = set()
    for row in ensure_capture(source).iter_api(include_args=False):
        value = float(row["Enter_us"])
        threads.add(int(row["Thread"]))
        if lo is None or value < lo:
            lo = value
    return lo, threads


def device_line(source) -> str:
    """The GPU name the tracer recorded, so the trace says what it came from."""
    return ensure_capture(source).device_name()


def kernel_info(source) -> dict[str, dict]:
    return {row["Kernel_Name"]: row for row in ensure_capture(source).iter_kernels()}


def emit(out, ev: dict, state: dict) -> None:
    """Append one event, keeping the JSON streamable rather than building a list."""
    # Trace processor drops negative timestamps outright. The origin is chosen
    # to make that impossible, so a negative here means the origin was computed
    # from an incomplete view of the inputs; clamp and count rather than lose
    # the event silently.
    if ev.get("ts", 0) < 0:
        state["clamped"] = state.get("clamped", 0) + 1
        ev["ts"] = 0.0
    if state.get("event_sink") is not None:
        state["event_sink"].event(ev)
    if not state.get("suppress_json"):
        out.write(("," if state["n"] else "") + json.dumps(ev, ensure_ascii=False))
    state["n"] += 1


def main(argv=None, event_sink=None, suppress_json: bool = False) -> int:
    ap = argparse.ArgumentParser(
        description="ktrace capture -> Chrome Trace Event JSON"
    )
    ap.add_argument(
        "capture", help="rocprofv3 prefix/domain/results file, or legacy *_ops.csv"
    )
    ap.add_argument(
        "out_json", nargs="?", help="output JSON (default: <prefix>_ktrace_chrome.json)"
    )
    ap.add_argument(
        "--start-ms",
        type=float,
        default=None,
        help="window start, ms from the first GPU command",
    )
    ap.add_argument("--span-ms", type=float, default=500.0, help="window length in ms")
    ap.add_argument(
        "--all",
        action="store_true",
        help="whole capture; host side is limited to blocking calls",
    )
    ap.add_argument(
        "--phase",
        default=None,
        metavar="NAME[:N]",
        help="window onto an inferred phase, e.g. decode or decode:0 "
        "(default: the last run of that phase). Beats a fixed "
        "--start-ms, whose meaning shifts between runs.",
    )
    flow_group = ap.add_mutually_exclusive_group()
    flow_group.add_argument(
        "--flows",
        action="store_true",
        help="with --all, keep every host API and correlation arrow",
    )
    flow_group.add_argument(
        "--no-flows",
        action="store_true",
        help="omit correlation arrows (they cost roughly a third of the size)",
    )
    args = ap.parse_args(argv)

    # A whole-capture export keeps only the blocking host calls, so most flows
    # would have a GPU end and no host end to start from. Dangling arrows are
    # worse than none, and they are a third of the file.
    if args.all and not args.flows:
        args.no_flows = True

    capture = ensure_capture(args.capture)
    if args.out_json is None:
        args.out_json = capture.auto_output("_ktrace_chrome.json")

    gpu_lo, raw_names, phases, timings = prescan(capture)
    host_lo, target_threads = earliest_host_enter(capture)
    roctx_events = sorted(capture.iter_roctx(), key=lambda row: row["Timestamp_us"])
    target_threads.update(int(row["Thread"]) for row in roctx_events)
    resume_origin = capture.resume_timestamp_us()
    if resume_origin is not None:
        origin = resume_origin
    else:
        origins = [gpu_lo]
        if host_lo is not None:
            origins.append(host_lo)
        # Preserve the historical GPU/HIP origin. ROCTx events are projected
        # onto that timeline and must not shift previously displayed slices.
        origin = min(origins)

    if args.all:
        t_from, t_to = float("-inf"), float("inf")
    elif args.phase:
        name, _, idx = args.phase.partition(":")
        runs = [p for p in phases if p[0] == name]
        if not runs:
            found = sorted({p[0] for p in phases}) or ["none"]
            sys.exit(f"no phase named {name!r}; this capture has: {', '.join(found)}")
        seg = runs[int(idx)] if idx else runs[-1]
        t_from = seg[1]
        t_to = min(seg[2], t_from + args.span_ms * 1000.0)
    else:
        start = args.start_ms if args.start_ms is not None else 0.0
        t_from = origin + start * 1000.0
        t_to = t_from + args.span_ms * 1000.0

    kinfo = kernel_info(capture)
    # Names are resolved up front so the slice titles are readable; the mangled
    # form stays in args for searching.
    labels = resolve_names(list(raw_names))
    lane_by_op, lane_counts = plan_lanes(timings, t_from, t_to)
    have_cpu_sched = capture.domain_path("cpu_sched").exists()
    have_cpu_samples = capture.domain_path("cpu_samples").exists()

    state = {
        "n": 0,
        "event_sink": event_sink,
        "suppress_json": suppress_json,
    }
    queues_seen: set[int] = set()
    threads_seen: set[int] = set()
    roctx_threads_seen: set[int] = set()
    corr_gpu: dict[int, bool] = {}

    # `otherData` is the Chrome format's slot for free-form provenance, shown in
    # the metadata panel of chrome://tracing and Perfetto. Worth filling in: the
    # file gets copied around, and on its own a wall of slices says nothing
    # about which machine, which capture, or which tool produced it.
    other = {
        "producer": "ktrace (hipRegisterTracerCallback, native runtime timestamps)",
        "format": "Chrome Trace Event JSON",
        "source": os.path.abspath(capture.source_path),
        "captured": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "scope": "full capture"
        if args.all
        else f"{(t_from - origin) / 1000:.1f}..{(t_to - origin) / 1000:.1f} ms",
        "host": platform.node(),
        "time_origin": (
            "ResumeThread QPC" if resume_origin is not None else "first captured event"
        ),
    }
    dev = device_line(capture)
    if dev:
        other["device"] = dev
    if phases:
        other["phases"] = ", ".join(
            f"{t} x{sum(1 for p in phases if p[0] == t)}"
            for t in dict.fromkeys(p[0] for p in phases)
        )
    if event_sink is not None:
        event_sink.metadata(other)

    with open(args.out_json, "w", encoding="utf-8") as out:
        out.write(
            '{"displayTimeUnit":"ms","otherData":'
            + json.dumps(other, ensure_ascii=False)
            + ',"traceEvents":['
        )

        emit(
            out,
            {
                "ph": "M",
                "pid": GPU_PID,
                "name": "process_name",
                "args": {"name": "GPU (hardware queues)"},
            },
            state,
        )
        emit(
            out,
            {
                "ph": "M",
                "pid": GPU_PID,
                "name": "process_sort_index",
                "args": {"sort_index": GPU_PID},
            },
            state,
        )
        emit(
            out,
            {
                "ph": "M",
                "pid": HOST_PID,
                "name": "process_name",
                "args": {"name": "Host (HIP API)"},
            },
            state,
        )
        emit(
            out,
            {
                "ph": "M",
                "pid": HOST_PID,
                "name": "process_sort_index",
                "args": {"sort_index": HOST_PID},
            },
            state,
        )
        if roctx_events:
            emit(
                out,
                {
                    "ph": "M",
                    "pid": ROCTX_PID,
                    "name": "process_name",
                    "args": {"name": "ROCTx annotations"},
                },
                state,
            )
            emit(
                out,
                {
                    "ph": "M",
                    "pid": ROCTX_PID,
                    "name": "process_sort_index",
                    "args": {"sort_index": ROCTX_PID},
                },
                state,
            )
        if have_cpu_sched:
            emit(
                out,
                {
                    "ph": "M",
                    "pid": CPU_PID,
                    "name": "process_name",
                    "args": {"name": "CPU scheduling"},
                },
                state,
            )
            emit(
                out,
                {
                    "ph": "M",
                    "pid": CPU_PID,
                    "name": "process_sort_index",
                    "args": {"sort_index": CPU_PID},
                },
                state,
            )
        if have_cpu_samples:
            emit(
                out,
                {
                    "ph": "M",
                    "pid": CPU_SAMPLE_PID,
                    "name": "process_name",
                    "args": {"name": "CPU samples"},
                },
                state,
            )
            emit(
                out,
                {
                    "ph": "M",
                    "pid": CPU_SAMPLE_PID,
                    "name": "process_sort_index",
                    "args": {"sort_index": CPU_SAMPLE_PID},
                },
                state,
            )

        roctx_n = 0
        nested: dict[int, list[dict]] = {}
        process_ranges: dict[int, dict] = {}

        def emit_roctx_range(begin, end, cross_thread):
            nonlocal roctx_n
            if end["Timestamp_us"] < t_from or begin["Timestamp_us"] > t_to:
                return
            lo_c = max(begin["Timestamp_us"], t_from)
            hi_c = min(end["Timestamp_us"], t_to)
            tid = int(begin["Thread"])
            if tid not in roctx_threads_seen:
                roctx_threads_seen.add(tid)
                emit(
                    out,
                    {
                        "ph": "M",
                        "pid": ROCTX_PID,
                        "tid": tid,
                        "name": "thread_name",
                        "args": {"name": f"ROCTx thread {tid}"},
                    },
                    state,
                )
            details = {
                "range_id": int(begin["Range_Id"]),
                "cross_thread": cross_thread,
            }
            if cross_thread:
                details["stop_thread"] = int(end["Thread"])
            emit(
                out,
                {
                    "ph": "X",
                    "pid": ROCTX_PID,
                    "tid": tid,
                    "name": begin["Message"] or "ROCTx range",
                    "cat": "roctx",
                    "ts": round(lo_c - origin, 3),
                    "dur": round(hi_c - lo_c, 3),
                    "args": details,
                },
                state,
            )
            roctx_n += 1

        for event in roctx_events:
            operation = event["Operation"]
            tid = int(event["Thread"])
            if operation == "MARK":
                if t_from <= event["Timestamp_us"] <= t_to:
                    if tid not in roctx_threads_seen:
                        roctx_threads_seen.add(tid)
                        emit(
                            out,
                            {
                                "ph": "M",
                                "pid": ROCTX_PID,
                                "tid": tid,
                                "name": "thread_name",
                                "args": {"name": f"ROCTx thread {tid}"},
                            },
                            state,
                        )
                    emit(
                        out,
                        {
                            "ph": "i",
                            "s": "t",
                            "pid": ROCTX_PID,
                            "tid": tid,
                            "name": event["Message"] or "ROCTx mark",
                            "cat": "roctx",
                            "ts": round(event["Timestamp_us"] - origin, 3),
                        },
                        state,
                    )
                    roctx_n += 1
            elif operation == "RANGE_PUSH":
                nested.setdefault(tid, []).append(event)
            elif operation == "RANGE_POP":
                stack = nested.setdefault(tid, [])
                if stack:
                    emit_roctx_range(stack.pop(), event, False)
            elif operation == "RANGE_START":
                process_ranges[int(event["Range_Id"])] = event
            elif operation == "RANGE_STOP":
                begin = process_ranges.pop(int(event["Range_Id"]), None)
                if begin is not None:
                    emit_roctx_range(begin, event, True)

        # Make a long window scan visibly alive: when the selected phase is near
        # the end of a multi-gigabyte capture, no matching event may be emitted
        # for minutes. Without this flush the output file remains at 0 bytes and
        # looks hung even though the reader is making progress.
        out.flush()
        print("[scan] GPU command domains ...", flush=True)

        # Phase bands, so the GPU and host tracks below can be read against
        # "this is prefill" rather than against a bare wall clock. Inferred from
        # kernel names by the same code the text report uses -- clipped to the
        # window rather than dropped, so a window inside a phase still shows it.
        phase_n = 0
        if phases:
            emit(
                out,
                {
                    "ph": "M",
                    "pid": PHASE_PID,
                    "name": "process_name",
                    "args": {"name": "Phases (inferred from kernel names)"},
                },
                state,
            )
            emit(
                out,
                {
                    "ph": "M",
                    "pid": PHASE_PID,
                    "name": "process_sort_index",
                    "args": {"sort_index": PHASE_PID},
                },
                state,
            )
            emit(
                out,
                {
                    "ph": "M",
                    "pid": PHASE_PID,
                    "tid": 0,
                    "name": "thread_name",
                    "args": {"name": "phase"},
                },
                state,
            )
            for i, (tag, a, b) in enumerate(phases):
                if b < t_from or a > t_to:
                    continue
                lo_c, hi_c = max(a, t_from), min(b, t_to)
                emit(
                    out,
                    {
                        "ph": "X",
                        "pid": PHASE_PID,
                        "tid": 0,
                        "name": tag,
                        "cat": "phase",
                        "ts": round(lo_c - origin, 3),
                        "dur": round(hi_c - lo_c, 3),
                        "args": {
                            "run": i,
                            "full_ms": round((b - a) / 1000, 1),
                            "clipped": (lo_c > a or hi_c < b),
                        },
                    },
                    state,
                )
                phase_n += 1

        # ---- GPU commands --------------------------------------------------
        gpu_n = 0
        for op_index, r in enumerate(capture.iter_ops()):
            begin, end = float(r["Begin_us"]), float(r["End_us"])
            if end < t_from or begin > t_to:
                continue
            op = r["Op"]
            queue = f"{r['Device']}/{r['Queue']}"
            qnum = int(r["Queue"])
            lane = lane_by_op[op_index]
            if lane == (1 << 32) - 1:
                raise RuntimeError("internal lane assignment mismatch")
            tid = qnum * 1000 + lane
            if tid not in queues_seen:
                queues_seen.add(tid)
                label = f"queue {queue}" if lane == 0 else f"queue {queue} ~{lane}"
                emit(
                    out,
                    {
                        "ph": "M",
                        "pid": GPU_PID,
                        "tid": tid,
                        "name": "thread_name",
                        "args": {"name": label},
                    },
                    state,
                )

            raw = r["Kernel_Name"]
            name = labels.get(raw, raw) if op == "dispatch" else r["Kind"]
            a = {"op": op, "kind": r["Kind"], "queue": queue}
            if op == "dispatch":
                a["mangled"] = raw
                k = kinfo.get(raw)
                if k:
                    a["occupancy_pct"] = k["Theoretical_Occupancy_Pct"]
                    a["VGPR"] = k["VGPR"]
                    a["scratch_bytes"] = k["Scratch_Bytes"]
                    a["static_lds"] = k["Static_LDS"]
            else:
                a["bytes"] = int(r["Bytes"])

            emit(
                out,
                {
                    "ph": "X",
                    "pid": GPU_PID,
                    "tid": tid,
                    "name": name or op,
                    "cat": op,
                    "ts": round(begin - origin, 3),
                    "dur": round(end - begin, 3),
                    "args": a,
                },
                state,
            )
            gpu_n += 1

            corr = int(r["Corr"])
            if corr and not args.no_flows:
                # Arrow lands on the GPU slice; the host end is emitted below.
                corr_gpu[corr] = True
                emit(
                    out,
                    {
                        "ph": "f",
                        "bp": "e",
                        "pid": GPU_PID,
                        "tid": tid,
                        "id": corr,
                        "cat": "launch",
                        "name": "issued by",
                        "ts": round(begin - origin, 3),
                    },
                    state,
                )

        # ---- host calls ----------------------------------------------------
        print("[scan] HIP API domain ...", flush=True)
        host_n = 0
        for r in capture.iter_api():
            enter, exit_ = float(r["Enter_us"]), float(r["Exit_us"])
            if exit_ < t_from or enter > t_to:
                continue
            api = r["Api"]
            # A whole-capture export cannot carry 8.7 M host slices, and
            # the ones worth keeping are those that block.
            if args.all and not args.flows and api not in BLOCKING_APIS:
                continue
            tid = int(r["Thread"])
            if tid not in threads_seen:
                threads_seen.add(tid)
                emit(
                    out,
                    {
                        "ph": "M",
                        "pid": HOST_PID,
                        "tid": tid,
                        "name": "thread_name",
                        "args": {"name": f"thread {tid}"},
                    },
                    state,
                )
            a = {"api": api}
            if r.get("Args"):
                for kv in r["Args"].split():
                    k, _, v = kv.partition("=")
                    a[k] = v
            emit(
                out,
                {
                    "ph": "X",
                    "pid": HOST_PID,
                    "tid": tid,
                    "name": api,
                    "cat": "hip_api",
                    "ts": round(enter - origin, 3),
                    "dur": round(exit_ - enter, 3),
                    "args": a,
                },
                state,
            )
            host_n += 1

            corr = int(r["Corr"])
            if corr and not args.no_flows and corr in corr_gpu:
                emit(
                    out,
                    {
                        "ph": "s",
                        "bp": "e",
                        "pid": HOST_PID,
                        "tid": tid,
                        "id": corr,
                        "cat": "launch",
                        "name": "issued by",
                        "ts": round(enter - origin, 3),
                    },
                    state,
                )

        cpu_sched_n = 0
        if have_cpu_sched:
            starts: dict[int, float] = {}
            cpu_tracks: set[int] = set()
            for row in capture.iter_cpu_sched():
                ts = float(row["Timestamp_us"])
                cpu = int(row["Cpu"])
                old_thread = int(row["Old_Thread"])
                if cpu in starts and old_thread in target_threads:
                    lo_c, hi_c = max(starts[cpu], t_from), min(ts, t_to)
                    if hi_c >= lo_c:
                        if cpu not in cpu_tracks:
                            cpu_tracks.add(cpu)
                            emit(
                                out,
                                {
                                    "ph": "M",
                                    "pid": CPU_PID,
                                    "tid": cpu,
                                    "name": "thread_name",
                                    "args": {"name": f"CPU {cpu}"},
                                },
                                state,
                            )
                        emit(
                            out,
                            {
                                "ph": "X",
                                "pid": CPU_PID,
                                "tid": cpu,
                                "name": f"thread {old_thread}",
                                "cat": "cpu_sched",
                                "ts": round(lo_c - origin, 3),
                                "dur": round(hi_c - lo_c, 3),
                                "args": {
                                    "thread_id": old_thread,
                                    "state": int(row["Old_State"]),
                                    "wait_reason": int(row["Wait_Reason"]),
                                },
                            },
                            state,
                        )
                        cpu_sched_n += 1
                starts[cpu] = ts

        cpu_sample_n = 0
        if have_cpu_samples:
            sample_tracks: set[int] = set()
            for row in capture.iter_cpu_samples():
                ts = float(row["Timestamp_us"])
                tid = int(row["Thread"])
                if tid not in target_threads or ts < t_from or ts > t_to:
                    continue
                if tid not in sample_tracks:
                    sample_tracks.add(tid)
                    emit(
                        out,
                        {
                            "ph": "M",
                            "pid": CPU_SAMPLE_PID,
                            "tid": tid,
                            "name": "thread_name",
                            "args": {"name": f"thread {tid}"},
                        },
                        state,
                    )
                emit(
                    out,
                    {
                        "ph": "i",
                        "s": "t",
                        "pid": CPU_SAMPLE_PID,
                        "tid": tid,
                        "name": "CPU sample",
                        "cat": "cpu_sample",
                        "ts": round(ts - origin, 3),
                        "args": {
                            "instruction_pointer": row["Instruction_Pointer"],
                            "stack": row["Stack"],
                            "cpu": int(row["Cpu"]),
                            "process_id": int(row["Process"]),
                        },
                    },
                    state,
                )
                cpu_sample_n += 1

        # ---- device memory counter -----------------------------------------
        vram_n = 0
        for r in capture.iter_vram():
            ts = float(r["Time_us"])
            if ts < t_from or ts > t_to:
                continue
            emit(
                out,
                {
                    "ph": "C",
                    "pid": GPU_PID,
                    "name": "device memory",
                    "ts": round(ts - origin, 3),
                    "args": {"used_MB": round(int(r["Used_Bytes"]) / 1048576, 1)},
                },
                state,
            )
            vram_n += 1

        out.write("]}")

    size = os.path.getsize(args.out_json)
    scope = (
        "full capture"
        if args.all
        else f"{(t_from - origin) / 1000:.1f}..{(t_to - origin) / 1000:.1f} ms"
    )
    print(f"scope       : {scope}")
    print(f"phases      : {phase_n} band(s) of {len(phases)} in the capture")
    print(
        f"GPU slices  : {gpu_n:,} on {len(queues_seen)} tracks "
        f"({len(lane_counts)} queues, "
        f"{max(lane_counts.values(), default=0)} lanes at most)"
    )
    print(
        f"host slices : {host_n:,} on {len(threads_seen)} threads"
        + ("  (blocking calls only)" if args.all and not args.flows else "")
    )
    print(f"ROCTx events: {roctx_n:,}")
    print(f"CPU sched   : {cpu_sched_n:,}")
    print(f"CPU samples : {cpu_sample_n:,}")
    print(f"vram points : {vram_n:,}")
    print(f"events      : {state['n']:,}")
    if state.get("clamped"):
        print(
            f"[warn] {state['clamped']} events had a timestamp before the trace origin "
            f"and were clamped to 0"
        )
    print(
        f"-> {args.out_json}  ({size / 1e6:.1f} MB)   open at https://ui.perfetto.dev"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
