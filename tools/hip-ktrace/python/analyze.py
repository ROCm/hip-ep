#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Report on a ktrace or rocprofv3-shaped capture.

    python python\\ktrace_cli.py analyze out\\probe
    python python\\ktrace_cli.py analyze out\\probe\\probe_kernel_trace.csv

Accepts a rocprofv3 output prefix, any of its domain files, its results JSON,
or a legacy ``*_ops.csv``. The point is not to dump the numbers -- the source
files already are the numbers -- but to put the facts next to each other that
only mean something together: a kernel's share of GPU time next to its
occupancy and register spill, host wait time next to the queue it waited on.

Every file is streamed. The API trace runs to hundreds of millions of rows on a
real workload and does not fit in memory.
"""

from __future__ import annotations

import argparse
import collections
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kernel_names import resolve_names  # noqa: E402
from rocprofv3_reader import ensure_capture  # noqa: E402

# A kernel below this is worth looking at, in the sense that the hardware could
# have held more waves had the kernel asked for fewer registers or less LDS.
LOW_OCCUPANCY_PCT = 50.0


def ktrace_version() -> str:
    path = Path(__file__).resolve().parent.parent / "VERSION"
    try:
        return path.read_text(encoding="utf-8").strip() or "unknown"
    except OSError:
        return "unknown"


# Phases are inferred from kernel names rather than from markers, because there
# is no roctx on Windows and the application would have to be modified to emit
# any. It works here because the kernels are named after the phase they serve
# (mha_flash_prefill_kernel, qmoe_decode_fc1_kernel). Where names carry no such
# hint this finds nothing and the section is skipped -- it never guesses.
DEFAULT_PHASE_WORDS = ("prefill", "decode")
# A phase ends when a different phase's kernels appear, or when its own stop for
# longer than this. Iterations here are seconds apart, individual kernels
# microseconds, so anything in between separates cleanly.
PHASE_GAP_S = 0.5


def phase_of(name: str, words: tuple[str, ...]) -> str | None:
    for w in words:
        if w in name:
            return w
    return None


def segment_phases(
    marks: list[tuple[float, str]], gap_s: float
) -> list[tuple[str, float, float]]:
    """Collapse per-dispatch phase tags into contiguous runs."""
    marks.sort()
    out: list[tuple[str, float, float]] = []
    cur = start = last = None
    for t, tag in marks:
        if cur != tag or (last is not None and t - last > gap_s * 1e6):
            if cur is not None:
                out.append((cur, start, last))
            cur, start = tag, t
        last = t
    if cur is not None:
        out.append((cur, start, last))
    out = [s for s in out if s[2] - s[1] > 0.05 * 1e6]
    expanded = [list(s) for s in out]
    max_bridge = gap_s * 1e6
    for i in range(len(expanded) - 1):
        gap = expanded[i + 1][1] - expanded[i][2]
        if 0 <= gap <= max_bridge:
            boundary = (expanded[i][2] + expanded[i + 1][1]) / 2
            expanded[i][2] = boundary
            expanded[i + 1][1] = boundary
    return [tuple(s) for s in expanded]


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} TB"


def read_ops(source, phase_words: tuple[str, ...] = DEFAULT_PHASE_WORDS):
    """Single pass over the GPU commands.

    Returns per-kernel totals keyed by the raw mangled name, per-queue busy
    time, transfer totals, and the overall extent. Individual records are never
    retained.
    """
    kern_time: collections.Counter = collections.Counter()
    kern_count: collections.Counter = collections.Counter()
    kern_max: dict[str, float] = {}
    queues: dict[str, list] = collections.defaultdict(lambda: [0, 0.0])
    xfer: dict[str, list] = collections.defaultdict(lambda: [0, 0, 0.0])
    corr_queue: dict[int, str] = {}
    lo, hi, busy = None, None, 0.0
    counts: collections.Counter = collections.Counter()
    phase_marks: list[tuple[float, str]] = []
    phase_kern: dict[str, collections.Counter] = collections.defaultdict(
        collections.Counter
    )
    phase_busy: collections.Counter = collections.Counter()

    capture = ensure_capture(source)
    for r in capture.iter_ops():
        begin, end = float(r["Begin_us"]), float(r["End_us"])
        dur = end - begin
        op = r["Op"]
        counts[op] += 1
        busy += dur
        if lo is None or begin < lo:
            lo = begin
        if hi is None or end > hi:
            hi = end

        q = f"{r['Device']}/{r['Queue']}"
        queues[q][0] += 1
        queues[q][1] += dur

        if op == "dispatch":
            name = r["Kernel_Name"]
            kern_time[name] += dur
            kern_count[name] += 1
            if dur > kern_max.get(name, 0.0):
                kern_max[name] = dur
            # Only needed to attach a hardware queue to a HIP stream, and
            # one sighting per queue is enough for that.
            corr = int(r["Corr"])
            if corr and len(corr_queue) < 400000:
                corr_queue[corr] = q

            ph = phase_of(name, phase_words)
            if ph:
                phase_marks.append((begin, ph))
        elif op == "copy":
            x = xfer[r["Kind"]]
            x[0] += 1
            x[1] += int(r["Bytes"])
            x[2] += dur

    phases = segment_phases(phase_marks, PHASE_GAP_S)
    phase_index = 0
    for r in capture.iter_ops():
        begin, end = float(r["Begin_us"]), float(r["End_us"])
        midpoint = (begin + end) / 2
        while phase_index < len(phases) and midpoint > phases[phase_index][2]:
            phase_index += 1
        if phase_index >= len(phases):
            break
        tag, phase_begin, phase_end = phases[phase_index]
        if midpoint < phase_begin:
            continue
        dur = end - begin
        phase_busy[tag] += dur
        if r["Op"] == "dispatch":
            phase_kern[tag][r["Kernel_Name"]] += dur

    return {
        "kern_time": kern_time,
        "kern_count": kern_count,
        "kern_max": kern_max,
        "queues": queues,
        "xfer": xfer,
        "corr_queue": corr_queue,
        "phases": phases,
        "phase_kern": phase_kern,
        "phase_busy": phase_busy,
        "span": (hi - lo) if lo is not None else 0.0,
        "busy": busy,
        "counts": counts,
    }


def read_api(source, corr_queue: dict[int, str], phases: list = ()):
    """Single pass over the host calls.

    Alongside the per-API totals this resolves two things that need the GPU side
    to mean anything: which hardware queue each HIP stream feeds, and how much
    host wait time each of those queues is responsible for.
    """
    per_api: dict[str, list] = collections.defaultdict(lambda: [0, 0.0])
    stream_queue: dict[str, collections.Counter] = collections.defaultdict(
        collections.Counter
    )
    stream_wait: dict[str, list] = collections.defaultdict(lambda: [0, 0.0])
    allocs: dict[str, int] = {}
    alloc_events: list[tuple[float, int, str]] = []
    alloc_peak = alloc_cur = alloc_total = 0
    unmatched_free = 0
    total_calls = 0
    phase_wait: collections.Counter = collections.Counter()

    capture = ensure_capture(source)
    allocation_domain = not capture.is_legacy and capture.domain_path("alloc").exists()
    for r in capture.iter_api():
        api, host = r["Api"], float(r["Host_us"])
        total_calls += 1
        e = per_api[api]
        e[0] += 1
        e[1] += host

        args = r.get("Args") or ""
        if not args:
            continue

        if "stream=" in args:
            stream = args.split("stream=")[1].split()[0]
            if api == "hipStreamSynchronize":
                w = stream_wait[stream]
                w[0] += 1
                w[1] += host
                # Which phase was running when the host blocked. Segments
                # are disjoint, so the first hit is the only hit.
                enter = float(r["Enter_us"])
                for tag, a, b in phases:
                    if a <= enter <= b:
                        phase_wait[tag] += host
                        break
            else:
                corr = int(r["Corr"])
                q = corr_queue.get(corr)
                if q:
                    stream_queue[stream][q] += 1

        if (
            not allocation_domain
            and api in ("hipMalloc", "hipHostMalloc")
            and "size=" in args
        ):
            ptr = args.split("ptr=")[1].split()[0]
            size = int(args.split("size=")[1].split()[0])
            allocs[ptr] = size
            alloc_cur += size
            alloc_total += size
            alloc_peak = max(alloc_peak, alloc_cur)
            alloc_events.append((float(r["Enter_us"]), alloc_cur, api))
        elif (
            not allocation_domain
            and api in ("hipFree", "hipHostFree")
            and "ptr=" in args
        ):
            ptr = args.split("ptr=")[1].split()[0]
            if ptr in allocs:
                alloc_cur -= allocs.pop(ptr)
                alloc_events.append((float(r["Enter_us"]), alloc_cur, api))
            else:
                unmatched_free += 1

    if allocation_domain:
        for row in capture.iter_allocations():
            operation = row["Operation"]
            ptr = row["Address"]
            if "ALLOCATE" in operation and "FREE" not in operation:
                size = int(row["Allocation_Size"])
                allocs[ptr] = size
                alloc_cur += size
                alloc_total += size
                alloc_peak = max(alloc_peak, alloc_cur)
                alloc_events.append((float(row["Begin_us"]), alloc_cur, operation))
            elif "FREE" in operation:
                if ptr in allocs:
                    alloc_cur -= allocs.pop(ptr)
                    alloc_events.append((float(row["Begin_us"]), alloc_cur, operation))
                else:
                    unmatched_free += 1

    return {
        "per_api": per_api,
        "stream_queue": stream_queue,
        "stream_wait": stream_wait,
        "alloc_peak": alloc_peak,
        "alloc_total": alloc_total,
        "alloc_leaked": alloc_cur,
        "alloc_blocks": len(allocs),
        "unmatched_free": unmatched_free,
        "total_calls": total_calls,
        "phase_wait": phase_wait,
    }


def read_kernels(source) -> dict[str, dict]:
    return {r["Kernel_Name"]: r for r in ensure_capture(source).iter_kernels()}


def read_vram(source):
    used, t = [], []
    for r in ensure_capture(source).iter_vram():
        used.append(int(r["Used_Bytes"]))
        t.append(float(r["Time_us"]))
    if not used:
        return None
    return {
        "n": len(used),
        "min": min(used),
        "max": max(used),
        "first": used[0],
        "last": used[-1],
        "total": None,
        "span_s": (t[-1] - t[0]) / 1e6,
    }


class _Tee:
    """Write the report to the terminal and to disk at once.

    Saving it matters because a report costs ~25 s to produce: the API trace it
    streams runs to hundreds of megabytes, and the answer should not have to be
    recomputed to be read again, or to be compared against a later capture.
    """

    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)

    def flush(self):
        for st in self.streams:
            st.flush()


def section(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="report on a ktrace capture")
    ap.add_argument(
        "capture", help="rocprofv3 prefix/domain/results file, or legacy *_ops.csv"
    )
    ap.add_argument("--top", type=int, default=20, help="rows in the hotspot tables")
    ap.add_argument(
        "-o",
        "--out",
        default=None,
        help="report file (default: <prefix>_ktrace_report.txt); - for stdout only",
    )
    args = ap.parse_args(argv)

    capture = ensure_capture(args.capture)
    ops = read_ops(capture)
    kinfo = read_kernels(capture)
    api = (
        read_api(capture, ops["corr_queue"], ops["phases"])
        if capture.domain_path("api").exists()
        else None
    )
    vram = read_vram(capture)

    labels = resolve_names(list(ops["kern_time"]))
    span, busy = ops["span"], ops["busy"]

    report_path = (
        None
        if args.out == "-"
        else (args.out or capture.auto_output("_ktrace_report.txt"))
    )
    fh = open(report_path, "w", encoding="utf-8") if report_path else None
    saved_stdout = sys.stdout
    if fh:
        sys.stdout = _Tee(saved_stdout, fh)
    try:
        rc = write_report(
            ops,
            api,
            kinfo,
            vram,
            labels,
            span,
            busy,
            capture.device_name(),
            capture.runtime_info(),
            args.top,
        )
    finally:
        sys.stdout = saved_stdout
        if fh:
            fh.close()
    if report_path:
        print(f"-> {report_path}")
    return rc


def write_report(
    ops, api, kinfo, vram, labels, span, busy, device_name, runtime_info, top
) -> int:
    args = argparse.Namespace(top=top)

    # ---- overview ----------------------------------------------------------
    section("overview")
    print(f"  ktrace version: {ktrace_version()}")
    if device_name:
        print(f"  device        : {device_name}")
    if runtime_info.get("path"):
        print(f"  amdhip64 path : {runtime_info['path']}")
    if runtime_info.get("version"):
        print(f"  amdhip64 ver  : {runtime_info['version']}")
    c = ops["counts"]
    print(
        f"  commands      : {sum(c.values()):,}  "
        f"(dispatch {c.get('dispatch', 0):,}, copy {c.get('copy', 0):,}, "
        f"barrier {c.get('barrier', 0):,})"
    )
    print(f"  span          : {span / 1e6:.2f} s")
    print(f"  GPU busy      : {busy / 1e6:.2f} s ({100 * busy / span:.1f}% of span)")
    overlap = busy / sum(v[1] for v in ops["queues"].values())
    print(f"  queues        : {len(ops['queues'])}")
    for q, v in sorted(ops["queues"].items(), key=lambda kv: -kv[1][1]):
        print(
            f"    {q:<8} {v[0]:>9,} cmds  busy {v[1] / 1e6:>7.2f} s "
            f"({100 * v[1] / span:>5.1f}% of span)"
        )
    if abs(overlap - 1.0) < 0.01 and len(ops["queues"]) > 1:
        print("    queues never overlapped: the work is serialised across them")

    # ---- phases ------------------------------------------------------------
    if ops["phases"]:
        section("phases (inferred from kernel names)")
        lo0 = ops["phases"][0][1]
        runs: dict[str, list] = collections.defaultdict(lambda: [0, 0.0])
        for tag, a, b in ops["phases"]:
            runs[tag][0] += 1
            runs[tag][1] += b - a
        for tag, a, b in ops["phases"]:
            print(
                f"  {tag:<9}{(a - lo0) / 1e6:>8.2f} .. {(b - lo0) / 1e6:>6.2f} s"
                f"   ({(b - a) / 1e6:.2f} s)"
            )
        print()
        print(
            f"  {'phase':<9}{'runs':>6}{'wall_s':>9}{'gpu_s':>9}{'busy':>7}"
            + (f"{'host_wait_s':>13}" if api else "")
        )
        for tag, v in sorted(runs.items(), key=lambda kv: -kv[1][1]):
            gpu = ops["phase_busy"][tag] / 1e6
            row = (
                f"  {tag:<9}{v[0]:>6}{v[1] / 1e6:>9.2f}{gpu:>9.2f}"
                f"{100 * gpu / (v[1] / 1e6):>6.0f}%"
            )
            if api:
                row += f"{api['phase_wait'][tag] / 1e6:>13.2f}"
            print(row)
        for tag in sorted(runs, key=lambda t: -ops["phase_busy"][t]):
            top3 = ops["phase_kern"][tag].most_common(3)
            print(
                f"  {tag} is mostly: "
                + ", ".join(
                    f"{labels.get(n, n)[:38]} ({t / 1e6:.2f} s)" for n, t in top3
                )
            )

    # ---- kernels -----------------------------------------------------------
    section(f"top {args.top} kernels by GPU time")
    hdr = f"  {'kernel':<44}{'total_s':>9}{'launches':>10}{'avg_us':>10}"
    if kinfo:
        hdr += f"{'occ':>8}{'VGPR':>6}{'scratch':>9}{'LDS':>8}"
    print(hdr)
    for name, us in ops["kern_time"].most_common(args.top):
        row = (
            f"  {labels.get(name, name)[:44]:<44}{us / 1e6:>9.2f}"
            f"{ops['kern_count'][name]:>10,}{us / ops['kern_count'][name]:>10.1f}"
        )
        k = kinfo.get(name)
        if kinfo:
            # Module-launched kernels have no attributes: the launch API hands
            # over a function handle, not the host symbol the query needs.
            if k:
                occ = (
                    k["Theoretical_Occupancy_Pct"] + "%"
                    if k["Theoretical_Occupancy_Pct"]
                    else "-"
                )
                row += (
                    f"{occ:>8}{k['VGPR']:>6}{k['Scratch_Bytes']:>9}{k['Static_LDS']:>8}"
                )
            else:
                row += f"{'-':>8}{'-':>6}{'-':>9}{'-':>8}"
        print(row)

    # ---- diagnosis ---------------------------------------------------------
    if kinfo:
        section("kernels whose static footprint limits them")
        spill = [
            (n, t)
            for n, t in ops["kern_time"].items()
            if n in kinfo and int(kinfo[n]["Scratch_Bytes"]) > 0
        ]
        lowocc = [
            (n, t)
            for n, t in ops["kern_time"].items()
            if n in kinfo
            and kinfo[n]["Theoretical_Occupancy_Pct"]
            and 0 < float(kinfo[n]["Theoretical_Occupancy_Pct"]) < LOW_OCCUPANCY_PCT
        ]
        if spill:
            print(
                f"  register spill  : {len(spill)} kernels, "
                f"{100 * sum(t for _, t in spill) / busy:.1f}% of GPU time"
            )
            for n, t in sorted(spill, key=lambda x: -x[1])[:6]:
                print(
                    f"    {labels.get(n, n)[:44]:<44}{t / 1e6:>8.2f} s  "
                    f"scratch {kinfo[n]['Scratch_Bytes']:>6} B  VGPR {kinfo[n]['VGPR']:>4}"
                )
        if lowocc:
            print(
                f"  occupancy <{LOW_OCCUPANCY_PCT:.0f}%  : {len(lowocc)} kernels, "
                f"{100 * sum(t for _, t in lowocc) / busy:.1f}% of GPU time"
            )
            for n, t in sorted(lowocc, key=lambda x: -x[1])[:6]:
                k = kinfo[n]
                why = (
                    "registers"
                    if int(k["VGPR"]) >= 128
                    else ("LDS" if int(k["Static_LDS"]) > 16384 else "block size")
                )
                print(
                    f"    {labels.get(n, n)[:44]:<44}{t / 1e6:>8.2f} s  "
                    f"occ {k['Theoretical_Occupancy_Pct']:>5}%  limited by {why}"
                )
        if not spill and not lowocc:
            print("  nothing flagged")

    # ---- host side ---------------------------------------------------------
    if api:
        section(f"top {args.top} HIP calls by host time")
        print(f"  {'api':<46}{'count':>12}{'total_s':>10}{'avg_us':>13}")
        for name, v in sorted(api["per_api"].items(), key=lambda kv: -kv[1][1])[
            : args.top
        ]:
            print(f"  {name:<46}{v[0]:>12,}{v[1] / 1e6:>10.2f}{v[1] / v[0]:>13.3f}")

        if api["stream_wait"]:
            section("host wait, by the queue it waited on")
            print(f"  {'stream':<20}{'hw queue':<10}{'sync calls':>12}{'wait_s':>10}")
            for s, (n, us) in sorted(
                api["stream_wait"].items(), key=lambda kv: -kv[1][1]
            ):
                q = ""
                if api["stream_queue"].get(s):
                    q = api["stream_queue"][s].most_common(1)[0][0]
                print(f"  {s:<20}{q or '?':<10}{n:>12,}{us / 1e6:>10.2f}")

    # ---- memory ------------------------------------------------------------
    section("memory")
    if ops["xfer"]:
        print(f"  {'transfer':<32}{'count':>10}{'total':>12}{'GB/s':>9}")
        for kind, v in sorted(ops["xfer"].items(), key=lambda kv: -kv[1][1]):
            rate = v[1] / (v[2] * 1000.0) if v[2] else 0.0
            print(f"  {kind:<32}{v[0]:>10,}{human_bytes(v[1]):>12}{rate:>9.1f}")
    if api:
        print("  requested by the application (from hipMalloc/hipFree):")
        print(f"    peak held   : {human_bytes(api['alloc_peak'])}")
        print(f"    cumulative  : {human_bytes(api['alloc_total'])}")
        print(
            f"    still held  : {human_bytes(api['alloc_leaked'])} "
            f"in {api['alloc_blocks']} blocks"
            + (
                f", {api['unmatched_free']} unmatched frees"
                if api["unmatched_free"]
                else ""
            )
        )
    if vram:
        print(
            f"  reported by the driver ({vram['n']} samples over {vram['span_s']:.1f} s):"
        )
        print(f"    peak used   : {human_bytes(vram['max'])}")
        print(f"    growth      : {human_bytes(vram['last'] - vram['first'])}")
        if api and api["alloc_peak"]:
            gap = vram["max"] - api["alloc_peak"]
            print(
                f"    over what the application asked for: {human_bytes(gap)} "
                f"(runtime buffers, pools, fragmentation)"
            )
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
