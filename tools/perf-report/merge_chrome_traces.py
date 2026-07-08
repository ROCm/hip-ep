"""Merge N Chrome-trace JSONs into ONE file (chrome://tracing / Perfetto load a
single file only). Each input is placed in its own process (pid) block so they
render as separate, stacked track groups.

Usage:
    python merge_traces.py out.json in1.json [in2.json ...] [--zero]

- Same-clock inputs (e.g. EP per-session files sharing the absolute trace axis,
  produced by op_profile.cpp) align automatically -- do NOT pass --zero.
- Different-run/clock inputs (EP vs SQTT vs RGP golden) do not share a clock;
  pass --zero to shift each to start at t=0 so they stack from a common origin
  for shape comparison (they are NOT wall-clock aligned across runs).
"""
import json, sys, os

def main(argv):
    if len(argv) < 2:
        print(__doc__); return 2
    zero = "--zero" in argv
    args = [a for a in argv if a != "--zero"]
    out_path, inputs = args[0], args[1:]
    if not inputs:
        print(__doc__); return 2

    merged = []
    for i, path in enumerate(inputs):
        d = json.load(open(path))
        evs = d["traceEvents"] if isinstance(d, dict) else d
        base_pid = 1000 * (i + 1)
        label = os.path.basename(path)
        # per-input time shift so inputs stack from a common origin if requested
        shift = 0.0
        if zero:
            ts = [e["ts"] for e in evs if e.get("ph") == "X" and "ts" in e]
            if ts:
                shift = min(ts)
        seen_pids = set()
        for e in evs:
            e = dict(e)
            if "pid" in e:
                e["pid"] = base_pid + int(e["pid"])
                seen_pids.add(e["pid"])
            if zero and e.get("ph") == "X" and "ts" in e:
                e["ts"] = e["ts"] - shift
            merged.append(e)
        # a process_name so each input is clearly labeled in the merged view
        for pid in (seen_pids or {base_pid}):
            merged.append({"name": "process_name", "ph": "M", "pid": pid, "tid": 0,
                           "args": {"name": f"[{i}] {label}"}})

    json.dump({"displayTimeUnit": "ns", "traceEvents": merged}, open(out_path, "w"))
    xs = sum(1 for e in merged if e.get("ph") == "X")
    print(f"wrote {out_path}: {len(inputs)} inputs, {xs} span events, zero={zero}")

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]) or 0)
