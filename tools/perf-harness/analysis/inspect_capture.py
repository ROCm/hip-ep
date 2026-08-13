#!/usr/bin/env python3
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""First look at a capture: is this actually the workload you meant to measure?

Run this before any analysis. A capture can be perfectly valid -- SqttData
present, kernels symbolised, totals plausible -- and still be worthless because
of where it landed. Two failure modes this catches, both of which have produced
confident wrong answers:

  autotune sweep   Fencing on the first instance of an op can land before the
                   tuner has settled. The window then shows hundreds of
                   consecutive dispatches of one kernel cycling through a handful
                   of configurations. Those are tuning trials, not steady state,
                   and their timings mean nothing. Re-capture with a larger
                   --skip.
  wrong depth      Attention cost depends on how much context exists, so a
                   capture's chunk position has to be known before its attention
                   numbers can be used. The layer stride and the KV-dependent
                   spread give it away.

Also cross-checks the model config when one is given, which is how you confirm
things like lm_head really running over the full vocabulary for every row.
"""

import argparse
import json
from collections import Counter, defaultdict

from perfcommon import Capture, add_common_args, specs_from_args


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap, many=True)
    ap.add_argument("--genai-config", help="genai_config.json to cross-check shapes against")
    ap.add_argument("--sweep-threshold", type=int, default=40,
                    help="consecutive same-kernel dispatches that look like a tuner sweep")
    args = ap.parse_args()
    spec, _ = specs_from_args(args)

    for path in args.captures:
        cap = Capture(path, spec)
        print(f"\n######## {path}")
        print(f"{len(cap.rows)} dispatches, {cap.total_us/1000:.1f} ms, "
              f"{cap.layers_in_window} topk_routing (layer groups)")

        tk = [i for i, r in enumerate(cap.rows) if r["family"] == "topk_routing"]
        if len(tk) > 1:
            strides = [b - a for a, b in zip(tk, tk[1:])]
            print(f"layer starts at {tk[:8]}{' ...' if len(tk) > 8 else ''}  "
                  f"stride {min(strides)}..{max(strides)}")

        # --- runaway repetition: the tuner-sweep signature --------------------
        runs, cur, n = [], None, 0
        for r in cap.rows:
            if r["family"] == cur:
                n += 1
            else:
                if cur is not None and n >= args.sweep_threshold:
                    runs.append((cur, n))
                cur, n = r["family"], 1
        if cur is not None and n >= args.sweep_threshold:
            runs.append((cur, n))
        if runs:
            print("\n!! long consecutive runs of one kernel -- check for a tuner sweep:")
            for fam, count in runs:
                durs = [float(r["dur_us"]) for r in cap.rows if r["family"] == fam]
                distinct = len({round(d, 1) for d in durs})
                share = 100 * sum(durs) / cap.total_us
                print(f"   {fam:34} {count:5d} in a row, {share:5.1f}% of the window, "
                      f"{distinct} distinct durations")
            print("   One kernel running back to back and dominating the window is the")
            print("   autotuner working through configurations, not steady state -- the")
            print("   durations are trials. Re-capture with a larger --skip.")
        else:
            print("no runaway kernel repetition (no obvious tuner sweep)")

        # --- the big dispatches ----------------------------------------------
        big = sorted(((float(r["dur_us"]), r["family"], int(r["threads"]))
                      for r in cap.rows), reverse=True)[:8]
        print(f"\n{'longest dispatches':34} {'us':>9} {'threads':>12}")
        for dur, fam, threads in big:
            print(f"  {fam:32} {dur:9.1f} {threads:12d}")
        if cap.lm_head_idx:
            print(f"lm_head: {len(cap.lm_head_idx)} dispatch(es), {cap.lm_head_us/1000:.1f} ms, "
                  f"threads={int(cap.rows[cap.lm_head_idx[0]]['threads']):,}")

        # --- expert routing spread -------------------------------------------
        if cap.blocks:
            ms = sorted(b.m for b in cap.blocks)
            print(f"\nexpert blocks {len(cap.blocks)}: M from {ms[0]} to {ms[-1]}, "
                  f"median {ms[len(ms)//2]}, {sum(ms)} routed tokens")

        # --- parser-flagged artifacts ----------------------------------------
        if cap.rows and "is_artifact" in cap.rows[0]:
            flags = Counter(r["is_artifact"] for r in cap.rows)
            art = [r for r in cap.rows if r["is_artifact"] not in ("0", "False", "false", "")]
            print(f"\nis_artifact {dict(flags)}"
                  + (f" -- {len(art)} flagged, "
                     f"{100*sum(float(r['dur_us']) for r in art)/cap.total_us:.1f}% of the window"
                     if art else ""))

        # --- config cross-check -----------------------------------------------
        if args.genai_config:
            cfg = json.load(open(args.genai_config))
            model = cfg.get("model", {})
            dec = model.get("decoder", {})
            print(f"\nconfig: vocab={model.get('vocab_size')} "
                  f"hidden={dec.get('hidden_size')} layers={dec.get('num_hidden_layers')}")
            for out in dec.get("outputs", {}).items():
                print(f"  output {out[0]}: {out[1]}")

        fams: dict[str, list] = defaultdict(lambda: [0, 0.0])
        for r in cap.rows:
            fams[r["family"]][0] += 1
            fams[r["family"]][1] += float(r["dur_us"])
        print(f"\n{'family':36} {'n':>6} {'ms':>9} {'%':>7}")
        for fam, (n, us) in sorted(fams.items(), key=lambda kv: -kv[1][1])[:14]:
            print(f"{fam:36} {n:6d} {us/1000:9.2f} {100*us/cap.total_us:7.2f}")


if __name__ == "__main__":
    main()
