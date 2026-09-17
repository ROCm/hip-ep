#!/usr/bin/env python3
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Gate a .rgp capture before any number decoded from it is trusted.

A capture with CodeObjects but no SqttData has no timeline at all, and one
without SpmCounterData has no bandwidth, so the memory/compute bound class
degrades to "undetermined". Both failure modes are silent downstream -- the
parser happily emits a CSV either way -- so they are checked up front rather
than discovered halfway through an analysis.
"""

import argparse
import pathlib
import sys

# tools/perf-harness/capture -> tools/rgp_parser
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "rgp_parser"))

from rgp_parser.struct.rdf import RdfFile  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("capture", help="path to the .rgp file")
    ap.add_argument("--require-spm", action="store_true",
                    help="also fail when the capture carries no SPM counters")
    args = ap.parse_args()

    counts = RdfFile.from_path(args.capture).counts()
    for name in sorted(counts):
        print(f"  {name:24s} {counts[name]}")

    sqtt = counts.get("SqttData", 0)
    spm = counts.get("SpmCounterData", 0)
    ok = bool(sqtt) and (bool(spm) or not args.require_spm)
    print(f"VERDICT sqtt={sqtt} spm={spm} -> {'USABLE' if ok else 'UNUSABLE'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
