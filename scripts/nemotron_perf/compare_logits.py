#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""Compare raw logits captured with identical prompt and decode histories."""

import argparse
import json
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument(
        "--output", type=Path, help="New JSON file for all per-step results"
    )
    args = parser.parse_args()
    for filename in ("prompt-tokens.txt", "tokens.txt"):
        if (args.baseline / filename).read_bytes() != (
            args.candidate / filename
        ).read_bytes():
            raise ValueError(f"History mismatch: {filename}")
    steps = len((args.baseline / "tokens.txt").read_text().splitlines())
    a = np.fromfile(args.baseline / "logits.f32", dtype=np.float32).reshape(steps, -1)
    b = np.fromfile(args.candidate / "logits.f32", dtype=np.float32).reshape(steps, -1)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Shape mismatch or non-finite logits")
    rows = []
    for step, (x, y) in enumerate(zip(a.astype(np.float64), b.astype(np.float64))):
        difference = y - x
        p = np.exp(x - x.max())
        q = np.exp(y - y.max())
        p /= p.sum()
        q /= q.sum()
        top = np.argpartition(x, -2)[-2:]
        rows.append(
            {
                "step": step,
                "max_abs": float(np.max(np.abs(difference))),
                "relative_l2": float(np.linalg.norm(difference) / np.linalg.norm(x)),
                "baseline_top1": int(np.argmax(x)),
                "candidate_top1": int(np.argmax(y)),
                "baseline_top2_margin": float(abs(x[top[0]] - x[top[1]])),
                "probability_total_variation": float(np.abs(p - q).sum() / 2),
            }
        )
    result = {
        "baseline": str(args.baseline),
        "candidate": str(args.candidate),
        "steps": steps,
        "vocab": a.shape[1],
        "identical_history": True,
        "top1_agreement": sum(r["baseline_top1"] == r["candidate_top1"] for r in rows)
        / steps,
        "worst_relative_l2": max(r["relative_l2"] for r in rows),
        "worst_max_abs": max(r["max_abs"] for r in rows),
        "worst_probability_total_variation": max(
            r["probability_total_variation"] for r in rows
        ),
        "per_step": rows,
    }
    if args.output:
        with args.output.open("x") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
        result = {key: value for key, value in result.items() if key != "per_step"}
        result["details"] = str(args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
