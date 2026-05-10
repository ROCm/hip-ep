#!/usr/bin/env python3
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""
Generate input_<idx>_<name>_<type>.bin files for hip-onnx-runner (-i).

Loads ONNX with load_external_data=False (metadata only). Resolves symbolic
dims via --dim NAME=INT (defaults suit GPT-OSS style decode: batch=1, new
tokens=1, past len=1, mask total = past+new => 2).

INT64:
  * Names containing "input_ids" / ending with "input_ids": uniform [0, vocab-1]
  * Names containing "mask": all ones
  * Other int64: zeros

FLOAT16: uniform in IEEE fp16 finite range (~[-65504, 65504]).

Vocab size is taken from initializer model.embed_tokens.weight dim0 if present.

Usage:
  python tools/gen_hip_onnx_runner_inputs.py Z:\\...\\model_q4f16.onnx
  python tools/gen_hip_onnx_runner_inputs.py model.onnx -o out_dir --seed 0
  python tools/gen_hip_onnx_runner_inputs.py model.onnx -D total_sequence_length=1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def default_dim_overrides() -> dict[str, int]:
    return {
        "batch_size": 1,
        "sequence_length": 1,
        "past_sequence_length": 1,
        # mask length often past_len + new_token_len for incremental decode
        "total_sequence_length": 2,
    }


def parse_dim_overrides(specs: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for s in specs:
        if "=" not in s:
            print(f"ERROR: bad -D {s!r} (want name=int)", file=sys.stderr)
            sys.exit(2)
        k, v = s.split("=", 1)
        out[k.strip()] = int(v.strip())
    return out


def vocab_from_embed(m) -> int | None:
    for init in m.graph.initializer:
        if init.name.endswith("embed_tokens.weight") and len(init.dims) >= 2:
            return int(init.dims[0])
    return None


def resolve_shape(
    tensor_type, dim_defaults: dict[str, int]
) -> tuple[list[int], list[str]]:
    shape: list[int] = []
    params: list[str] = []
    for d in tensor_type.shape.dim:
        if d.dim_value:
            shape.append(int(d.dim_value))
            params.append("")
        elif d.dim_param:
            p = d.dim_param
            params.append(p)
            if p not in dim_defaults:
                print(
                    f"ERROR: unknown dim_param {p!r}; add -D {p}=<int>",
                    file=sys.stderr,
                )
                sys.exit(2)
            shape.append(int(dim_defaults[p]))
        else:
            shape.append(1)
            params.append("")
    return shape, params


def fill_tensor(
    name: str,
    shape: list[int],
    elem_type: int,
    rng: np.random.Generator,
    vocab: int | None,
) -> np.ndarray:
    import onnx  # noqa: PLC0415

    if elem_type == onnx.TensorProto.INT64:
        n = int(np.prod(shape)) if shape else 0
        if n == 0:
            return np.array([], dtype=np.int64).reshape(shape)
        if name == "input_ids" or name.endswith("input_ids"):
            hi = vocab if vocab is not None else 32000
            return rng.integers(0, hi, size=shape, dtype=np.int64)
        if "mask" in name.lower():
            return np.ones(shape, dtype=np.int64)
        return np.zeros(shape, dtype=np.int64)
    if elem_type == onnx.TensorProto.FLOAT16:
        f16_hi = float(np.finfo(np.float16).max)
        f16_lo = float(np.finfo(np.float16).min)
        return rng.uniform(f16_lo, f16_hi, size=shape).astype(np.float16)
    print(f"ERROR: unsupported elem_type {elem_type} for {name}", file=sys.stderr)
    sys.exit(2)


def file_tag_for_numpy(arr: np.ndarray) -> str:
    if arr.dtype == np.int64:
        return "i64"
    if arr.dtype == np.float16:
        return "fp16"
    if arr.dtype == np.float32:
        return "fp32"
    print(f"ERROR: no file tag for dtype {arr.dtype}", file=sys.stderr)
    sys.exit(2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("onnx_path", type=Path, help="Path to .onnx")
    ap.add_argument(
        "-o",
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: <onnx_stem>_inputs next to model)",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "-D",
        "--dim",
        action="append",
        default=[],
        metavar="NAME=INT",
        help="Override symbolic dimension (repeatable)",
    )
    args = ap.parse_args()

    onnx_p = args.onnx_path
    if not onnx_p.is_file():
        print(f"ERROR: model not found: {onnx_p}", file=sys.stderr)
        return 1

    import onnx  # noqa: PLC0415

    model = onnx.load(str(onnx_p), load_external_data=False)
    initializer_names = {x.name for x in model.graph.initializer}
    dim_defaults = default_dim_overrides()
    dim_defaults.update(parse_dim_overrides(args.dim))

    vocab = vocab_from_embed(model)

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = onnx_p.parent / f"{onnx_p.stem}_inputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    user_inputs: list[tuple[int, str, np.ndarray]] = []
    idx = 0
    for inp in model.graph.input:
        if inp.name in initializer_names:
            continue
        tt = inp.type.tensor_type
        shape, _params = resolve_shape(tt, dim_defaults)
        et = int(tt.elem_type)
        arr = fill_tensor(inp.name, shape, et, rng, vocab)
        user_inputs.append((idx, inp.name, arr))
        idx += 1

    lines = [
        f"Generated for: {onnx_p}",
        f"seed={args.seed}",
        f"dim_defaults_used={dim_defaults!r}",
        f"vocab_size(from embed)={vocab}",
        "",
        "Run:",
        f'  hip-onnx-runner.exe -m "{onnx_p}" -i "{out_dir}" ...',
        "",
    ]

    for i, name, arr in user_inputs:
        tag = file_tag_for_numpy(arr)
        fn = out_dir / f"input_{i}_{name}_{tag}.bin"
        fn.write_bytes(arr.tobytes(order="C"))
        print(f"Wrote {fn} ({arr.nbytes} B) shape={arr.shape} dtype={arr.dtype}")
        lines.append(f"  {fn.name} shape={list(arr.shape)} {arr.dtype}")

    readme = out_dir / "README.txt"
    readme.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(readme.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
