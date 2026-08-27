#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Extract every distinct convolution shape a model asks the EP to run.

The tile-selection heuristic in conv_kernel.hip is a pure function of a
convolution's shape and the device's CU count, so it can be evaluated against
the shapes the models actually contain without running the models. This dumps
those shapes; conv_microbench.py times them.

Shapes are deduplicated and counted. That is the whole point of the file: a
model's convolution *node* count is a bad guide to its distinct *shape* count
-- esrgan has 351 Conv nodes and 8 distinct shapes -- and a heuristic change
only has to be re-measured once per distinct shape.

The `count` on each record is the node count, which is also the weight to use
when ranking shapes by how much of a model's runtime they can explain.

Output is JSON on stdout, or to --out:

  {"<model>": [{"op": "Conv", "dtype": "float16", "N": 1, ...,
                "count": 69}, ...], ...}
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys

import onnx
from onnx import TensorProto

# ONNX elem_type -> the name hip_conv's dtype switch understands.
_DTYPE = {
    TensorProto.FLOAT: "float32",
    TensorProto.FLOAT16: "float16",
    TensorProto.BFLOAT16: "bfloat16",
}


def _attr(node, name, default=None):
    for a in node.attribute:
        if a.name == name:
            if a.ints:
                return list(a.ints)
            if a.HasField("i"):
                return a.i
            if a.s:
                return a.s.decode()
    return default


def _dims(vi):
    """Static dims of a ValueInfo, or None if any axis is dynamic.

    A dynamic axis is almost always the batch, which the harness pins to 1
    anyway, so it is substituted rather than dropping the shape.
    """
    if not vi.type.HasField("tensor_type"):
        return None
    out = []
    for d in vi.type.tensor_type.shape.dim:
        if d.HasField("dim_value") and d.dim_value > 0:
            out.append(int(d.dim_value))
        else:
            out.append(1)  # dynamic axis: the harness runs batch 1
    return out


def _infer(model):
    """Shape-infer, tolerating models the checker rejects.

    Several of the vision models fail strict checking on unrelated nodes; the
    inferred shapes for the convolutions are still usable, so failures here
    fall back to whatever the graph already declares.
    """
    try:
        return onnx.shape_inference.infer_shapes(model, strict_mode=False)
    except Exception:  # noqa: BLE001
        return model


def shapes_for_model(path: str):
    model = onnx.load(path, load_external_data=False)
    model = _infer(model)
    g = model.graph

    vis = {}
    for coll in (g.input, g.output, g.value_info):
        for vi in coll:
            vis[vi.name] = vi
    winit = {i.name: i for i in g.initializer}

    seen = collections.OrderedDict()
    for node in g.node:
        if node.op_type not in ("Conv", "ConvTranspose"):
            continue
        if len(node.input) < 2:
            continue

        xvi = vis.get(node.input[0])
        yvi = vis.get(node.output[0])
        w = winit.get(node.input[1])
        if xvi is None or yvi is None or w is None:
            continue
        xd, yd, wd = _dims(xvi), _dims(yvi), list(w.dims)
        if not xd or not yd or len(xd) < 3 or len(yd) < 3 or len(wd) < 3:
            continue

        rank = len(xd) - 2
        if rank < 1 or rank > 3:
            continue

        dtype = _DTYPE.get(xvi.type.tensor_type.elem_type)
        if dtype is None:
            continue

        group = _attr(node, "group", 1) or 1
        kernel = _attr(node, "kernel_shape") or wd[2:]
        stride = _attr(node, "strides") or [1] * rank
        dil = _attr(node, "dilations") or [1] * rank
        pads = _attr(node, "pads") or [0] * (2 * rank)

        # hip_conv takes pads_begin only; the trailing pad is already folded
        # into the caller's output extent.
        rec = {
            "op": node.op_type,
            "dtype": dtype,
            "rank": rank,
            "N": xd[0],
            "Cin": xd[1],
            "Cout": yd[1],
            "in": xd[2:] + [1] * (3 - rank),
            "out": yd[2:] + [1] * (3 - rank),
            "kernel": list(kernel) + [1] * (3 - rank),
            "stride": list(stride) + [1] * (3 - rank),
            "pad_begin": list(pads[:rank]) + [0] * (3 - rank),
            "dilation": list(dil) + [1] * (3 - rank),
            "group": group,
            "bias": len(node.input) > 2,
        }
        key = json.dumps(rec, sort_keys=True)
        if key in seen:
            seen[key]["count"] += 1
        else:
            rec["count"] = 1
            seen[key] = rec

    return list(seen.values())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("models", nargs="+", metavar="NAME=PATH",
                    help="Model to census. Bare paths are named by basename.")
    ap.add_argument("--out", help="Write JSON here instead of stdout.")
    args = ap.parse_args()

    result = {}
    for spec in args.models:
        name, _, path = spec.partition("=")
        if not path:
            name, path = os.path.basename(os.path.dirname(spec)) or spec, spec
        if not os.path.exists(path):
            print(f"MISSING {name}: {path}", file=sys.stderr)
            continue
        try:
            result[name] = shapes_for_model(path)
        except Exception as e:  # noqa: BLE001
            print(f"FAILED {name}: {e}", file=sys.stderr)
            continue
        nodes = sum(r["count"] for r in result[name])
        print(f"{name:<34} {len(result[name]):>4} distinct / {nodes:>4} nodes",
              file=sys.stderr)

    text = json.dumps(result, indent=2)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
