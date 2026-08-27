#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""The convolution shape sweep: what to test, and how to build it.

Kept apart from test_conv.py, and free of any dependency beyond onnx and
numpy, so the same cases can be driven either through pytest (against the ORT
CPU reference) or through hip-onnx-runner directly. The two paths need
different plumbing but must not be allowed to test different shapes.

The sweep exists because `hip_conv` is not one kernel. It picks among a WMMA
tile ladder, a scalar tile ladder, a depthwise kernel and a
one-thread-per-output fallback, and the choice is a function of the
output-channel count, the output extent, the batch, the group count *and* the
device's CU count. A test that covers one shape covers one kernel, and the
shapes selecting the rarely-taken rungs have the least coverage precisely
because they are rare.

Cases are grouped by what selects a rung rather than by which model they came
from, so a new tile in the ladder has an obvious place to be tested from, and
each is the smallest shape that still lands where it is named to land -- the
rung depends on channel counts and extents, not on how much data flows.
"""

from __future__ import annotations

import numpy as np
from onnx import TensorProto, helper, numpy_helper

_ONNX_DTYPE = {
    "float16": TensorProto.FLOAT16,
    "float32": TensorProto.FLOAT,
}
_NP_DTYPE = {"float16": np.float16, "float32": np.float32}


# Each case names the property that puts it in the sweep. `id` is what pytest
# prints and what names the generated model file, so it names the rung or the
# hazard rather than the shape.
CONV_CASES = [
    # -- WMMA tile ladder: one case per rung -------------------------------
    dict(id="wmma-128x256", cin=64, cout=256, spatial=(56, 56), k=(1, 1)),
    dict(id="wmma-256x128", cin=256, cout=1024, spatial=(14, 14), k=(1, 1)),
    dict(id="wmma-64x256", cin=3, cout=64, spatial=(224, 224), k=(7, 7),
         stride=(2, 2), pad=(3, 3)),
    dict(id="wmma-32x256", cin=64, cout=32, spatial=(64, 64)),
    dict(id="wmma-16x256-narrow", cin=64, cout=3, spatial=(64, 64)),
    # resnet50 stage 4. On a 20-CU part every wider tile fails the coverage
    # test and these land on the 16-row rung; on a 16-CU part they do not.
    # Both have to be correct, and the sweep runs on whatever the device is.
    dict(id="deep-stage-512ch-7x7", cin=2048, cout=512, spatial=(7, 7), k=(1, 1)),
    dict(id="deep-stage-512ch-7x7-3x3", cin=512, cout=512, spatial=(7, 7)),
    dict(id="deep-stage-2048ch-7x7", cin=512, cout=2048, spatial=(7, 7), k=(1, 1)),
    # -- Contraction tail: K not a multiple of the 16-deep staging slice ---
    # The staging loops walk a full slice whether or not the contraction has
    # that many taps left, so each of these drives the out-of-range sentinel
    # in makeKSlice at a different remainder.
    dict(id="ktail-K27", cin=3, cout=64, spatial=(32, 32)),        # 3*9
    dict(id="ktail-K9", cin=1, cout=32, spatial=(32, 32)),         # 1*9
    dict(id="ktail-K45", cin=5, cout=64, spatial=(32, 32)),        # 5*9
    dict(id="ktail-K7", cin=7, cout=64, spatial=(32, 32), k=(1, 1)),
    dict(id="ktail-K33", cin=11, cout=48, spatial=(17, 17)),       # 11*9, odd extent
    # -- 3x3 stride 1: the class Winograd takes over -----------------------
    dict(id="w3x3-64to64", cin=64, cout=64, spatial=(56, 56)),
    dict(id="w3x3-128to128", cin=128, cout=128, spatial=(28, 28)),
    dict(id="w3x3-256to256", cin=256, cout=256, spatial=(14, 14)),
    dict(id="w3x3-192to64", cin=192, cout=64, spatial=(32, 32)),
    dict(id="w3x3-odd-extent", cin=32, cout=32, spatial=(13, 13)),
    dict(id="w3x3-batch4", cin=32, cout=64, spatial=(16, 16), n=4),
    dict(id="w3x3-tiny-extent", cin=32, cout=32, spatial=(3, 3)),
    dict(id="w3x3-pad0", cin=32, cout=64, spatial=(16, 16), pad=(0, 0)),
    # -- Strided, dilated, asymmetric --------------------------------------
    dict(id="stride2-3x3", cin=64, cout=128, spatial=(28, 28), stride=(2, 2)),
    dict(id="stride3-3x3", cin=32, cout=64, spatial=(30, 30), stride=(3, 3)),
    dict(id="dilation2", cin=32, cout=64, spatial=(24, 24), dilation=(2, 2)),
    dict(id="nonsquare-kernel", cin=32, cout=64, spatial=(24, 24), k=(1, 3)),
    dict(id="nonsquare-spatial", cin=32, cout=64, spatial=(12, 40)),
    # -- Grouped and depthwise ---------------------------------------------
    dict(id="depthwise-3x3", cin=64, cout=64, group=64, spatial=(28, 28)),
    dict(id="depthwise-7x7", cin=96, cout=96, group=96, spatial=(28, 28),
         k=(7, 7)),
    dict(id="grouped-4", cin=64, cout=128, group=4, spatial=(28, 28)),
    dict(id="grouped-narrow-rows", cin=64, cout=8, group=4, spatial=(28, 28)),
    # -- fp32 goes down the scalar ladder, not the matrix cores ------------
    dict(id="fp32-3x3", cin=64, cout=64, spatial=(28, 28), dtype="float32"),
    dict(id="fp32-1x1", cin=256, cout=256, spatial=(14, 14), k=(1, 1),
         dtype="float32"),
    dict(id="fp32-ktail", cin=3, cout=64, spatial=(32, 32), dtype="float32"),
    dict(id="fp32-deep-512ch", cin=512, cout=512, spatial=(7, 7),
         dtype="float32"),
    # -- bevformer FPN: the shape that produced 33% NaN after MIOpen removal
    dict(id="bevformer-fpn-1x1", cin=2048, cout=256, spatial=(15, 25),
         k=(1, 1)),
    dict(id="bevformer-fpn-3x3", cin=256, cout=256, spatial=(15, 25)),
    dict(id="bevformer-fpn-c5", cin=1024, cout=256, spatial=(29, 50),
         k=(1, 1)),
]

CONV_TRANSPOSE_CASES = [
    # sam2.1's decoder: kernel == stride, so output footprints are disjoint
    # and every output cell has exactly one contributing tap.
    dict(id="ct-2x2s2-256to64", op="ConvTranspose", cin=256, cout=64,
         spatial=(16, 16), k=(2, 2), stride=(2, 2), pad=(0, 0)),
    dict(id="ct-2x2s2-64to32", op="ConvTranspose", cin=64, cout=32,
         spatial=(32, 32), k=(2, 2), stride=(2, 2), pad=(0, 0)),
    # Overlapping footprints: stride < kernel, the case a disjoint fast path
    # must refuse.
    dict(id="ct-4x4s2-overlap", op="ConvTranspose", cin=32, cout=16,
         spatial=(16, 16), k=(4, 4), stride=(2, 2), pad=(1, 1)),
    dict(id="ct-3x3s1", op="ConvTranspose", cin=32, cout=32,
         spatial=(16, 16), k=(3, 3), stride=(1, 1), pad=(1, 1)),
    dict(id="ct-3x3s2-pad", op="ConvTranspose", cin=16, cout=16,
         spatial=(12, 12), k=(3, 3), stride=(2, 2), pad=(1, 1)),
    dict(id="ct-grouped", op="ConvTranspose", cin=32, cout=32, group=4,
         spatial=(16, 16), k=(2, 2), stride=(2, 2), pad=(0, 0)),
    dict(id="ct-fp32-2x2s2", op="ConvTranspose", cin=64, cout=32,
         spatial=(16, 16), k=(2, 2), stride=(2, 2), pad=(0, 0),
         dtype="float32"),
    dict(id="ct-dilation2", op="ConvTranspose", cin=16, cout=16,
         spatial=(12, 12), k=(3, 3), stride=(1, 1), pad=(2, 2),
         dilation=(2, 2)),
]

ALL_CASES = CONV_CASES + CONV_TRANSPOSE_CASES


def output_extent(case):
    op = case.get("op", "Conv")
    kh, kw = case.get("k", (3, 3))
    sh, sw = case.get("stride", (1, 1))
    dh, dw = case.get("dilation", (1, 1))
    ih, iw = case["spatial"]
    ph, pw = case.get("pad", ((kh - 1) * dh // 2, (kw - 1) * dw // 2))
    if op == "Conv":
        oh = (ih + 2 * ph - (dh * (kh - 1) + 1)) // sh + 1
        ow = (iw + 2 * pw - (dw * (kw - 1) + 1)) // sw + 1
    else:
        oh = (ih - 1) * sh - 2 * ph + dh * (kh - 1) + 1
        ow = (iw - 1) * sw - 2 * pw + dw * (kw - 1) + 1
    return oh, ow, ph, pw


def build(case):
    """Return (ModelProto, input ndarray) for one sweep case."""
    op = case.get("op", "Conv")
    dt = case.get("dtype", "float16")
    onnx_t, np_t = _ONNX_DTYPE[dt], _NP_DTYPE[dt]

    n = case.get("n", 1)
    cin, cout = case["cin"], case["cout"]
    g = case.get("group", 1)
    kh, kw = case.get("k", (3, 3))
    sh, sw = case.get("stride", (1, 1))
    dh, dw = case.get("dilation", (1, 1))
    ih, iw = case["spatial"]
    oh, ow, ph, pw = output_extent(case)

    assert cin % g == 0 and cout % g == 0, f"{case['id']}: group does not divide"
    assert oh > 0 and ow > 0, f"{case['id']}: empty output"

    wshape = [cout, cin // g, kh, kw] if op == "Conv" else [cin, cout // g, kh, kw]

    inp = helper.make_tensor_value_info("input", onnx_t, [n, cin, ih, iw])
    out = helper.make_tensor_value_info("output", onnx_t, [n, cout, oh, ow])

    # A narrow weight range keeps the fp16 accumulation well inside range for
    # the long contractions here (K reaches 2048); the sweep is about kernel
    # selection, not saturation behaviour.
    rng = np.random.default_rng(case.get("seed", 11))
    scale = 0.1 if dt == "float16" else 0.5
    w = rng.uniform(-scale, scale, wshape).astype(np_t)
    b = rng.uniform(-scale, scale, [cout]).astype(np_t)

    node = helper.make_node(
        op,
        ["input", "weight", "bias"],
        ["output"],
        group=g,
        kernel_shape=[kh, kw],
        strides=[sh, sw],
        pads=[ph, pw, ph, pw],
        dilations=[dh, dw],
    )
    graph = helper.make_graph(
        [node],
        f"conv_sweep_{case['id']}",
        [inp],
        [out],
        initializer=[
            numpy_helper.from_array(w, name="weight"),
            numpy_helper.from_array(b, name="bias"),
        ],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", 17)]
    )
    model.ir_version = 10

    x = rng.uniform(-1.0, 1.0, [n, cin, ih, iw]).astype(np_t)
    return model, x
