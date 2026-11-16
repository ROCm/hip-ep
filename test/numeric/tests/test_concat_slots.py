#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Numeric checks for channel concats written in place (hip-forward-concat-slots).

A batch-1 channel concat lets the producers write straight into the concat
buffer, readers of a concat input read the concat buffer, and a dense block
(each concat repeats the previous one's inputs and appends one more) shares
one buffer for the whole block. Batch 2 makes every slot strided, so the
same graphs keep their copies and act as the baseline.

The rewrite only applies when the convs go through rocMLIR, so run with
HIPDNN_EP_ROCMLIR=1; without it the same graphs check the MIOpen path.
"""

import numpy as np
import pytest
from onnx import helper, numpy_helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes, np_to_onnx_type

_DTYPES = [np.float32, np.float16]
_TOL = {np.float32: 1e-3, np.float16: 2e-2}


class _Graph:
    def __init__(self, dtype, seed):
        self.dtype = dtype
        self.rng = np.random.default_rng(seed)
        self.nodes = []
        self.inits = []

    def conv(self, x, in_c, out_c, k=3, relu=False):
        name = f"t{len(self.nodes)}"
        bound = 1.0 / np.sqrt(in_c * k * k)
        w = self.rng.uniform(-bound, bound, [out_c, in_c, k, k]).astype(self.dtype)
        b = self.rng.uniform(-0.1, 0.1, [out_c]).astype(self.dtype)
        self.inits += [
            numpy_helper.from_array(w, name=f"{name}_w"),
            numpy_helper.from_array(b, name=f"{name}_b"),
        ]
        pad = k // 2
        self.nodes.append(
            helper.make_node(
                "Conv",
                [x, f"{name}_w", f"{name}_b"],
                [name],
                kernel_shape=[k, k],
                pads=[pad] * 4,
            )
        )
        if not relu:
            return name
        return self.op("Relu", [name])

    def op(self, op_type, inputs, **attrs):
        name = f"t{len(self.nodes)}"
        self.nodes.append(helper.make_node(op_type, inputs, [name], **attrs))
        return name

    def const(self, value):
        name = f"c{len(self.inits)}"
        self.inits.append(numpy_helper.from_array(np.asarray(value), name=name))
        return name

    def model(self, inputs, outputs):
        tp = np_to_onnx_type(self.dtype)
        ins = [helper.make_tensor_value_info(n, tp, s) for n, s in inputs]
        outs = [helper.make_tensor_value_info(n, tp, s) for n, s in outputs]
        return make_model_from_nodes(self.nodes, ins, outs, initializers=self.inits)


def _rename_output(g, old, new):
    for node in g.nodes:
        node.output[:] = [new if o == old else o for o in node.output]
        node.input[:] = [new if i == old else i for i in node.input]


def _dense_block_model(dtype, batch, nf=8, gc=4, spatial=16, seed=3):
    """ESRGAN-style residual dense block.

    x  = conv(input)
    c1 = relu(conv([x]))           c2 = relu(conv([x, c1]))
    c3 = relu(conv([x, c1, c2]))   c4 = relu(conv([x, c1, c2, c3]))
    y  = conv([x, c1, c2, c3, c4]) * 0.2 + x
    """
    g = _Graph(dtype, seed)
    x = g.conv("input", nf, nf)
    feats = [x]
    channels = nf
    for _ in range(4):
        src = feats[0] if len(feats) == 1 else g.op("Concat", feats, axis=1)
        feats.append(g.conv(src, channels, gc, relu=True))
        channels += gc
    cat = g.op("Concat", feats, axis=1)
    y = g.conv(cat, channels, nf)
    scaled = g.op("Mul", [y, g.const(np.array(0.2, dtype=dtype))])
    out = g.op("Add", [scaled, x])
    _rename_output(g, out, "output")
    shape = [batch, nf, spatial, spatial]
    return g.model([("input", shape)], [("output", shape)])


def _resize_slot_model(dtype, c=4, spatial=8, seed=5):
    """concat([resize(a), conv(b)]) read by a conv, with a skip reader of conv(b)."""
    g = _Graph(dtype, seed)
    up = g.op(
        "Resize",
        ["a", "", g.const(np.array([1, 1, 2, 2], dtype=np.float32))],
        mode="nearest",
    )
    side = g.conv("b", c, c)
    cat = g.op("Concat", [up, side], axis=1)
    y = g.conv(cat, 2 * c, c)
    skip = g.op("Add", [side, "b"])
    _rename_output(g, y, "output")
    _rename_output(g, skip, "skip")
    big = [1, c, 2 * spatial, 2 * spatial]
    return g.model(
        [("a", [1, c, spatial, spatial]), ("b", big)],
        [("output", big), ("skip", big)],
    )


def _concat_output_model(dtype, c=4, spatial=16, seed=9):
    """The concat itself is a graph output, and one input is read again later."""
    g = _Graph(dtype, seed)
    left = g.conv("input", c, c, relu=True)
    right = g.conv("input", c, c)
    cat = g.op("Concat", [left, right], axis=1)
    tail = g.conv(left, c, c)
    _rename_output(g, cat, "concat")
    _rename_output(g, tail, "tail")
    shape = [1, c, spatial, spatial]
    return g.model(
        [("input", shape)],
        [("concat", [1, 2 * c, spatial, spatial]), ("tail", shape)],
    )


class TestConcatSlots:
    @pytest.mark.parametrize("dtype", _DTYPES, ids=["fp32", "fp16"])
    @pytest.mark.parametrize("batch", [1, 2], ids=["packed", "strided"])
    def test_dense_block(self, model_runner, dtype, batch):
        """Batch 1 shares one buffer across the block; batch 2 keeps copies."""
        model = _dense_block_model(dtype, batch)
        x = np.random.default_rng(11).uniform(-1, 1, [batch, 8, 16, 16]).astype(dtype)
        actual, expected = model_runner.run_sample(model, [x], reference="cpu")
        tol = _TOL[dtype]
        compare_outputs(actual, expected, atol=tol, rtol=tol, cos_threshold=0.9999)

    @pytest.mark.parametrize("dtype", _DTYPES, ids=["fp32", "fp16"])
    def test_resize_into_slot(self, model_runner, dtype):
        model = _resize_slot_model(dtype)
        rng = np.random.default_rng(13)
        a = rng.uniform(-1, 1, [1, 4, 8, 8]).astype(dtype)
        b = rng.uniform(-1, 1, [1, 4, 16, 16]).astype(dtype)
        actual, expected = model_runner.run_sample(model, [a, b], reference="cpu")
        tol = _TOL[dtype]
        compare_outputs(actual, expected, atol=tol, rtol=tol, cos_threshold=0.9999)

    @pytest.mark.parametrize("dtype", _DTYPES, ids=["fp32", "fp16"])
    def test_concat_is_output(self, model_runner, dtype):
        model = _concat_output_model(dtype)
        x = np.random.default_rng(17).uniform(-1, 1, [1, 4, 16, 16]).astype(dtype)
        actual, expected = model_runner.run_sample(model, [x], reference="cpu")
        tol = _TOL[dtype]
        compare_outputs(actual, expected, atol=tol, rtol=tol, cos_threshold=0.9999)
