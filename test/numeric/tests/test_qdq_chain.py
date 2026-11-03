#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Per-tensor UINT16 Q/DQ chains fused by hip-fusion-transform's second phase.

The quantized SwiGLU (Q(gate), Sigmoid, two Mul, Q(up), each wrapped in
Q/DQ) runs as hip.qswiglu and a lone Q->DQ round trip as hip.fake_quantize.
Run with HIPDNN_EP_STRICT=1 so a compilation failure cannot fall back to CPU.
"""

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

from framework.onnx_utils import make_model_from_nodes

_OPSET = 21


def _qparams(name, scale, zp):
    return [
        numpy_helper.from_array(np.array(scale, dtype=np.float32), f"{name}_s"),
        numpy_helper.from_array(np.array(zp, dtype=np.uint16), f"{name}_z"),
    ]


def _qdq(nodes, src, name):
    nodes.append(
        helper.make_node(
            "QuantizeLinear", [src, f"{name}_s", f"{name}_z"], [f"{name}_q"]
        )
    )
    nodes.append(
        helper.make_node(
            "DequantizeLinear", [f"{name}_q", f"{name}_s", f"{name}_z"], [f"{name}_dq"]
        )
    )
    return f"{name}_dq"


# Parameters in the range a 16-bit QDQ export of a decoder MLP produces.
_PARAMS = {
    "gate": (9.887e-5, 40798),
    "sig": (1.5259e-5, 0),
    "silu": (3.8593e-5, 7215),
    "up": (6.5923e-5, 33492),
    "out": (7.0061e-5, 33430),
}


def _make_qswiglu_model(shape):
    G = helper.make_tensor_value_info("G", TensorProto.FLOAT, shape)
    U = helper.make_tensor_value_info("U", TensorProto.FLOAT, shape)
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, shape)
    inits = []
    for name, (s, z) in _PARAMS.items():
        inits += _qparams(name, s, z)
    nodes = []
    g = _qdq(nodes, "G", "gate")
    nodes.append(helper.make_node("Sigmoid", [g], ["sig_f"]))
    s = _qdq(nodes, "sig_f", "sig")
    nodes.append(helper.make_node("Mul", [g, s], ["silu_f"]))
    a = _qdq(nodes, "silu_f", "silu")
    u = _qdq(nodes, "U", "up")
    nodes.append(helper.make_node("Mul", [a, u], ["out_f"]))
    nodes.append(
        helper.make_node("QuantizeLinear", ["out_f", "out_s", "out_z"], ["out_q"])
    )
    nodes.append(
        helper.make_node("DequantizeLinear", ["out_q", "out_s", "out_z"], ["Y"])
    )
    return make_model_from_nodes(nodes, [G, U], [Y], initializers=inits, opset=_OPSET)


def _assert_codes_match(actual, expected, step, max_mismatch_fraction):
    # Both sides snap onto the same output grid, so any difference is a whole
    # number of quantization steps. The CPU's sigmoid and quantize can round a
    # value next to a code boundary the other way (about 0.1% of elements),
    # which moves it by a step or two; anything else means the fused
    # arithmetic diverged.
    diff = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    steps = np.rint(diff / step)
    assert steps.max() <= 2, f"max diff {steps.max()} output steps"
    assert np.mean(steps > 0) <= max_mismatch_fraction, (
        f"{np.mean(steps > 0):.2e} of elements differ"
    )


@pytest.mark.parametrize(
    "model_shape, run_shape",
    [
        (["batch", "seq", 256], [1, 1, 256]),
        (["batch", "seq", 256], [1, 37, 256]),
        # Not a multiple of four: the scalar kernel path.
        ([3, 5, 7], [3, 5, 7]),
    ],
)
def test_qswiglu(model_runner, model_shape, run_shape):
    model = _make_qswiglu_model(model_shape)
    rng = np.random.default_rng(7)
    g = rng.uniform(-4.0, 2.4, run_shape).astype(np.float32)
    u = rng.uniform(-2.2, 2.2, run_shape).astype(np.float32)
    actual, expected = model_runner.run_sample(model, [g, u])
    _assert_codes_match(actual[0], expected[0], _PARAMS["out"][0], 5e-3)


@pytest.mark.parametrize(
    "run_shape, scale, zp",
    [
        ([2, 33, 128], 1.25e-3, 32768),
        # Values far outside the grid saturate at both ends.
        ([4, 17], 1.0e-4, 100),
    ],
)
def test_fake_quantize(model_runner, run_shape, scale, zp):
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, run_shape)
    nodes = []
    out = _qdq(nodes, "X", "x")
    Y = helper.make_tensor_value_info(out, TensorProto.FLOAT, run_shape)
    model = make_model_from_nodes(
        nodes, [X], [Y], initializers=_qparams("x", scale, zp), opset=_OPSET
    )
    rng = np.random.default_rng(11)
    x = rng.uniform(-50.0, 50.0, run_shape).astype(np.float32)
    actual, expected = model_runner.run_sample(model, [x])
    np.testing.assert_array_equal(actual[0], expected[0])
