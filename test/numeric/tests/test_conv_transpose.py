#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Numeric tests for ONNX ConvTranspose against the ORT CPU reference.

ConvTranspose is wired end to end -- ``onnx.ConvTranspose`` to
``hip.conv_transpose`` to ``wrap_conv_transpose`` -- but no model the EP
supports contains one, so nothing exercised its arithmetic: the only coverage
was LIT, which checks that the call is emitted, not that the result is right.

That gap matters because every attribute here changes the index mapping rather
than just the loop bounds. Transposed convolution scatters each input element
across a strided, dilated footprint, so stride, dilation and the two pad ends
each move where a given input lands, and ``output_padding`` exists only to
disambiguate output sizes that stride alone leaves ambiguous. A test per
attribute is what pins that mapping.

The lowering requires rank-4 throughout: input ``[N, C, H, W]``, weight
``[C, M/group, kH, kW]`` (input channels first, unlike forward Conv), output
``[N, M, H', W']``.
"""

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes

_DTYPES = {
    TensorProto.FLOAT16: np.float16,
    TensorProto.FLOAT: np.float32,
}


def _out_dim(in_dim, k, stride, pad_begin, pad_end, dilation, out_pad):
    """The ONNX ConvTranspose output extent for one spatial axis."""
    return (
        stride * (in_dim - 1) + out_pad + ((k - 1) * dilation + 1) - pad_begin - pad_end
    )


def _conv_transpose_model(
    *,
    n=1,
    c=1,
    hw=(3, 3),
    m_per_group=2,
    k=(3, 3),
    strides=(1, 1),
    pads=(0, 0, 0, 0),
    dilations=(1, 1),
    output_padding=(0, 0),
    group=1,
    with_bias=True,
    dtype=TensorProto.FLOAT,
    seed=11,
):
    """A single ConvTranspose node with weight (and optional bias) folded in.

    `m_per_group` is the weight's second dim, so the total output channel count
    is `m_per_group * group`.
    """
    np_dtype = _DTYPES[dtype]
    m = m_per_group * group
    out_h = _out_dim(
        hw[0], k[0], strides[0], pads[0], pads[2], dilations[0], output_padding[0]
    )
    out_w = _out_dim(
        hw[1], k[1], strides[1], pads[1], pads[3], dilations[1], output_padding[1]
    )

    X = helper.make_tensor_value_info("x", dtype, [n, c, hw[0], hw[1]])
    Y = helper.make_tensor_value_info("y", dtype, [n, m, out_h, out_w])

    rng = np.random.default_rng(seed)
    w = (rng.standard_normal((c, m_per_group, k[0], k[1])) * 0.3).astype(np_dtype)
    initializers = [numpy_helper.from_array(w, name="w")]
    node_inputs = ["x", "w"]
    if with_bias:
        b = (rng.standard_normal((m,)) * 0.3).astype(np_dtype)
        initializers.append(numpy_helper.from_array(b, name="b"))
        node_inputs.append("b")

    node = helper.make_node(
        "ConvTranspose",
        node_inputs,
        ["y"],
        kernel_shape=list(k),
        strides=list(strides),
        pads=list(pads),
        dilations=list(dilations),
        output_padding=list(output_padding),
        group=group,
    )
    model = make_model_from_nodes([node], [X], [Y], initializers=initializers, opset=17)
    return model, (n, c, hw[0], hw[1]), np_dtype


def _check(model_runner, model, in_shape, np_dtype, seed):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(in_shape).astype(np_dtype)
    actual, expected = model_runner.run_sample(model, [x])
    if np_dtype == np.float32:
        compare_outputs(actual, expected, atol=1e-4, rtol=1e-4)
    else:
        compare_outputs(actual, expected, atol=5e-3, rtol=1e-2, cos_threshold=0.9999)


class TestConvTranspose:
    """One case per attribute that moves the input-to-output index mapping."""

    def test_basic(self, model_runner):
        """stride 1, no padding: the plain full correlation."""
        model, shape, dt = _conv_transpose_model()
        _check(model_runner, model, shape, dt, seed=1)

    def test_stride(self, model_runner):
        """stride 2 spreads each input element two apart in the output, which
        leaves cells only some kernel taps ever reach."""
        model, shape, dt = _conv_transpose_model(strides=(2, 2))
        _check(model_runner, model, shape, dt, seed=2)

    def test_asymmetric_pads(self, model_runner):
        """Padding on ConvTranspose *crops* the output, and the two ends crop
        independently, so an implementation that only reads pad_begin still
        produces the right shape with the wrong contents."""
        model, shape, dt = _conv_transpose_model(
            hw=(5, 5), strides=(2, 2), pads=(1, 0, 0, 1)
        )
        _check(model_runner, model, shape, dt, seed=3)

    def test_dilation(self, model_runner):
        """Dilation spaces the taps out without spacing the inputs out, which
        is a different stride on the kernel axis than on the input axis."""
        model, shape, dt = _conv_transpose_model(hw=(4, 4), dilations=(2, 2))
        _check(model_runner, model, shape, dt, seed=4)

    def test_group(self, model_runner):
        """group 2 over C=4: each half of the input channels feeds only its own
        half of the output channels."""
        model, shape, dt = _conv_transpose_model(c=4, hw=(4, 4), group=2)
        _check(model_runner, model, shape, dt, seed=5)

    def test_output_padding(self, model_runner):
        """With stride > 1 several input sizes map onto overlapping output
        ranges; output_padding is what picks between them, extending the output
        with cells no input reaches."""
        model, shape, dt = _conv_transpose_model(strides=(2, 2), output_padding=(1, 1))
        _check(model_runner, model, shape, dt, seed=6)

    def test_no_bias(self, model_runner):
        """The bias operand is optional; absent it the runtime gets a null."""
        model, shape, dt = _conv_transpose_model(with_bias=False)
        _check(model_runner, model, shape, dt, seed=7)

    def test_non_square_kernel_and_stride(self, model_runner):
        """Different extents per axis, so an H/W transposition shows up."""
        model, shape, dt = _conv_transpose_model(
            hw=(5, 3), k=(3, 5), strides=(2, 1), pads=(1, 2, 1, 2)
        )
        _check(model_runner, model, shape, dt, seed=8)

    def test_batch_and_channels(self, model_runner):
        """A decoder-shaped upsample: N>1 with real channel counts, so the
        per-(n, m) output addressing is exercised rather than assumed."""
        model, shape, dt = _conv_transpose_model(
            n=2,
            c=16,
            hw=(8, 8),
            m_per_group=8,
            k=(4, 4),
            strides=(2, 2),
            pads=(1, 1, 1, 1),
        )
        _check(model_runner, model, shape, dt, seed=9)

    @pytest.mark.parametrize("strides", [(1, 1), (2, 2)])
    def test_fp16(self, model_runner, strides):
        """fp16 in and out, accumulating in fp32."""
        model, shape, dt = _conv_transpose_model(
            c=8, hw=(6, 6), m_per_group=8, strides=strides, dtype=TensorProto.FLOAT16
        )
        _check(model_runner, model, shape, dt, seed=10)
