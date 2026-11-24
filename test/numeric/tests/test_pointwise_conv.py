#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Numeric tests for 1x1 Conv run as a hipBLASLt matmul, and for the
per-channel broadcast add that carries its bias.

hip-fusion-transform rewrites a float 1x1 Conv with stride 1, no padding,
dilation 1 and one group into hip.matmul of the [Cout, Cin] filter against
the [N, Cin, pixels] input -- one filter shared by every batch element, which
lowers to wrap_hipblasLtMatmulStrided once the batch exceeds one -- plus a
broadcast hip.add of the [1, Cout, 1] bias. The cases cover:

  * channel counts and pixel extents that are not powers of two (the bias add
    then takes the per-channel elementwise fast path), and pixel extents that
    are not a multiple of the vector width (the general broadcast kernel);
  * a dynamic batch, run at several sizes in one session;
  * f16 and f32, with and without a bias, rank 4 and rank 5, and bf16 behind
    f32 inputs and outputs (ORT Python cannot feed bf16 arrays);
  * convs the rewrite must leave alone (stride 2, a 3x3 kernel), as controls.

A result that matches the reference does not show that the conv became a
matmul, so the conv tests also check the compiler's IR dump
(HIPDNN_EP_IR_DUMP_PATH) after hip-fusion-transform.
"""

from __future__ import annotations

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes

_NP = {TensorProto.FLOAT16: np.float16, TensorProto.FLOAT: np.float32}
# Tolerances: the GEMM and the reference accumulate in different orders.
_TOL = {TensorProto.FLOAT16: (2e-2, 2e-2), TensorProto.FLOAT: (1e-4, 1e-4)}


def _make_conv_model(
    cin,
    cout,
    spatial,
    *,
    batch=1,
    bias=True,
    kernel=1,
    stride=1,
    dtype=TensorProto.FLOAT16,
    dynamic_batch=False,
):
    """Single-op Conv model, weight and bias as initializers."""
    np_dtype = _NP[dtype]
    nsp = len(spatial)
    pad = (kernel - 1) // 2
    out_spatial = [(s + 2 * pad - kernel) // stride + 1 for s in spatial]
    n = "N" if dynamic_batch else batch
    X = helper.make_tensor_value_info("x", dtype, [n, cin, *spatial])
    Y = helper.make_tensor_value_info("y", dtype, [n, cout, *out_spatial])
    rng = np.random.default_rng(cin * 131 + cout)
    w = (
        rng.standard_normal((cout, cin, *([kernel] * nsp))) / np.sqrt(cin * kernel**nsp)
    ).astype(np_dtype)
    inits = [numpy_helper.from_array(w, name="w")]
    ins = ["x", "w"]
    if bias:
        b = rng.standard_normal((cout,)).astype(np_dtype)
        inits.append(numpy_helper.from_array(b, name="b"))
        ins.append("b")
    node = helper.make_node(
        "Conv",
        ins,
        ["y"],
        kernel_shape=[kernel] * nsp,
        strides=[stride] * nsp,
        pads=[pad] * (2 * nsp),
        dilations=[1] * nsp,
        group=1,
    )
    return make_model_from_nodes([node], [X], [Y], initializers=inits, opset=17)


def _input(batch, cin, spatial, dtype, seed):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((batch, cin, *spatial)).astype(_NP[dtype])


def _check(actual, expected, dtype):
    atol, rtol = _TOL[dtype]
    compare_outputs(actual, expected, atol=atol, rtol=rtol)


@pytest.fixture
def ir_dump(tmp_path, monkeypatch):
    """Directory that receives the compiler's IR dump for this test."""
    monkeypatch.setenv("HIPDNN_EP_IR_DUMP_PATH", str(tmp_path / "ir.mlir"))
    return tmp_path


def _ir_after_fusion(dump_dir):
    """The IR printed after hip-fusion-transform, one text per compilation."""
    texts = []
    for dump in sorted(dump_dir.glob("ir.*.mlir")):
        for section in dump.read_text().split("// -----// IR Dump ")[1:]:
            if section.startswith("After HipFusionTransformPass"):
                texts.append(section)
    return texts


def _assert_rewritten(dump_dir):
    """Every compilation turned the conv into a matmul."""
    texts = _ir_after_fusion(dump_dir)
    assert texts, f"no IR dump after hip-fusion-transform in {dump_dir}"
    for text in texts:
        assert "hip.matmul" in text, "the conv was not rewritten to hip.matmul"
        assert "hip.conv" not in text, "a hip.conv is left after the rewrite"


def _assert_left_alone(dump_dir):
    """No compilation produced a matmul from the conv."""
    for text in _ir_after_fusion(dump_dir):
        assert "hip.matmul" not in text, "the conv was rewritten"


def _make_conv_bf16_models(cin, cout, spatial, *, batch=1, dynamic_batch=False):
    """A bf16 1x1 Conv behind f32 inputs and outputs, plus an f32 CPU reference.

    ORT CPU has no bf16 Conv, so the reference computes in f32 from the same
    bf16-rounded input, weight and bias and rounds the result to bf16. The
    weight and the bias are graph inputs: the ONNX constant import does not
    take a bf16 initializer.
    """
    f32, bf16 = TensorProto.FLOAT, TensorProto.BFLOAT16
    # Conv takes bf16 from opset 22.
    nsp = len(spatial)
    n = "N" if dynamic_batch else batch
    X = helper.make_tensor_value_info("x", f32, [n, cin, *spatial])
    Y = helper.make_tensor_value_info("y", f32, [n, cout, *spatial])
    W = helper.make_tensor_value_info("w", f32, [cout, cin, *([1] * nsp)])
    B = helper.make_tensor_value_info("b", f32, [cout])
    conv_attrs = dict(
        kernel_shape=[1] * nsp,
        strides=[1] * nsp,
        pads=[0] * (2 * nsp),
        dilations=[1] * nsp,
        group=1,
    )
    to_bf16 = [
        helper.make_node("Cast", [name], [name + "b"], to=bf16)
        for name in ("x", "w", "b")
    ]
    model = make_model_from_nodes(
        to_bf16
        + [
            helper.make_node("Conv", ["xb", "wb", "bb"], ["yb"], **conv_attrs),
            helper.make_node("Cast", ["yb"], ["y"], to=f32),
        ],
        [X, W, B],
        [Y],
        opset=22,
    )
    reference = make_model_from_nodes(
        to_bf16
        + [
            helper.make_node("Cast", ["xb"], ["x32"], to=f32),
            helper.make_node("Cast", ["wb"], ["w32"], to=f32),
            helper.make_node("Cast", ["bb"], ["b32"], to=f32),
            helper.make_node("Conv", ["x32", "w32", "b32"], ["y32"], **conv_attrs),
            helper.make_node("Cast", ["y32"], ["yr"], to=bf16),
            helper.make_node("Cast", ["yr"], ["y"], to=f32),
        ],
        [X, W, B],
        [Y],
        opset=22,
    )
    rng = np.random.default_rng(cin * 17 + cout)
    w = (rng.standard_normal((cout, cin, *([1] * nsp))) / np.sqrt(cin)).astype(
        np.float32
    )
    b = rng.standard_normal((cout,)).astype(np.float32)
    return model, reference, w, b


class TestPointwiseConvAsMatmul:
    @pytest.mark.parametrize("dtype", [TensorProto.FLOAT16, TensorProto.FLOAT])
    @pytest.mark.parametrize(
        "cin,cout,spatial",
        [
            (64, 256, [56, 56]),  # wide pixel axis, non-power-of-two
            (2048, 512, [7, 7]),  # long contraction, 49 pixels
            (1024, 256, [14, 14]),
            (24, 144, [56, 56]),  # narrow K
            (32, 48, [5, 3]),  # 15 pixels: not a whole vector
        ],
    )
    def test_static_bias(self, model_runner, ir_dump, dtype, cin, cout, spatial):
        model = _make_conv_model(cin, cout, spatial, dtype=dtype)
        x = _input(1, cin, spatial, dtype, seed=cin + cout)
        actual, expected = model_runner.run_sample(model, [x])
        _check(actual, expected, dtype)
        _assert_rewritten(ir_dump)

    @pytest.mark.parametrize("dtype", [TensorProto.FLOAT16, TensorProto.FLOAT])
    def test_static_batch_no_bias(self, model_runner, ir_dump, dtype):
        model = _make_conv_model(96, 40, [28, 28], batch=3, bias=False, dtype=dtype)
        x = _input(3, 96, [28, 28], dtype, seed=7)
        actual, expected = model_runner.run_sample(model, [x])
        _check(actual, expected, dtype)
        _assert_rewritten(ir_dump)

    @pytest.mark.parametrize("dtype", [TensorProto.FLOAT16, TensorProto.FLOAT])
    def test_dynamic_batch(self, model_runner, ir_dump, dtype):
        """One session, several batch sizes: the shared filter must be
        applied to every image (batch stride 0 for the filter, Cin*P for the
        input), and a batch of one must still take the plain path."""
        model = _make_conv_model(256, 64, [14, 14], dtype=dtype, dynamic_batch=True)
        for batch in (2, 1, 4, 8):
            x = _input(batch, 256, [14, 14], dtype, seed=batch)
            actual, expected = model_runner.run_sample(
                model, [x], name=f"pw_dyn_b{batch}"
            )
            _check(actual, expected, dtype)
        _assert_rewritten(ir_dump)

    def test_rank5_bias(self, model_runner, ir_dump):
        """NCDHW: three spatial axes collapse into the pixel axis; the bias add
        stays rank 3."""
        model = _make_conv_model(32, 48, [4, 6, 6], batch=2)
        x = _input(2, 32, [4, 6, 6], TensorProto.FLOAT16, seed=5)
        actual, expected = model_runner.run_sample(model, [x])
        _check(actual, expected, TensorProto.FLOAT16)
        _assert_rewritten(ir_dump)

    @pytest.mark.parametrize("kernel,stride", [(1, 2), (3, 1)])
    def test_not_rewritten_controls(self, model_runner, ir_dump, kernel, stride):
        """Stride 2 and a 3x3 kernel stay on hip.conv; same tolerances."""
        model = _make_conv_model(
            64, 128, [28, 28], batch=2, kernel=kernel, stride=stride
        )
        x = _input(2, 64, [28, 28], TensorProto.FLOAT16, seed=11)
        actual, expected = model_runner.run_sample(model, [x])
        _check(actual, expected, TensorProto.FLOAT16)
        _assert_left_alone(ir_dump)

    @pytest.mark.parametrize(
        "cin,cout,spatial",
        [(64, 256, [56, 56]), (2048, 512, [7, 7]), (32, 48, [5, 3])],
    )
    def test_bf16_static_bias(self, model_runner, ir_dump, cin, cout, spatial):
        model, reference, w, b = _make_conv_bf16_models(cin, cout, spatial, batch=2)
        x = _input(2, cin, spatial, TensorProto.FLOAT, seed=cin)
        actual, expected = model_runner.run_sample(
            model, [x, w, b], reference="cpu", reference_model=reference
        )
        compare_outputs(actual, expected, atol=5e-2, rtol=2e-2)
        _assert_rewritten(ir_dump)

    def test_bf16_dynamic_batch(self, model_runner, ir_dump):
        model, reference, w, b = _make_conv_bf16_models(
            256, 64, [14, 14], dynamic_batch=True
        )
        for batch in (2, 1, 4):
            x = _input(batch, 256, [14, 14], TensorProto.FLOAT, seed=batch)
            actual, expected = model_runner.run_sample(
                model,
                [x, w, b],
                reference="cpu",
                reference_model=reference,
                name=f"pw_bf16_dyn_b{batch}",
            )
            compare_outputs(actual, expected, atol=5e-2, rtol=2e-2)
        _assert_rewritten(ir_dump)


class TestPerChannelBroadcastAdd:
    """Add of a [1, C, 1, 1] (or [C, 1, 1]) tensor to [N, C, H, W]: the bias add
    the rewrite emits, also on its own, in either operand order."""

    @pytest.mark.parametrize("dtype", [TensorProto.FLOAT16, TensorProto.FLOAT])
    @pytest.mark.parametrize(
        "shape,bshape,swap",
        [
            ([2, 24, 28, 28], [1, 24, 1, 1], False),
            ([2, 24, 28, 28], [24, 1, 1], True),
            ([1, 1000, 7, 7], [1, 1000, 1, 1], False),  # 49: not a whole vector
            ([3, 40, 3136], [1, 40, 1], False),
        ],
    )
    def test_channel_add(self, model_runner, dtype, shape, bshape, swap):
        np_dtype = _NP[dtype]
        A = helper.make_tensor_value_info("a", dtype, shape)
        B = helper.make_tensor_value_info("b", dtype, bshape)
        Y = helper.make_tensor_value_info("y", dtype, shape)
        ins = ["b", "a"] if swap else ["a", "b"]
        node = helper.make_node("Add", ins, ["y"])
        model = make_model_from_nodes([node], [A, B], [Y])
        rng = np.random.default_rng(len(shape) * 1000 + shape[1])
        a = rng.standard_normal(shape).astype(np_dtype)
        b = rng.standard_normal(bshape).astype(np_dtype)
        actual, expected = model_runner.run_sample(model, [a, b])
        compare_outputs(actual, expected, atol=1e-3, rtol=1e-3)
