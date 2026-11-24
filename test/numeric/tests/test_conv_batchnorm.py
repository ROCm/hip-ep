#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Tests for BatchNormalization folded into the producing Conv weights.

hip-fold-into-conv rewrites an inference BatchNormalization with constant
parameters into the filter and bias of the Conv(s) feeding it, directly or
through an Add of two Convs. A wrong per-channel scale, a shift added twice,
or a shift dropped all give outputs that stay correlated with the reference,
so these compare tightly against the ORT CPU reference.

The fold reads only inline constants, and every filter and BN parameter vector
here (64 channels, 256 B in f32) is above the importer's default 127-byte
inline threshold. Run with
`--ep-option XLNX_model_clone_external_data_threshold=1073741824` (or the
`XLNX_ABI_2_0_CLONE_EXTERNAL_DATA_THRESHOLD` environment variable) to exercise
the fold; without it the same graphs check the unfolded path.

TestConvChannelAffine runs a Conv followed by a Mul by a per-channel constant
(a layer scale), an Add of one, both in a row, or a BatchNormalization, with a
dynamic batch. It checks the result against the reference. ONNX Runtime
applies its own Conv + Mul / Add / BatchNormalization fusions before the EP
sees the graph, so these cases do not show that hip-fold-into-conv ran: the
LIT tests of the pass (test/lit/Transforms/test_fold_into_conv.mlir) cover the
fold itself, including the cases that must not fold.

test_decomposed_bn_graph_inputs covers a BatchNormalization exported as
elementwise ops (Sub, Mul, Mul, Add over Unsqueeze'd per-channel vectors and
an Add/Sqrt/Reciprocal chain for the inverse standard deviation), which the
fusion pass turns back into one batch norm. Its parameters are graph inputs
because ORT constant-folds the same chain away when they are initializers. It
also checks the compiler's IR dump (HIPDNN_EP_IR_DUMP_PATH), because a result
that matches the reference does not show that the chain was fused.
"""

import re

import numpy as np
import pytest
from onnx import helper, numpy_helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes, np_to_onnx_type

CHANNELS = 64
SPATIAL = 8


def _bn_initializers(rng, dtype, prefix="bn"):
    """BatchNorm scale, B, mean, var with a spread wide enough to matter."""
    return [
        numpy_helper.from_array(
            rng.uniform(0.5, 2.0, [CHANNELS]).astype(dtype), f"{prefix}_scale"
        ),
        numpy_helper.from_array(
            rng.uniform(-1.0, 1.0, [CHANNELS]).astype(dtype), f"{prefix}_b"
        ),
        numpy_helper.from_array(
            rng.uniform(-0.5, 0.5, [CHANNELS]).astype(dtype), f"{prefix}_mean"
        ),
        numpy_helper.from_array(
            rng.uniform(0.1, 2.0, [CHANNELS]).astype(dtype), f"{prefix}_var"
        ),
    ]


def _bn_node(x, y, prefix="bn"):
    return helper.make_node(
        "BatchNormalization",
        [x, f"{prefix}_scale", f"{prefix}_b", f"{prefix}_mean", f"{prefix}_var"],
        [y],
        epsilon=1e-5,
    )


def _conv_node(x, w, y, bias=None, group=1):
    return helper.make_node(
        "Conv",
        [x, w] + ([bias] if bias else []),
        [y],
        kernel_shape=[3, 3],
        strides=[1, 1],
        pads=[1, 1, 1, 1],
        dilations=[1, 1],
        group=group,
    )


def _weight(rng, dtype, name, in_c, group=1):
    w = rng.uniform(-0.2, 0.2, [CHANNELS, in_c // group, 3, 3]).astype(dtype)
    return numpy_helper.from_array(w, name)


_TOLERANCE = {
    np.float32: dict(atol=1e-3, rtol=1e-3, cos_threshold=0.99999),
    np.float16: dict(atol=3e-2, rtol=2e-2, cos_threshold=0.999),
}


@pytest.fixture
def ir_dump(tmp_path, monkeypatch):
    """Directory that receives the compiler's IR dump for this test.

    The EP reads HIPDNN_EP_IR_DUMP_PATH on every compilation and writes one
    `ir.<n>.mlir` file per compilation, holding the IR around each pass.
    """
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


def _assert_batch_norm_fused(dump_dir):
    """The elementwise chain became hip.batch_norm and left nothing behind."""
    texts = _ir_after_fusion(dump_dir)
    assert texts, f"no IR dump after hip-fusion-transform in {dump_dir}"
    for text in texts:
        assert "hip.batch_norm" in text, "chain was not fused into hip.batch_norm"
        left = set(re.findall(r"hip\.(?:sub|sqrt|reciprocal)\b", text))
        assert not left, f"elementwise chain ops left after fusion: {sorted(left)}"


class TestConvBatchNorm:
    @pytest.mark.parametrize("dtype", [np.float32, np.float16])
    @pytest.mark.parametrize("with_bias", [False, True])
    @pytest.mark.parametrize("group", [1, 4])
    def test_conv_bn(self, model_runner, dtype, with_bias, group):
        """Conv -> BatchNormalization folds into one conv with a bias."""
        in_c = 16
        tp = np_to_onnx_type(dtype)
        rng = np.random.default_rng(11)
        inits = [_weight(rng, dtype, "w", in_c, group)] + _bn_initializers(rng, dtype)
        if with_bias:
            inits.append(
                numpy_helper.from_array(
                    rng.uniform(-0.5, 0.5, [CHANNELS]).astype(dtype), "cb"
                )
            )
        nodes = [
            _conv_node("x", "w", "c", bias="cb" if with_bias else None, group=group),
            _bn_node("c", "y"),
        ]
        model = make_model_from_nodes(
            nodes,
            [helper.make_tensor_value_info("x", tp, [1, in_c, SPATIAL, SPATIAL])],
            [helper.make_tensor_value_info("y", tp, [1, CHANNELS, SPATIAL, SPATIAL])],
            initializers=inits,
        )
        x = rng.uniform(-1.0, 1.0, [1, in_c, SPATIAL, SPATIAL]).astype(dtype)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, **_TOLERANCE[dtype])

    @pytest.mark.parametrize("dtype", [np.float32, np.float16])
    def test_add_conv_conv_bn_relu(self, model_runner, dtype):
        """Relu(BN(Conv(x) + Conv(z))): both filters scaled, shift added once."""
        in_x, in_z = 16, 48
        tp = np_to_onnx_type(dtype)
        rng = np.random.default_rng(23)
        inits = [
            _weight(rng, dtype, "wa", in_x),
            _weight(rng, dtype, "wb", in_z),
        ] + _bn_initializers(rng, dtype)
        nodes = [
            _conv_node("x", "wa", "ca"),
            _conv_node("z", "wb", "cb"),
            helper.make_node("Add", ["ca", "cb"], ["s"]),
            _bn_node("s", "n"),
            helper.make_node("Relu", ["n"], ["y"]),
        ]
        model = make_model_from_nodes(
            nodes,
            [
                helper.make_tensor_value_info("x", tp, [1, in_x, SPATIAL, SPATIAL]),
                helper.make_tensor_value_info("z", tp, [1, in_z, SPATIAL, SPATIAL]),
            ],
            [helper.make_tensor_value_info("y", tp, [1, CHANNELS, SPATIAL, SPATIAL])],
            initializers=inits,
        )
        x = rng.uniform(-1.0, 1.0, [1, in_x, SPATIAL, SPATIAL]).astype(dtype)
        z = rng.uniform(-1.0, 1.0, [1, in_z, SPATIAL, SPATIAL]).astype(dtype)
        actual, expected = model_runner.run_sample(model, [x, z])
        compare_outputs(actual, expected, **_TOLERANCE[dtype])

    @pytest.mark.parametrize("dtype", [np.float32, np.float16])
    @pytest.mark.parametrize("affine", [True, False])
    def test_decomposed_bn_graph_inputs(self, model_runner, ir_dump, dtype, affine):
        """Conv -> elementwise batch norm with per-channel parameters as inputs.

        y = (c - mean) * reciprocal(sqrt(var + eps)) * scale + bias, each vector
        unsqueezed to [C, 1, 1]; without `affine` there is no scale and no bias.
        A swapped operand order or a misplaced epsilon gives outputs that stay
        correlated with the reference, so compare tightly. The compiler's IR must
        also show the chain fused into hip.batch_norm: a CPU fallback or an
        unfused chain would pass the comparison just the same.
        """
        in_c = 16
        tp = np_to_onnx_type(dtype)
        rng = np.random.default_rng(37)
        inits = [
            _weight(rng, dtype, "w", in_c),
            numpy_helper.from_array(np.array(1e-5, dtype=dtype), "eps"),
            numpy_helper.from_array(np.array([1, 2], dtype=np.int64), "axes"),
        ]
        nodes = [
            _conv_node("x", "w", "c"),
            helper.make_node("Add", ["var", "eps"], ["var_eps"]),
            helper.make_node("Sqrt", ["var_eps"], ["std"]),
            helper.make_node("Reciprocal", ["std"], ["inv_std"]),
        ]
        parameters = ["mean", "var"] + (["scale", "bias"] if affine else [])
        for name in ["mean", "inv_std"] + (["scale", "bias"] if affine else []):
            nodes.append(helper.make_node("Unsqueeze", [name, "axes"], [f"{name}_v"]))
        nodes += [
            helper.make_node("Sub", ["c", "mean_v"], ["centered"]),
            helper.make_node(
                "Mul", ["centered", "inv_std_v"], ["normalized" if affine else "y"]
            ),
        ]
        if affine:
            nodes += [
                helper.make_node("Mul", ["normalized", "scale_v"], ["scaled"]),
                helper.make_node("Add", ["scaled", "bias_v"], ["y"]),
            ]
        model = make_model_from_nodes(
            nodes,
            [helper.make_tensor_value_info("x", tp, [1, in_c, SPATIAL, SPATIAL])]
            + [helper.make_tensor_value_info(n, tp, [CHANNELS]) for n in parameters],
            [helper.make_tensor_value_info("y", tp, [1, CHANNELS, SPATIAL, SPATIAL])],
            initializers=inits,
        )
        x = rng.uniform(-1.0, 1.0, [1, in_c, SPATIAL, SPATIAL]).astype(dtype)
        ranges = {
            "mean": (-0.5, 0.5),
            "var": (0.1, 2.0),
            "scale": (0.5, 2.0),
            "bias": (-1.0, 1.0),
        }
        vectors = [
            rng.uniform(*ranges[n], [CHANNELS]).astype(dtype) for n in parameters
        ]
        actual, expected = model_runner.run_sample(model, [x] + vectors)
        compare_outputs(actual, expected, **_TOLERANCE[dtype])
        _assert_batch_norm_fused(ir_dump)


# Each case is the chain after the Conv: (op, constant shape) pairs.
_AFFINE_CHAINS = {
    "bn": None,
    "mul_chw": [("Mul", [CHANNELS, 1, 1])],
    "mul_1chw": [("Mul", [1, CHANNELS, 1, 1])],
    "mul_scalar": [("Mul", [])],
    "add_1chw": [("Add", [1, CHANNELS, 1, 1])],
    "mul_add": [("Mul", [CHANNELS, 1, 1]), ("Add", [1, CHANNELS, 1, 1])],
}


class TestConvChannelAffine:
    @pytest.mark.parametrize("dtype", [np.float32, np.float16])
    @pytest.mark.parametrize("batch", [1, 3])
    @pytest.mark.parametrize("chain", list(_AFFINE_CHAINS))
    def test_conv_then_affine_dynamic_batch(self, model_runner, dtype, batch, chain):
        """Conv -> per-channel affine map, with a dynamic batch."""
        in_c = 16
        tp = np_to_onnx_type(dtype)
        rng = np.random.default_rng(21)
        inits = [
            _weight(rng, dtype, "w", in_c),
            numpy_helper.from_array(
                rng.uniform(-0.5, 0.5, [CHANNELS]).astype(dtype), "cb"
            ),
        ]
        nodes = [_conv_node("x", "w", "c0", bias="cb")]
        if _AFFINE_CHAINS[chain] is None:
            inits += _bn_initializers(rng, dtype)
            nodes.append(_bn_node("c0", "y"))
        else:
            last = "c0"
            for i, (op, shape) in enumerate(_AFFINE_CHAINS[chain]):
                const = rng.uniform(0.5, 1.5 if op == "Mul" else 1.0, shape)
                inits.append(numpy_helper.from_array(const.astype(dtype), f"k{i}"))
                out = "y" if i == len(_AFFINE_CHAINS[chain]) - 1 else f"c{i + 1}"
                nodes.append(helper.make_node(op, [last, f"k{i}"], [out]))
                last = out
        model = make_model_from_nodes(
            nodes,
            [helper.make_tensor_value_info("x", tp, ["batch", in_c, SPATIAL, SPATIAL])],
            [
                helper.make_tensor_value_info(
                    "y", tp, ["batch", CHANNELS, SPATIAL, SPATIAL]
                )
            ],
            initializers=inits,
        )
        x = rng.uniform(-1.0, 1.0, [batch, in_c, SPATIAL, SPATIAL]).astype(dtype)
        actual, expected = model_runner.run_sample(model, [x])
        assert actual[0].shape == (batch, CHANNELS, SPATIAL, SPATIAL)
        compare_outputs(actual, expected, **_TOLERANCE[dtype])
