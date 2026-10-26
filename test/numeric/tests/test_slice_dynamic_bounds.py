#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Numeric checks for onnx.Slice whose bounds are runtime shape arithmetic."""

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes

_ROWS, _WIDTH, _VOCAB = 448, 64, 32


def _i64(name, value):
    return numpy_helper.from_array(np.array(value, dtype=np.int64), name)


def _position_slice_nodes():
    """Whisper decoder_with_past: embed_positions[past : past + len]."""
    return [
        helper.make_node("Shape", ["past"], ["past_shape"]),
        helper.make_node("Gather", ["past_shape", "k2"], ["past_len"], axis=0),
        helper.make_node("Unsqueeze", ["past_len", "axis0"], ["starts"]),
        helper.make_node("Shape", ["ids"], ["ids_shape"]),
        helper.make_node("Gather", ["ids_shape", "k1"], ["len"], axis=0),
        helper.make_node("Add", ["past_len", "len"], ["end"]),
        helper.make_node("Unsqueeze", ["end", "axis0"], ["ends"]),
        helper.make_node(
            "Slice", ["table", "starts", "ends", "axis0", "one"], ["pos"]
        ),
    ]


def _initializers(rng):
    table = rng.standard_normal((_ROWS, _WIDTH)).astype(np.float32)
    return [
        numpy_helper.from_array(table, "table"),
        _i64("k1", 1),
        _i64("k2", 2),
        _i64("axis0", [0]),
        _i64("one", [1]),
    ]


def _inputs(batch, seq, past_len, rng):
    ids = rng.integers(0, _VOCAB, size=(batch, seq), dtype=np.int64)
    past = rng.standard_normal((batch, 2, past_len, 8)).astype(np.float32)
    return [ids, past]


_IDS = helper.make_tensor_value_info("ids", TensorProto.INT64, ["B", "S"])
_PAST = helper.make_tensor_value_info(
    "past", TensorProto.FLOAT, ["B", 2, "P", 8]
)


class TestSliceDynamicBounds:
    @pytest.mark.parametrize("seq,past_len", [(1, 4), (3, 0), (2, 446)])
    def test_position_embedding_slice(self, model_runner, seq, past_len):
        rng = np.random.default_rng(0)
        y = helper.make_tensor_value_info("pos", TensorProto.FLOAT, ["L", _WIDTH])
        model = make_model_from_nodes(
            _position_slice_nodes(), [_IDS, _PAST], [y],
            initializers=_initializers(rng),
        )
        actual, expected = model_runner.run_sample(
            model, _inputs(1, seq, past_len, rng)
        )
        compare_outputs(actual, expected)

    def test_position_embedding_slice_static_len(self, model_runner):
        """decoder_with_past types input_ids as [B, 1], so Shape(ids)[1] folds
        to a constant that the GPU adds to the past length."""
        rng = np.random.default_rng(2)
        ids = helper.make_tensor_value_info("ids", TensorProto.INT64, ["B", 1])
        y = helper.make_tensor_value_info("pos", TensorProto.FLOAT, ["L", _WIDTH])
        model = make_model_from_nodes(
            _position_slice_nodes(), [ids, _PAST], [y],
            initializers=_initializers(rng),
        )
        actual, expected = model_runner.run_sample(
            model, _inputs(1, 1, 4, rng)
        )
        compare_outputs(actual, expected)

    def test_token_plus_position_embedding(self, model_runner):
        """Gather(embed_tokens, ids) + embed_positions slice, as in the
        decoder_with_past embedding prologue."""
        rng = np.random.default_rng(1)
        tokens = rng.standard_normal((_VOCAB, _WIDTH)).astype(np.float32)
        nodes = _position_slice_nodes() + [
            helper.make_node("Gather", ["tokens", "ids"], ["tok"], axis=0),
            helper.make_node("Add", ["tok", "pos"], ["Y"]),
        ]
        y = helper.make_tensor_value_info(
            "Y", TensorProto.FLOAT, ["B", "S", _WIDTH]
        )
        model = make_model_from_nodes(
            nodes, [_IDS, _PAST], [y],
            initializers=_initializers(rng)
            + [numpy_helper.from_array(tokens, "tokens")],
        )
        actual, expected = model_runner.run_sample(
            model, _inputs(1, 1, 4, rng)
        )
        compare_outputs(actual, expected)
