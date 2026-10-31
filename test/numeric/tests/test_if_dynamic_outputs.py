#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""If outputs whose dynamic sizes differ per branch, as in merged decoders:
the cache branch returns past + new rows and a zero-row dummy, the no-cache
branch computes both outputs from the new rows only."""

from __future__ import annotations

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes

_F32 = TensorProto.FLOAT
_WIDTH = 8


def _cache_branch():
    dummy = numpy_helper.from_array(np.zeros((0, _WIDTH), np.float32), "dummy_v")
    return helper.make_graph(
        [
            helper.make_node("Concat", ["past", "x"], ["present"], axis=0),
            helper.make_node("Constant", [], ["side"], value=dummy),
        ],
        "cache_branch",
        [],
        [
            helper.make_tensor_value_info("present", _F32, ["T", _WIDTH]),
            helper.make_tensor_value_info("side", _F32, [0, _WIDTH]),
        ],
    )


def _no_cache_branch():
    return helper.make_graph(
        [
            helper.make_node("Relu", ["x"], ["present"]),
            helper.make_node("Neg", ["x"], ["side"]),
        ],
        "no_cache_branch",
        [],
        [
            helper.make_tensor_value_info("present", _F32, ["S", _WIDTH]),
            helper.make_tensor_value_info("side", _F32, ["S", _WIDTH]),
        ],
    )


def _model():
    x = helper.make_tensor_value_info("x", _F32, ["S", _WIDTH])
    past = helper.make_tensor_value_info("past", _F32, ["P", _WIDTH])
    cond = helper.make_tensor_value_info("use_cache", TensorProto.BOOL, [1])
    present = helper.make_tensor_value_info("present", _F32, ["T", _WIDTH])
    side = helper.make_tensor_value_info("side", _F32, ["U", _WIDTH])
    node = helper.make_node(
        "If",
        ["use_cache"],
        ["present", "side"],
        then_branch=_cache_branch(),
        else_branch=_no_cache_branch(),
    )
    return make_model_from_nodes([node], [x, past, cond], [present, side])


class TestIfDynamicOutputs:
    @pytest.mark.parametrize(
        "use_cache,seq,past_len", [(True, 1, 4), (True, 3, 0), (False, 5, 0)]
    )
    def test_branch_sizes(self, model_runner, use_cache, seq, past_len):
        rng = np.random.default_rng(seq * 10 + past_len)
        x = rng.standard_normal((seq, _WIDTH)).astype(np.float32)
        past = rng.standard_normal((past_len, _WIDTH)).astype(np.float32)
        actual, expected = model_runner.run_sample(
            _model(), [x, past, np.array([use_cache])]
        )
        compare_outputs(actual, expected)
