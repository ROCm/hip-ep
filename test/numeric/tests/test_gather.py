#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Tests for the Gather operation.

Embedding dimensions:
  Llama-3.1-8B:  vocab = 128256, hidden = 4096
  GPT-OSS-20B:   vocab = 201088, hidden = 2880
"""

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes

SEQ_LENS = [1, 128]

LLAMA_VOCAB = 128256
LLAMA_HIDDEN = 4096

GPT_OSS_VOCAB = 201088
GPT_OSS_HIDDEN = 2880


def _make_gather_model(data_shape: list[int], axis: int = 0):
    """Build a Gather ONNX model with i64 data and scalar i64 index."""
    data = helper.make_tensor_value_info("data", TensorProto.INT64, data_shape)
    indices = helper.make_tensor_value_info("indices", TensorProto.INT64, [])
    output_shape = data_shape[:axis] + data_shape[axis + 1 :]
    output = helper.make_tensor_value_info(
        "output", TensorProto.INT64, output_shape if output_shape else []
    )
    node = helper.make_node("Gather", ["data", "indices"], ["output"], axis=axis)
    return make_model_from_nodes([node], [data, indices], [output])


def _make_embedding_gather_model(
    vocab_size: int,
    hidden: int,
    seq_len: int,
    embed_range: tuple[float, float] = (-0.1, 0.1),
    seed: int = 99,
):
    """Build a Gather ONNX model matching embedding lookup pattern.

    data = initializer [vocab_size, hidden] f16 (embedding table)
    indices = input [1, seq_len] i64 (token IDs)
    output = [1, seq_len, hidden] f16
    """
    indices = helper.make_tensor_value_info(
        "input_ids",
        TensorProto.INT64,
        [1, seq_len],
    )
    output = helper.make_tensor_value_info(
        "output",
        TensorProto.FLOAT16,
        [1, seq_len, hidden],
    )

    rng = np.random.default_rng(seed)
    embed_data = rng.uniform(
        embed_range[0],
        embed_range[1],
        [vocab_size, hidden],
    ).astype(np.float16)
    embed_init = numpy_helper.from_array(embed_data, name="embed_weight")

    node = helper.make_node(
        "Gather",
        ["embed_weight", "input_ids"],
        ["output"],
        axis=0,
    )
    return make_model_from_nodes(
        [node],
        [indices],
        [output],
        initializers=[embed_init],
    )


class TestGather:
    @pytest.mark.parametrize(
        "data_shape",
        [
            [2],
            [8],
        ],
    )
    def test_gather_axis0_scalar_index(self, model_runner, data_shape):
        model = _make_gather_model(data_shape, axis=0)

        rng = np.random.default_rng(42)
        data = rng.integers(-100, 100, data_shape, dtype=np.int64)
        index = np.array(rng.integers(0, data_shape[0]), dtype=np.int64)

        actual, expected = model_runner.run_sample(model, [data, index])
        compare_outputs(actual, expected, atol=0)

    @pytest.mark.parametrize("seq_len", SEQ_LENS)
    def test_gather_embedding_llama_shape(self, model_runner, seq_len):
        """Embedding lookup: [128256, 4096] f16 gather by [1, S] i64."""
        model = _make_embedding_gather_model(LLAMA_VOCAB, LLAMA_HIDDEN, seq_len)

        rng = np.random.default_rng(42)
        ids = rng.integers(0, LLAMA_VOCAB, [1, seq_len], dtype=np.int64)

        actual, expected = model_runner.run_sample(model, [ids])
        compare_outputs(actual, expected, atol=0)

    @pytest.mark.parametrize("seq_len", SEQ_LENS)
    def test_gather_embedding_gpt_oss_shape(self, model_runner, seq_len):
        """Embedding lookup: [201088, 2880] f16 gather by [1, S] i64.

        Real embedding weights span [-104, 118] (much wider than Llama).
        """
        model = _make_embedding_gather_model(
            GPT_OSS_VOCAB,
            GPT_OSS_HIDDEN,
            seq_len,
            embed_range=(-100.0, 100.0),
        )

        rng = np.random.default_rng(42)
        ids = rng.integers(0, GPT_OSS_VOCAB, [1, seq_len], dtype=np.int64)

        actual, expected = model_runner.run_sample(model, [ids])
        compare_outputs(actual, expected, atol=0)
