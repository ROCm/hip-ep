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


def _make_gather_model(
    data_shape: list[int], axis: int = 0, indices_shape: list[int] | None = None
):
    """Build a Gather ONNX model with i64 data and indices."""
    indices_shape = [] if indices_shape is None else indices_shape
    data = helper.make_tensor_value_info("data", TensorProto.INT64, data_shape)
    indices = helper.make_tensor_value_info("indices", TensorProto.INT64, indices_shape)
    output_shape = data_shape[:axis] + indices_shape + data_shape[axis + 1 :]
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
    idx_dtype: int = TensorProto.INT64,
):
    """Build a Gather ONNX model matching embedding lookup pattern.

    data = initializer [vocab_size, hidden] f16 (embedding table)
    indices = input [1, seq_len] (int32 or int64 token IDs)
    output = [1, seq_len, hidden] f16

    ONNX allows the indices tensor to be int32 or int64; transformer text
    models commonly feed an int32 input_ids straight into the embedding Gather,
    so both index widths must produce identical results.
    """
    indices = helper.make_tensor_value_info(
        "input_ids",
        idx_dtype,
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

    @pytest.mark.parametrize(
        "data_shape,axis,indices,output_shape",
        [
            ([3, 4], 0, np.empty([0], dtype=np.int64), (0, 4)),
            ([2, 0, 3], 2, np.array([0, 2], dtype=np.int64), (2, 0, 2)),
        ],
    )
    def test_gather_empty_output(
        self, model_runner, data_shape, axis, indices, output_shape
    ):
        """Empty indices and a zero non-gather extent return shaped empties."""
        model = _make_gather_model(data_shape, axis, list(indices.shape))
        data = np.arange(np.prod(data_shape), dtype=np.int64).reshape(data_shape)

        actual, expected = model_runner.run_sample(model, [data, indices])

        assert actual[0].shape == output_shape
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
    def test_gather_embedding_int32_indices(self, model_runner, seq_len):
        """Embedding lookup with int32 token IDs (ONNX allows int32 indices).

        Regression guard: the gather kernel previously read the indices buffer
        as int64 unconditionally, fusing adjacent int32 values into huge
        out-of-range indices that the bounds check mapped to 0 -> all-zero
        embeddings. int32 and int64 indices must match the CPU reference.
        """
        model = _make_embedding_gather_model(
            LLAMA_VOCAB,
            LLAMA_HIDDEN,
            seq_len,
            idx_dtype=TensorProto.INT32,
        )

        rng = np.random.default_rng(42)
        ids = rng.integers(0, LLAMA_VOCAB, [1, seq_len], dtype=np.int32)

        actual, expected = model_runner.run_sample(model, [ids])
        compare_outputs(actual, expected, atol=0)

    @pytest.mark.parametrize(
        "dtype,inner", [(np.float16, 32), (np.float32, 4), (np.float16, 12)]
    )
    def test_gather_middle_axis_negative_indices(self, model_runner, dtype, inner):
        data_shape = [3, 50, inner]
        tp = helper.np_dtype_to_tensor_dtype(np.dtype(dtype))
        data = helper.make_tensor_value_info("data", tp, data_shape)
        indices = helper.make_tensor_value_info("indices", TensorProto.INT64, [2, 5])
        output = helper.make_tensor_value_info("output", tp, [3, 2, 5, inner])
        node = helper.make_node("Gather", ["data", "indices"], ["output"], axis=1)
        model = make_model_from_nodes([node], [data, indices], [output])

        rng = np.random.default_rng(7)
        x = rng.uniform(-1, 1, data_shape).astype(dtype)
        ids = rng.integers(-50, 50, [2, 5], dtype=np.int64)
        actual, expected = model_runner.run_sample(model, [x, ids])
        compare_outputs(actual, expected, atol=0)

    @pytest.mark.parametrize(
        "dtype,inner",
        [
            # One element per row (1, 2, 4 and 8 bytes): one thread per row.
            (np.int8, 1),
            (np.float16, 1),
            (np.float32, 1),
            (np.int64, 1),
            # Rows of 2 to 4 chunks: 6 B and 12 B rows, one thread per row.
            (np.float16, 3),
            (np.float32, 3),
            (np.float16, 4),
            # Rows off every vector width: 1-byte and 2-byte chunks.
            (np.int8, 7),
            (np.float16, 147),
            (np.float16, 387),
            # Rows of whole 16-byte chunks, one thread per chunk.
            (np.float32, 64),
        ],
    )
    @pytest.mark.parametrize("idx_type", [np.int32, np.int64])
    def test_gather_row_sizes(self, model_runner, dtype, inner, idx_type):
        """Gather [outer, axis, inner] on axis 1 for every row-size path."""
        data_shape = [3, 40, inner]
        indices_shape = [2, 9]
        tp = helper.np_dtype_to_tensor_dtype(np.dtype(dtype))
        itp = helper.np_dtype_to_tensor_dtype(np.dtype(idx_type))
        data = helper.make_tensor_value_info("data", tp, data_shape)
        indices = helper.make_tensor_value_info("indices", itp, indices_shape)
        output = helper.make_tensor_value_info("output", tp, [3, 2, 9, inner])
        node = helper.make_node("Gather", ["data", "indices"], ["output"], axis=1)
        model = make_model_from_nodes([node], [data, indices], [output])

        rng = np.random.default_rng(11)
        if np.issubdtype(dtype, np.integer):
            x = rng.integers(-100, 100, data_shape).astype(dtype)
        else:
            x = rng.uniform(-1, 1, data_shape).astype(dtype)
        ids = rng.integers(-40, 40, indices_shape).astype(idx_type)
        actual, expected = model_runner.run_sample(model, [x, ids])
        compare_outputs(actual, expected, atol=0)

    @pytest.mark.parametrize("index", [0, 1, -1])
    def test_gather_scalar_index_last_axis(self, model_runner, index):
        """A scalar index into a size-2 innermost axis: 4-byte rows."""
        data_shape = [2, 3, 17, 19, 2]
        data = helper.make_tensor_value_info("data", TensorProto.FLOAT, data_shape)
        indices = helper.make_tensor_value_info("indices", TensorProto.INT64, [])
        output = helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, data_shape[:4]
        )
        node = helper.make_node("Gather", ["data", "indices"], ["output"], axis=4)
        model = make_model_from_nodes([node], [data, indices], [output])

        rng = np.random.default_rng(3)
        x = rng.uniform(-1, 1, data_shape).astype(np.float32)
        actual, expected = model_runner.run_sample(
            model, [x, np.array(index, dtype=np.int64)]
        )
        compare_outputs(actual, expected, atol=0)

    @pytest.mark.parametrize(
        "dtype,data_shape,axis,index",
        [
            # Scalar index into a size-2 innermost axis (a slice, not a gather).
            (np.float32, [2, 3, 33, 35, 2], 4, np.array(1, dtype=np.int64)),
            (np.float16, [2, 3, 33, 35, 2], -1, np.array(-2, dtype=np.int64)),
            # Middle axis, int32 index.
            (np.float32, [4, 6, 1024], 1, np.array(4, dtype=np.int32)),
            # One-element 1-D index keeps the axis.
            (np.float16, [5, 4096], 0, np.array([3], dtype=np.int64)),
            # Results under the slice threshold stay hip.gather.
            (np.float32, [2, 3, 17, 19, 2], 4, np.array(1, dtype=np.int64)),
            (np.float32, [9], 0, np.array(-1, dtype=np.int64)),
        ],
    )
    def test_gather_constant_index(self, model_runner, dtype, data_shape, axis, index):
        """A constant single index lowers to a slice of the gathered axis."""
        tp = helper.np_dtype_to_tensor_dtype(np.dtype(dtype))
        rank = len(data_shape)
        ax = axis % rank
        out_shape = data_shape[:ax] + list(index.shape) + data_shape[ax + 1 :]
        data = helper.make_tensor_value_info("data", tp, data_shape)
        output = helper.make_tensor_value_info("output", tp, out_shape)
        idx_init = numpy_helper.from_array(index, name="index")
        node = helper.make_node("Gather", ["data", "index"], ["output"], axis=axis)
        model = make_model_from_nodes([node], [data], [output], initializers=[idx_init])

        rng = np.random.default_rng(5)
        x = rng.uniform(-1, 1, data_shape).astype(dtype)
        actual, expected = model_runner.run_sample(model, [x])
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
