#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Tests for the RotaryEmbedding custom op (com.microsoft).

RotaryEmbedding applies rotary position encoding to the input tensor.
The cos/sin caches are stored as ONNX initializers.

Dimensions match the Llama-3.1-8B ONNX model:
  hidden = 4096, kv_hidden = 1024, head_dim = 128
  cos/sin cache shape: [131072, 64]  (max_pos, head_dim // 2)
  num_heads = 0 (inferred from hidden / head_dim at runtime)
  interleaved = 0, rotary_embedding_dim = 0
"""

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes

SEQ_LENS = [1, 128]
HIDDEN = 4096
KV_HIDDEN = 1024
HEAD_DIM = 128
MAX_POS = 131072


def _make_rotary_embedding_model(
    batch,
    seq_len,
    hidden,
    head_dim,
    max_pos,
):
    """Build a RotaryEmbedding ONNX model matching the Llama-3.1-8B pattern.

    Uses num_heads=0 so ORT infers num_heads from hidden / head_dim.
    """
    half_dim = head_dim // 2

    X = helper.make_tensor_value_info(
        "X", TensorProto.FLOAT16, [batch, seq_len, hidden]
    )
    pos_ids = helper.make_tensor_value_info(
        "pos_ids", TensorProto.INT64, [batch, seq_len]
    )
    Y = helper.make_tensor_value_info(
        "Y", TensorProto.FLOAT16, [batch, seq_len, hidden]
    )

    rng = np.random.default_rng(99)
    cos_data = rng.uniform(-1, 1, [max_pos, half_dim]).astype(np.float16)
    sin_data = rng.uniform(-1, 1, [max_pos, half_dim]).astype(np.float16)
    cos_init = numpy_helper.from_array(cos_data, name="cos_cache")
    sin_init = numpy_helper.from_array(sin_data, name="sin_cache")

    node = helper.make_node(
        "RotaryEmbedding",
        ["X", "pos_ids", "cos_cache", "sin_cache"],
        ["Y"],
        domain="com.microsoft",
        interleaved=0,
        num_heads=0,
        rotary_embedding_dim=0,
    )
    ms_opset = helper.make_opsetid("com.microsoft", 1)
    model = make_model_from_nodes(
        [node],
        [X, pos_ids],
        [Y],
        initializers=[cos_init, sin_init],
        extra_opsets=[ms_opset],
    )
    return model


class TestRotaryEmbedding:
    @pytest.mark.parametrize(
        "hidden,head_dim",
        [
            (64, 16),
            (128, 32),
        ],
    )
    def test_rotary_embedding(self, model_runner, hidden, head_dim):
        """RotaryEmbedding with small shapes for fast sanity checking."""
        max_pos = 256
        seq_len = 8
        model = _make_rotary_embedding_model(1, seq_len, hidden, head_dim, max_pos)

        rng = np.random.default_rng(42)
        x = rng.uniform(-1, 1, [1, seq_len, hidden]).astype(np.float16)
        pos_ids = np.arange(seq_len, dtype=np.int64).reshape(1, seq_len)

        actual, expected = model_runner.run_sample(model, [x, pos_ids])
        compare_outputs(actual, expected, atol=1e-3)

    @pytest.mark.parametrize("seq_len", SEQ_LENS)
    def test_rotary_embedding_q_proj_llama_shape(self, model_runner, seq_len):
        """RotaryEmbedding on Q projection: [1, S, 4096], head_dim=128."""
        model = _make_rotary_embedding_model(
            1,
            seq_len,
            HIDDEN,
            HEAD_DIM,
            MAX_POS,
        )

        rng = np.random.default_rng(42)
        x = rng.uniform(-1, 1, [1, seq_len, HIDDEN]).astype(np.float16)
        pos_ids = np.arange(seq_len, dtype=np.int64).reshape(1, seq_len)

        actual, expected = model_runner.run_sample(model, [x, pos_ids])
        compare_outputs(actual, expected, atol=1e-3)

    @pytest.mark.parametrize("seq_len", SEQ_LENS)
    def test_rotary_embedding_k_proj_llama_shape(self, model_runner, seq_len):
        """RotaryEmbedding on K projection: [1, S, 1024], head_dim=128."""
        model = _make_rotary_embedding_model(
            1,
            seq_len,
            KV_HIDDEN,
            HEAD_DIM,
            MAX_POS,
        )

        rng = np.random.default_rng(42)
        x = rng.uniform(-1, 1, [1, seq_len, KV_HIDDEN]).astype(np.float16)
        pos_ids = np.arange(seq_len, dtype=np.int64).reshape(1, seq_len)

        actual, expected = model_runner.run_sample(model, [x, pos_ids])
        compare_outputs(actual, expected, atol=1e-3)

    @pytest.mark.parametrize(
        "hidden,offset",
        [
            (HIDDEN, 128),
            (KV_HIDDEN, 128),
            (HIDDEN, 255),
            (KV_HIDDEN, 255),
        ],
    )
    def test_rotary_embedding_decode_offset_llama_shape(
        self,
        model_runner,
        hidden,
        offset,
    ):
        """RotaryEmbedding single-token decode with non-zero position offset.

        During autoregressive decode, position_ids is a single offset value
        (e.g. 128 after a 128-token prefill, or 255 near cache end).
        """
        seq_len = 1
        model = _make_rotary_embedding_model(
            1,
            seq_len,
            hidden,
            HEAD_DIM,
            MAX_POS,
        )

        rng = np.random.default_rng(42)
        x = rng.uniform(-1, 1, [1, seq_len, hidden]).astype(np.float16)
        pos_ids = np.array([[offset]], dtype=np.int64)

        actual, expected = model_runner.run_sample(model, [x, pos_ids])
        compare_outputs(actual, expected, atol=1e-3)


def _make_native_rope_precomputed_model(
    batch,
    seq_len,
    hidden,
    num_heads,
    head_dim,
):
    """Build a native ai.onnx RotaryEmbedding model (opset 23) WITHOUT
    position_ids.

    This is the HF-export form used by Gemma: cos/sin are precomputed and
    position-expanded to 3D [batch, seq, head_dim // 2], so the op has only 3
    inputs (X, cos_cache, sin_cache) and no position_ids. num_heads is required
    for a 3D input per the ONNX spec.
    """
    half_dim = head_dim // 2

    X = helper.make_tensor_value_info(
        "X", TensorProto.FLOAT16, [batch, seq_len, hidden]
    )
    Y = helper.make_tensor_value_info(
        "Y", TensorProto.FLOAT16, [batch, seq_len, hidden]
    )

    rng = np.random.default_rng(123)
    cos_data = rng.uniform(-1, 1, [batch, seq_len, half_dim]).astype(np.float16)
    sin_data = rng.uniform(-1, 1, [batch, seq_len, half_dim]).astype(np.float16)
    cos_init = numpy_helper.from_array(cos_data, name="cos_cache")
    sin_init = numpy_helper.from_array(sin_data, name="sin_cache")

    node = helper.make_node(
        "RotaryEmbedding",
        ["X", "cos_cache", "sin_cache"],
        ["Y"],
        domain="",
        interleaved=0,
        num_heads=num_heads,
        rotary_embedding_dim=0,
    )
    model = make_model_from_nodes(
        [node],
        [X],
        [Y],
        initializers=[cos_init, sin_init],
        opset=23,
    )
    return model


class TestNativeRotaryEmbeddingNoPosIds:
    """Native ai.onnx RotaryEmbedding (opset 23) with precomputed 3D cos/sin
    and no position_ids -- the Gemma export form. Exercises the hip.rope path
    where the runtime indexes cos/sin by the flat token position b*seq+s.
    """

    @pytest.mark.parametrize(
        "hidden,num_heads,head_dim",
        [
            (128, 4, 32),
            (4096, 16, 256),
        ],
    )
    def test_native_rope_no_posids(
        self,
        model_runner,
        hidden,
        num_heads,
        head_dim,
    ):
        seq_len = 8
        model = _make_native_rope_precomputed_model(
            1, seq_len, hidden, num_heads, head_dim
        )

        rng = np.random.default_rng(42)
        x = rng.uniform(-1, 1, [1, seq_len, hidden]).astype(np.float16)

        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=1e-3)
