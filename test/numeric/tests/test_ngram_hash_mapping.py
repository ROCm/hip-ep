#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Tests for the NGramHashMapping custom op (com.microsoft, Engram).

Inputs: input_ids [B, S], multipliers [>=N], vocab_sizes [H], past_ids [B, L]
(optional), head_offsets [H] (optional), eos_token_id scalar (optional),
segment_ids [B, S] int32 (optional). Outputs: hash_ids [B, S, H],
present_ids [B, L] (optional). L = N - 1, H = L * n_head_per_ngram.
"""

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

from framework.onnx_utils import make_model_from_nodes

_NP_TO_TP = {np.int32: TensorProto.INT32, np.int64: TensorProto.INT64}
DTYPES = [np.int32, np.int64]

# Qwen3.8-Flash-Next engram.onnx (model.ple.*).
QWEN38_MULTIPLIERS = [23703573157769, 20109073645365, 8052911324071]
QWEN38_VOCAB = [
    20000003,
    20000023,
    20000033,
    20000047,
    20000059,
    20000063,
    20000069,
    20000077,
    20000081,
    20000093,
    20000107,
    20000147,
    20000153,
    20000159,
    20000161,
    20000171,
]
QWEN38_OFFSETS = [
    0,
    20000003,
    40000026,
    60000059,
    80000106,
    100000165,
    120000228,
    140000297,
    160000374,
    180000455,
    200000548,
    220000655,
    240000802,
    260000955,
    280001114,
    300001275,
]
QWEN38_EOS = 248044


def _make_model(
    dtype,
    batch,
    seq_len,
    max_ngram_size,
    n_head_per_ngram,
    multipliers,
    vocab_sizes,
    pad_id,
    *,
    with_past=False,
    head_offsets=None,
    eos_token_id=None,
    reset_on_eos=0,
    with_segment_ids=False,
    with_present=True,
):
    tp = _NP_TO_TP[dtype]
    hist = max_ngram_size - 1
    heads = hist * n_head_per_ngram

    inputs = [helper.make_tensor_value_info("input_ids", tp, [batch, seq_len])]
    initializers = [
        numpy_helper.from_array(np.array(multipliers, dtype=dtype), "multipliers"),
        numpy_helper.from_array(np.array(vocab_sizes, dtype=dtype), "vocab_sizes"),
    ]
    node_inputs = ["input_ids", "multipliers", "vocab_sizes"]

    if with_past:
        inputs.append(helper.make_tensor_value_info("past_ids", tp, [batch, hist]))
        node_inputs.append("past_ids")
    else:
        node_inputs.append("")
    if head_offsets is not None:
        initializers.append(
            numpy_helper.from_array(np.array(head_offsets, dtype=dtype), "head_offsets")
        )
        node_inputs.append("head_offsets")
    else:
        node_inputs.append("")
    if eos_token_id is not None:
        initializers.append(
            numpy_helper.from_array(np.array(eos_token_id, dtype=dtype), "eos_token_id")
        )
        node_inputs.append("eos_token_id")
    else:
        node_inputs.append("")
    if with_segment_ids:
        inputs.append(
            helper.make_tensor_value_info(
                "segment_ids", TensorProto.INT32, [batch, seq_len]
            )
        )
        node_inputs.append("segment_ids")
    while node_inputs[-1] == "":
        node_inputs.pop()

    outputs = [helper.make_tensor_value_info("hash_ids", tp, [batch, seq_len, heads])]
    node_outputs = ["hash_ids"]
    if with_present:
        outputs.append(helper.make_tensor_value_info("present_ids", tp, [batch, hist]))
        node_outputs.append("present_ids")

    node = helper.make_node(
        "NGramHashMapping",
        node_inputs,
        node_outputs,
        domain="com.microsoft",
        max_ngram_size=max_ngram_size,
        n_head_per_ngram=n_head_per_ngram,
        pad_id=pad_id,
        reset_on_eos=reset_on_eos,
    )
    return make_model_from_nodes(
        [node],
        inputs,
        outputs,
        initializers=initializers,
        opset=17,
        extra_opsets=[helper.make_opsetid("com.microsoft", 1)],
    )


def _assert_equal(actual, expected):
    assert len(actual) == len(expected)
    for act, exp in zip(actual, expected):
        np.testing.assert_array_equal(act, exp)


class TestNGramHashMapping:
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_basic_no_past(self, model_runner, dtype):
        """Missing history reads as pad_id; present_ids comes from input."""
        model = _make_model(dtype, 1, 4, 3, 2, [11, 13, 17], [101, 103, 107, 109], 9)
        ids = np.array([[3, 4, 5, 6]], dtype=dtype)
        actual, expected = model_runner.run_sample(model, [ids])
        _assert_equal(actual, expected)

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_wrapping_multiply(self, model_runner, dtype):
        """Products overflow T and must wrap in the unsigned type."""
        big = 1 << 40 if dtype == np.int64 else 1 << 20
        rng = np.random.default_rng(7)
        mult = [big + 3, big * 3 + 1, big * 5 + 7]
        model = _make_model(
            dtype,
            2,
            8,
            3,
            4,
            mult,
            [1009 + 2 * i for i in range(8)],
            0,
            with_past=True,
        )
        hi = 1 << 30 if dtype == np.int64 else 1 << 16
        ids = rng.integers(0, hi, size=(2, 8)).astype(dtype)
        past = rng.integers(0, hi, size=(2, 2)).astype(dtype)
        actual, expected = model_runner.run_sample(model, [ids, past])
        _assert_equal(actual, expected)

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_head_offsets_with_past(self, model_runner, dtype):
        model = _make_model(
            dtype,
            1,
            4,
            3,
            2,
            [11, 13, 17],
            [101, 103, 107, 109],
            9,
            with_past=True,
            head_offsets=[1000, 2000, 3000, 4000],
        )
        ids = np.array([[3, 4, 5, 6]], dtype=dtype)
        past = np.array([[1, 2]], dtype=dtype)
        actual, expected = model_runner.run_sample(model, [ids, past])
        _assert_equal(actual, expected)

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_eos_reset(self, model_runner, dtype):
        """EOS inside the input resets the window; fill value is eos."""
        model = _make_model(
            dtype,
            1,
            6,
            3,
            1,
            [11, 13, 17],
            [101, 103],
            0,
            with_past=True,
            eos_token_id=9,
            reset_on_eos=1,
        )
        ids = np.array([[3, 9, 5, 6, 9, 7]], dtype=dtype)
        past = np.array([[2, 4]], dtype=dtype)
        actual, expected = model_runner.run_sample(model, [ids, past])
        _assert_equal(actual, expected)

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_eos_interior_past(self, model_runner, dtype):
        """An EOS in the middle of past_ids (N=4) is a boundary too."""
        model = _make_model(
            dtype,
            1,
            3,
            4,
            2,
            [11, 13, 17, 19],
            [101, 103, 107, 109, 113, 127],
            0,
            with_past=True,
            eos_token_id=9,
            reset_on_eos=1,
        )
        ids = np.array([[5, 6, 7]], dtype=dtype)
        past = np.array([[3, 9, 4]], dtype=dtype)
        actual, expected = model_runner.run_sample(model, [ids, past])
        _assert_equal(actual, expected)

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_eos_without_reset(self, model_runner, dtype):
        """eos_token_id without reset_on_eos only changes the fill value."""
        model = _make_model(
            dtype,
            1,
            4,
            3,
            2,
            [11, 13, 17],
            [101, 103, 107, 109],
            0,
            eos_token_id=9,
        )
        ids = np.array([[3, 9, 5, 6]], dtype=dtype)
        actual, expected = model_runner.run_sample(model, [ids])
        _assert_equal(actual, expected)

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_segment_ids(self, model_runner, dtype):
        model = _make_model(
            dtype,
            1,
            4,
            2,
            2,
            [11, 13],
            [101, 103],
            3,
            with_past=True,
            with_segment_ids=True,
        )
        ids = np.array([[3, 4, 5, 6]], dtype=dtype)
        past = np.array([[8]], dtype=dtype)
        seg = np.array([[0, 0, 1, 1]], dtype=np.int32)
        actual, expected = model_runner.run_sample(model, [ids, past, seg])
        _assert_equal(actual, expected)

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_negative_ids(self, model_runner, dtype):
        """Negative mixes exercise the PositiveMod correction."""
        model = _make_model(
            dtype,
            1,
            4,
            3,
            2,
            [11, 13, 17],
            [101, 103, 107, 109],
            -4,
        )
        ids = np.array([[-3, 4, -5, 6]], dtype=dtype)
        actual, expected = model_runner.run_sample(model, [ids])
        _assert_equal(actual, expected)

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_batched(self, model_runner, dtype):
        model = _make_model(
            dtype,
            3,
            5,
            3,
            2,
            [11, 13, 17],
            [101, 103, 107, 109],
            9,
            with_past=True,
            head_offsets=[0, 101, 204, 311],
            eos_token_id=1,
            reset_on_eos=1,
        )
        rng = np.random.default_rng(3)
        ids = rng.integers(0, 6, size=(3, 5)).astype(dtype)
        past = rng.integers(0, 6, size=(3, 2)).astype(dtype)
        actual, expected = model_runner.run_sample(model, [ids, past])
        _assert_equal(actual, expected)

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_hash_only(self, model_runner, dtype):
        """No present_ids output: the op has a single DPS output."""
        model = _make_model(
            dtype,
            2,
            4,
            3,
            2,
            [11, 13, 17],
            [101, 103, 107, 109],
            9,
            with_past=True,
            with_present=False,
        )
        ids = np.array([[3, 4, 5, 6], [7, 8, 9, 10]], dtype=dtype)
        past = np.array([[1, 2], [5, 6]], dtype=dtype)
        actual, expected = model_runner.run_sample(model, [ids, past])
        _assert_equal(actual, expected)

    def test_qwen38_engram_prefill_decode(self, model_runner):
        """Qwen3.8 engram configuration: 16-token prefill, then two decode
        steps feeding present_ids back as past_ids; the chunked hashes must
        equal a single pass over the whole sequence."""
        kwargs = dict(
            with_past=True,
            head_offsets=QWEN38_OFFSETS,
            eos_token_id=QWEN38_EOS,
            reset_on_eos=1,
        )
        args = (3, 8, QWEN38_MULTIPLIERS, QWEN38_VOCAB, QWEN38_EOS)
        rng = np.random.default_rng(0)
        ids = rng.integers(0, 248320, size=(1, 18), dtype=np.int64)
        ids[0, 9] = QWEN38_EOS
        past = np.full((1, 2), QWEN38_EOS, dtype=np.int64)

        full_model = _make_model(np.int64, 1, 18, *args, **kwargs)
        full, full_ref = model_runner.run_sample(
            full_model, [ids, past], name="qwen38_full"
        )
        _assert_equal(full, full_ref)

        prefill_model = _make_model(np.int64, 1, 16, *args, **kwargs)
        hashes = []
        (hash0, present), ref = model_runner.run_sample(
            prefill_model, [ids[:, :16], past], name="qwen38_prefill"
        )
        _assert_equal([hash0, present], ref)
        hashes.append(hash0)

        decode_model = _make_model(np.int64, 1, 1, *args, **kwargs)
        for step in (16, 17):
            (h, present), ref = model_runner.run_sample(
                decode_model,
                [ids[:, step : step + 1], present],
                name=f"qwen38_decode_{step}",
                reference="cpu",
            )
            _assert_equal([h, present], ref)
            hashes.append(h)

        np.testing.assert_array_equal(np.concatenate(hashes, axis=1), full[0])
        np.testing.assert_array_equal(present, full[1])
