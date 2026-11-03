#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Numeric tests for Whisper's decoder attention as onnxruntime-genai exports it.

The onnxruntime-genai export differs from the surgery form the other Whisper
tests use (see test_whisper_self_attention.py, test_whisper_cross_attention.py):

* self-attention is the 8-input ``com.microsoft.MultiHeadAttention`` with
  ``past_key``/``past_value`` ``[B, N, P, d]`` and no ``past_sequence_length``;
  present is unshared, ``concat(past, this step)`` = ``[B, N, P + S, d]``;
* cross-attention K/V ``[B, N, Skv, d]`` leave the batch dynamic.

MultiHeadAttentionConversion.cpp lowers both to ``hip.gqa``: the self-attention
with ``seqlens_k = P + S - 1`` and causal masking, the cross-attention with
``seqlens_k = Skv`` splatted to the batch once at function entry. Batch,
sequence and past lengths are dynamic in these models, as in the export.

Reference: ORT CPU runs the same models (MultiHeadAttention with an unshared
past is a standard ORT form), comparing the output and both present buffers.
"""

import numpy as np
import pytest
from onnx import TensorProto, helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes

HEAD_DIM = 64
SCALE = 1.0 / np.sqrt(HEAD_DIM)
MS_OPSET = [helper.make_opsetid("com.microsoft", 1)]


def _bsh(name, hidden):
    return helper.make_tensor_value_info(
        name, TensorProto.FLOAT16, ["batch", "seq", hidden]
    )


def _bnsh(name, num_heads, seq):
    return helper.make_tensor_value_info(
        name, TensorProto.FLOAT16, ["batch", num_heads, seq, HEAD_DIM]
    )


def _self_mha(layer, q, k, v, num_heads):
    """Unshared self-attention with past: outputs out_<l>, present_{key,value}_<l>."""
    return helper.make_node(
        "MultiHeadAttention",
        [q, k, v, "", "", "", f"past_key_{layer}", f"past_value_{layer}"],
        [f"out_{layer}", f"present_key_{layer}", f"present_value_{layer}"],
        domain="com.microsoft",
        num_heads=num_heads,
        scale=SCALE,
        unidirectional=1,
    )


def _cross_mha(layer, q, num_heads):
    """Cross-attention on encoder K/V cross_{key,value}_<l>: output cross_<l>.

    The 8-input form with the five optional slots empty, as the export writes it.
    """
    return helper.make_node(
        "MultiHeadAttention",
        [q, f"cross_key_{layer}", f"cross_value_{layer}", "", "", "", "", ""],
        [f"cross_{layer}"],
        domain="com.microsoft",
        num_heads=num_heads,
        scale=SCALE,
        unidirectional=0,
    )


def _rand(rng, shape):
    return rng.uniform(-1.0, 1.0, shape).astype(np.float16)


class TestWhisperOgaAttention:
    """onnxruntime-genai Whisper decoder attention -> hip.gqa."""

    @pytest.mark.parametrize("num_heads", [6, 20])
    @pytest.mark.parametrize(
        "batch,past,seq",
        [(1, 0, 4), (1, 1, 1), (1, 17, 1), (1, 64, 1), (1, 15, 4), (2, 9, 1)],
        ids=lambda v: str(v),
    )
    def test_unshared_self_attention(self, model_runner, batch, past, seq, num_heads):
        """Self-attention with past; present = concat(past, this step).

        past 0 is the first decoder step (the prompt tokens), seq 1 a decode
        step, seq 4 with past a chunk after the cache already holds tokens.
        """
        hidden = num_heads * HEAD_DIM
        model = make_model_from_nodes(
            [_self_mha(0, "query", "key", "value", num_heads)],
            [
                _bsh("query", hidden),
                _bsh("key", hidden),
                _bsh("value", hidden),
                _bnsh("past_key_0", num_heads, "past"),
                _bnsh("past_value_0", num_heads, "past"),
            ],
            [
                _bsh("out_0", hidden),
                _bnsh("present_key_0", num_heads, "total"),
                _bnsh("present_value_0", num_heads, "total"),
            ],
            extra_opsets=MS_OPSET,
        )
        rng = np.random.default_rng(past * 131 + seq * 7 + num_heads)
        inputs = [_rand(rng, [batch, seq, hidden]) for _ in range(3)]
        inputs += [_rand(rng, [batch, num_heads, past, HEAD_DIM]) for _ in range(2)]

        actual, expected = model_runner.run_sample(model, inputs)
        assert actual[1].shape == (batch, num_heads, past + seq, HEAD_DIM)
        compare_outputs(actual, expected, atol=2e-2, rtol=2e-2, cos_threshold=0.999)

    @pytest.mark.parametrize("batch", [1, 2])
    def test_decoder_layers(self, model_runner, batch):
        """Two decoder layers: self, cross, self, cross (whisper-tiny geometry).

        Cross-attention K/V have a dynamic batch and share one seqlens_k; the
        self-attention layers carry their own. Run twice: the output must be
        the same both times (a seqlens_k buffer reused at the same address
        used to make decoding differ from run to run).
        """
        num_heads, skv, past, seq = 6, 1500, 12, 1
        hidden = num_heads * HEAD_DIM
        nodes = [
            _self_mha(0, "query", "key", "value", num_heads),
            _cross_mha(0, "out_0", num_heads),
            _self_mha(1, "cross_0", "cross_0", "cross_0", num_heads),
            _cross_mha(1, "out_1", num_heads),
        ]
        inputs_info = [
            _bsh("query", hidden),
            _bsh("key", hidden),
            _bsh("value", hidden),
        ]
        outputs_info = []
        for layer in range(2):
            inputs_info += [
                _bnsh(f"past_key_{layer}", num_heads, "past"),
                _bnsh(f"past_value_{layer}", num_heads, "past"),
                _bnsh(f"cross_key_{layer}", num_heads, skv),
                _bnsh(f"cross_value_{layer}", num_heads, skv),
            ]
            outputs_info += [
                _bnsh(f"present_key_{layer}", num_heads, "total"),
                _bnsh(f"present_value_{layer}", num_heads, "total"),
            ]
        outputs_info.append(_bsh("cross_1", hidden))
        model = make_model_from_nodes(
            nodes, inputs_info, outputs_info, extra_opsets=MS_OPSET
        )
        rng = np.random.default_rng(batch)
        inputs = [_rand(rng, [batch, seq, hidden]) for _ in range(3)]
        for _ in range(2):
            inputs += [_rand(rng, [batch, num_heads, past, HEAD_DIM]) for _ in range(2)]
            inputs += [_rand(rng, [batch, num_heads, skv, HEAD_DIM]) for _ in range(2)]

        first, expected = model_runner.run_sample(model, inputs)
        compare_outputs(first, expected, atol=2e-2, rtol=2e-2, cos_threshold=0.999)
        second, _ = model_runner.run_sample(model, inputs)
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)
