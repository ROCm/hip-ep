#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Tests for inject_seqlens_k ONNX surgery helper.

Validates that ``conftest.inject_seqlens_k`` correctly rewrites the 32
self-attention ``MultiHeadAttention`` nodes in a Whisper decoder to take a
``past_sequence_length`` graph input (threaded into slot 8), while leaving the
32 cross-attention nodes untouched. The surgery deliberately adds NO
``past_present_share_buffer`` attribute: ORT's MHA schema rejects it, so the
slot-8 input is the sole shared-buffer signal (see ``inject_seqlens_k`` doc).
"""

from __future__ import annotations

import pathlib
import sys

import onnx
import pytest
from onnx import TensorProto, helper

# conftest.py (shared with the llama tests) lives one level up in test/python/.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from conftest import REPO_ROOT, inject_seqlens_k  # noqa: E402


# Real Whisper decoder lives here; auto-skip if not downloaded yet.
WHISPER_DECODER = REPO_ROOT / "models" / "whisper-large-v3-onnx" / "decoder.onnx"


def _make_synthetic_decoder() -> onnx.ModelProto:
    """Build a tiny synthetic 'decoder' with one self-attn + one cross-attn MHA.

    We don't need numerical correctness here — only that the helper rewrites
    self-attn nodes and leaves cross-attn alone. Cross-attn is identified by an
    EMPTY ``input[6]`` (no past_key) per the Whisper-decoder convention.
    """

    # 4-d MHA tensors: [B, S, H, D] not required; we only run graph rewrite.
    # Use empty strings + dummy initializers so onnx.checker accepts the graph.
    def vinfo(name, dtype, shape):
        return helper.make_tensor_value_info(name, dtype, shape)

    # Self-attn MHA: input[6]="past_key_self_0", input[7]="past_value_self_0"
    self_attn = helper.make_node(
        "MultiHeadAttention",
        inputs=[
            "q_self",
            "k_self",
            "v_self",
            "",  # bias
            "",  # key_padding_mask
            "",  # attention_bias
            "past_key_self_0",
            "past_value_self_0",
        ],
        outputs=["out_self", "present_key_self_0", "present_value_self_0"],
        domain="com.microsoft",
        num_heads=4,
    )
    # Cross-attn MHA: input[6]=="" (no past_key) — must NOT be rewritten.
    cross_attn = helper.make_node(
        "MultiHeadAttention",
        inputs=["q_cross", "k_cross", "v_cross"],
        outputs=["out_cross"],
        domain="com.microsoft",
        num_heads=4,
    )

    graph = helper.make_graph(
        nodes=[self_attn, cross_attn],
        name="synthetic_decoder",
        inputs=[
            vinfo("q_self", TensorProto.FLOAT16, [1, 1, 4, 8]),
            vinfo("k_self", TensorProto.FLOAT16, [1, 1, 4, 8]),
            vinfo("v_self", TensorProto.FLOAT16, [1, 1, 4, 8]),
            vinfo("past_key_self_0", TensorProto.FLOAT16, [1, 4, 16, 8]),
            vinfo("past_value_self_0", TensorProto.FLOAT16, [1, 4, 16, 8]),
            vinfo("q_cross", TensorProto.FLOAT16, [1, 1, 4, 8]),
            vinfo("k_cross", TensorProto.FLOAT16, [1, 16, 4, 8]),
            vinfo("v_cross", TensorProto.FLOAT16, [1, 16, 4, 8]),
        ],
        outputs=[
            vinfo("out_self", TensorProto.FLOAT16, [1, 1, 4, 8]),
            vinfo("present_key_self_0", TensorProto.FLOAT16, [1, 4, 16, 8]),
            vinfo("present_value_self_0", TensorProto.FLOAT16, [1, 4, 16, 8]),
            vinfo("out_cross", TensorProto.FLOAT16, [1, 1, 4, 8]),
        ],
    )
    return helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 17),
            helper.make_opsetid("com.microsoft", 1),
        ],
    )


def _self_attn_nodes(m: onnx.ModelProto):
    return [
        n
        for n in m.graph.node
        if n.op_type == "MultiHeadAttention"
        and len(n.input) >= 7
        and n.input[6].startswith("past_key_self_")
    ]


def _cross_attn_nodes(m: onnx.ModelProto):
    return [
        n
        for n in m.graph.node
        if n.op_type == "MultiHeadAttention"
        and (len(n.input) < 7 or not n.input[6].startswith("past_key_self_"))
    ]


def _get_attr(node, name):
    for a in node.attribute:
        if a.name == name:
            return a
    return None


def test_inject_seqlens_k_synthetic(tmp_path: pathlib.Path) -> None:
    src = tmp_path / "synthetic_decoder.onnx"
    dst = tmp_path / "synthetic_decoder.seqlens.onnx"
    onnx.save(_make_synthetic_decoder(), str(src))

    inject_seqlens_k(src, dst)

    m = onnx.load(str(dst))
    # Graph input added
    assert "past_sequence_length" in {i.name for i in m.graph.input}
    psl = next(i for i in m.graph.input if i.name == "past_sequence_length")
    assert psl.type.tensor_type.elem_type == TensorProto.INT32
    assert [d.dim_value for d in psl.type.tensor_type.shape.dim] == [1]

    # Self-attn rewritten: 9 inputs, input[8]==past_sequence_length, and NO
    # past_present_share_buffer attribute (ORT's MHA schema rejects it).
    self_nodes = _self_attn_nodes(m)
    assert len(self_nodes) == 1
    n = self_nodes[0]
    assert len(n.input) == 9
    assert n.input[8] == "past_sequence_length"
    assert _get_attr(n, "past_present_share_buffer") is None

    # Cross-attn untouched: still 3 inputs, no ppsb attribute
    cross_nodes = _cross_attn_nodes(m)
    assert len(cross_nodes) == 1
    c = cross_nodes[0]
    assert len(c.input) == 3
    assert _get_attr(c, "past_present_share_buffer") is None


def test_inject_seqlens_k_idempotent(tmp_path: pathlib.Path) -> None:
    src = tmp_path / "synthetic_decoder.onnx"
    dst = tmp_path / "synthetic_decoder.seqlens.onnx"
    onnx.save(_make_synthetic_decoder(), str(src))

    inject_seqlens_k(src, dst)
    mtime1 = dst.stat().st_mtime_ns
    # Calling again with existing dst is a no-op (idempotent).
    inject_seqlens_k(src, dst)
    assert dst.stat().st_mtime_ns == mtime1


@pytest.mark.skipif(
    not WHISPER_DECODER.exists(),
    reason=f"Whisper decoder not downloaded at {WHISPER_DECODER}",
)
def test_inject_seqlens_k_real_whisper_decoder(tmp_path: pathlib.Path) -> None:
    dst = tmp_path / "decoder.seqlens.onnx"
    inject_seqlens_k(WHISPER_DECODER, dst)

    m = onnx.load(str(dst), load_external_data=False)
    assert "past_sequence_length" in {i.name for i in m.graph.input}

    self_nodes = _self_attn_nodes(m)
    cross_nodes = _cross_attn_nodes(m)
    # Whisper-large-v3 has 32 decoder layers → 32 self + 32 cross MHAs.
    assert len(self_nodes) == 32, f"expected 32 self-attn MHAs, got {len(self_nodes)}"
    assert len(cross_nodes) == 32, (
        f"expected 32 cross-attn MHAs, got {len(cross_nodes)}"
    )
    for n in self_nodes:
        assert len(n.input) == 9
        assert n.input[8] == "past_sequence_length"
        assert _get_attr(n, "past_present_share_buffer") is None
    for c in cross_nodes:
        assert _get_attr(c, "past_present_share_buffer") is None
