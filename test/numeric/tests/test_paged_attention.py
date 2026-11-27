#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Tests for com.microsoft PagedAttention (causal, without rotary).

The query tokens of every sequence are packed back to back. Each token's key
and value go into the paged cache at the slot its position maps to through the
block table, and the token then attends to all earlier positions of its
sequence, read through the same table.

ORT CPU has no PagedAttention kernel, so the reference is an equivalent graph.
Each sequence gathers its history from the cache, appends its new keys and
values, and runs plain scaled-dot-product attention with a causal mask;
ScatterND produces the updated caches. The sequence lengths, past lengths and
block table are baked into the reference graph, so its cumulative_sequence_length,
past_seqlens and block_table inputs go unused.

The cache outputs are separate buffers from the cache inputs here, so these
cases cover the path that first copies the whole input cache.
"""

import math

import numpy as np
import pytest
from onnx import TensorProto as TP
from onnx import helper, numpy_helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes

_OPSETS = dict(opset=22, extra_opsets=[helper.make_opsetid("com.microsoft", 1)])

_ATOL, _RTOL = 5e-3, 1e-2


def _layout(seq_lens, past_lens, block_size, rng):
    """Returns (block_table, num_blocks, max_blocks) with scattered blocks."""
    needed = [-(-(p + n) // block_size) for n, p in zip(seq_lens, past_lens)]
    max_blocks = max(needed)
    num_blocks = sum(needed) + 3
    ids = rng.permutation(num_blocks)
    table = np.full((len(seq_lens), max_blocks), -1, dtype=np.int32)
    used = 0
    for b, count in enumerate(needed):
        table[b, :count] = ids[used : used + count]
        used += count
    return table, num_blocks, max_blocks


def _slot(table, block_size, b, pos):
    return int(table[b, pos // block_size]) * block_size + pos % block_size


def _io(seq_lens, num_heads, kv_heads, head_size, block_size, num_blocks, max_blocks):
    total = sum(seq_lens)
    batch = len(seq_lens)
    cache = [num_blocks, block_size, kv_heads, head_size]
    f16 = TP.FLOAT16
    inputs = [
        helper.make_tensor_value_info("query", f16, [total, num_heads * head_size]),
        helper.make_tensor_value_info("key", f16, [total, kv_heads * head_size]),
        helper.make_tensor_value_info("value", f16, [total, kv_heads * head_size]),
        helper.make_tensor_value_info("key_cache", f16, cache),
        helper.make_tensor_value_info("value_cache", f16, cache),
        helper.make_tensor_value_info(
            "cumulative_sequence_length", TP.INT32, [batch + 1]
        ),
        helper.make_tensor_value_info("past_seqlens", TP.INT32, [batch]),
        helper.make_tensor_value_info("block_table", TP.INT32, [batch, max_blocks]),
    ]
    outputs = [
        helper.make_tensor_value_info("output", f16, [total, num_heads * head_size]),
        helper.make_tensor_value_info("key_cache_out", f16, cache),
        helper.make_tensor_value_info("value_cache_out", f16, cache),
    ]
    return inputs, outputs


def _make_model(shape, scale):
    seq_lens, num_heads, kv_heads, head_size, block_size, num_blocks, max_blocks = shape
    attrs = dict(num_heads=num_heads, kv_num_heads=kv_heads)
    if scale is not None:
        attrs["scale"] = scale
    node = helper.make_node(
        "PagedAttention",
        [
            "query",
            "key",
            "value",
            "key_cache",
            "value_cache",
            "cumulative_sequence_length",
            "past_seqlens",
            "block_table",
        ],
        ["output", "key_cache_out", "value_cache_out"],
        domain="com.microsoft",
        **attrs,
    )
    inputs, outputs = _io(*shape)
    return make_model_from_nodes([node], inputs, outputs, **_OPSETS)


def _make_reference_model(shape, past_lens, table, scale):
    seq_lens, num_heads, kv_heads, head_size, block_size, num_blocks, max_blocks = shape
    total = sum(seq_lens)
    group = num_heads // kv_heads
    attn_scale = scale if scale is not None else 1.0 / math.sqrt(head_size)
    slots = num_blocks * block_size
    inits = []

    def const(name, values, dtype=np.int64):
        inits.append(numpy_helper.from_array(np.array(values, dtype=dtype), name))
        return name

    nodes = [
        helper.make_node("Cast", [n], [f"{n}32"], to=TP.FLOAT)
        for n in ("query", "key", "value", "key_cache", "value_cache")
    ]

    def reshape(src, dst, dims):
        nodes.append(helper.make_node("Reshape", [src, const(f"{dst}_s", dims)], [dst]))

    reshape("query32", "q3", [total, num_heads, head_size])
    reshape("key32", "k3", [total, kv_heads, head_size])
    reshape("value32", "v3", [total, kv_heads, head_size])
    reshape("key_cache32", "kc", [slots, kv_heads, head_size])
    reshape("value_cache32", "vc", [slots, kv_heads, head_size])

    head_map = const("head_map", [h // group for h in range(num_heads)])
    outs = []
    new_slots = []
    start = 0
    for b, (n, past) in enumerate(zip(seq_lens, past_lens)):
        s = f"s{b}"
        hist = [_slot(table, block_size, b, pos) for pos in range(past)]
        new_slots += [_slot(table, block_size, b, past + i) for i in range(n)]

        def rows(src_cache, src_new, tag):
            parts = []
            if past:
                nodes.append(
                    helper.make_node(
                        "Gather",
                        [src_cache, const(f"{s}_{tag}_slots", hist)],
                        [f"{s}_{tag}_hist"],
                        axis=0,
                    )
                )
                parts.append(f"{s}_{tag}_hist")
            nodes.append(
                helper.make_node(
                    "Slice",
                    [
                        src_new,
                        const(f"{s}_{tag}_a", [start]),
                        const(f"{s}_{tag}_b", [start + n]),
                        const(f"{s}_{tag}_ax", [0]),
                    ],
                    [f"{s}_{tag}_new"],
                )
            )
            parts.append(f"{s}_{tag}_new")
            nodes.append(helper.make_node("Concat", parts, [f"{s}_{tag}_all"], axis=0))
            nodes.append(
                helper.make_node(
                    "Gather", [f"{s}_{tag}_all", head_map], [f"{s}_{tag}_rep"], axis=1
                )
            )
            return f"{s}_{tag}_rep"

        k_rep = rows("kc", "k3", "k")
        v_rep = rows("vc", "v3", "v")
        nodes += [
            helper.make_node(
                "Slice",
                [
                    "q3",
                    const(f"{s}_qa", [start]),
                    const(f"{s}_qb", [start + n]),
                    const(f"{s}_qax", [0]),
                ],
                [f"{s}_q"],
            ),
            helper.make_node("Transpose", [f"{s}_q"], [f"{s}_qt"], perm=[1, 0, 2]),
            helper.make_node("Transpose", [k_rep], [f"{s}_kt"], perm=[1, 2, 0]),
            helper.make_node("Transpose", [v_rep], [f"{s}_vt"], perm=[1, 0, 2]),
            helper.make_node("MatMul", [f"{s}_qt", f"{s}_kt"], [f"{s}_scores"]),
            helper.make_node(
                "Mul",
                [f"{s}_scores", const(f"{s}_scale", attn_scale, np.float32)],
                [f"{s}_scaled"],
            ),
        ]
        keys = past + n
        mask = np.where(
            np.arange(keys)[None, :] <= (past + np.arange(n))[:, None], 0.0, -1e30
        ).astype(np.float32)[None]
        nodes += [
            helper.make_node(
                "Add",
                [f"{s}_scaled", const(f"{s}_mask", mask, np.float32)],
                [f"{s}_masked"],
            ),
            helper.make_node("Softmax", [f"{s}_masked"], [f"{s}_probs"], axis=-1),
            helper.make_node("MatMul", [f"{s}_probs", f"{s}_vt"], [f"{s}_ctx"]),
            helper.make_node("Transpose", [f"{s}_ctx"], [f"{s}_ctxt"], perm=[1, 0, 2]),
            helper.make_node(
                "Reshape",
                [f"{s}_ctxt", const(f"{s}_flat", [n, num_heads * head_size])],
                [f"{s}_out"],
            ),
        ]
        outs.append(f"{s}_out")
        start += n

    indices = const("new_slots", np.array(new_slots).reshape(total, 1))
    nodes += [
        helper.make_node("Concat", outs, ["out32"], axis=0),
        helper.make_node("Cast", ["out32"], ["output"], to=TP.FLOAT16),
        helper.make_node("ScatterND", ["kc", indices, "k3"], ["kc_new"]),
        helper.make_node("ScatterND", ["vc", indices, "v3"], ["vc_new"]),
    ]
    cache_dims = [num_blocks, block_size, kv_heads, head_size]
    for src, dst in (("kc_new", "key_cache_out"), ("vc_new", "value_cache_out")):
        nodes += [
            helper.make_node(
                "Reshape", [src, const(f"{dst}_s", cache_dims)], [f"{dst}32"]
            ),
            helper.make_node("Cast", [f"{dst}32"], [dst], to=TP.FLOAT16),
        ]
    inputs, outputs = _io(*shape)
    return make_model_from_nodes(nodes, inputs, outputs, inits, **_OPSETS)


class TestPagedAttention:
    @pytest.mark.parametrize(
        "num_heads,kv_heads,head_size,block_size,seq_lens,past_lens,scale",
        [
            # Per-token kernel: prefill with and without history.
            (4, 2, 64, 16, [20, 9], [0, 5], 0.125),
            # Decode: one token per request, default scale.
            (4, 2, 64, 16, [1, 1, 1], [17, 40, 3], None),
            # Long history, so the context is split across blocks.
            (6, 2, 64, 16, [1], [700], 0.125),
            # Head size 256: decode with grouped heads.
            (8, 2, 256, 32, [1, 1], [33, 100], None),
            # Head size 256: tiled prefill, with and without history.
            (8, 2, 256, 32, [40], [0], 0.0625),
            (6, 2, 256, 64, [33, 3], [50, 7], 0.0625),
            # Head size 256: split decode with one KV head.
            (4, 1, 256, 32, [1], [700], None),
        ],
    )
    def test_paged_attention(
        self,
        model_runner,
        num_heads,
        kv_heads,
        head_size,
        block_size,
        seq_lens,
        past_lens,
        scale,
    ):
        rng = np.random.default_rng(0xA77 + sum(seq_lens) + sum(past_lens) + head_size)
        table, num_blocks, max_blocks = _layout(seq_lens, past_lens, block_size, rng)
        shape = (
            seq_lens,
            num_heads,
            kv_heads,
            head_size,
            block_size,
            num_blocks,
            max_blocks,
        )
        total = sum(seq_lens)
        cache_shape = (num_blocks, block_size, kv_heads, head_size)

        def rand(*dims):
            return (rng.standard_normal(dims) * 0.5).astype(np.float16)

        feeds = [
            rand(total, num_heads * head_size),
            rand(total, kv_heads * head_size),
            rand(total, kv_heads * head_size),
            rand(*cache_shape),
            rand(*cache_shape),
            np.concatenate([[0], np.cumsum(seq_lens)]).astype(np.int32),
            np.array(past_lens, dtype=np.int32),
            table,
        ]

        actual, expected = model_runner.run_sample(
            _make_model(shape, scale),
            feeds,
            reference_model=_make_reference_model(shape, past_lens, table, scale),
        )
        compare_outputs(actual[:1], expected[:1], atol=_ATOL, rtol=_RTOL)
        # The caches only move fp16 values, so they must match exactly,
        # including the blocks this call does not write.
        compare_outputs(actual[1:], expected[1:], atol=0, rtol=0)
