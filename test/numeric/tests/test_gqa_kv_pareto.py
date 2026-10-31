# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.

"""KV-Pareto per-token KV-cache codec (k8v8_g32_planar_v1), operator level.

The graph keeps com.microsoft GroupQueryAttention (no QDQ). Its past/present
caches are [1, 4, C, 140] FP16 carriers holding planar INT8 K/V with one BF16
scale and INT8 zero point per token, KV head and 32 channels, selected by the
provider options kvCacheQuantizationBits=8 + kv_cache_codec, with past and
present bound to the same device buffers. Geometries: Qwen3.8-27B (H24/G4/D256)
and Qwen3.5-4B (H16/G4/D256); gfx1151 only.

KV-Pareto is compiled by default but inference is opt-in. The runtime reads
HIPDNN_EP_USE_KV_PARETO once per process, so this module follows the pytest
process environment:

  unset/0: the codec request is rejected and dense GQA is unchanged;
  1:       codec numerics, cache bytes and contract rejections are checked:

    HIPDNN_EP_USE_KV_PARETO=1 python -m pytest test/numeric/tests/test_gqa_kv_pareto.py \
        --ep-name hipgpu --ep-dll <libhipgpu.so|hipgpu.dll>

Semantics checked against an FP64 oracle: every call attends to the
dequantized packed history plus the direct FP16 new K/V; the new rows are then
stored packed, byte-exact to the reference quantizer below.
"""

import os

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from ml_dtypes import bfloat16
from onnx import TensorProto as T
from onnx import helper as h

WIDTH = 140
CODEC = {"kvCacheQuantizationBits": "8", "kv_cache_codec": "k8v8_g32_planar_v1"}
ENABLED = os.environ.get("HIPDNN_EP_USE_KV_PARETO") == "1"
requires_kv_pareto = pytest.mark.skipif(
    not ENABLED, reason="KV-Pareto inference is opt-in: set HIPDNN_EP_USE_KV_PARETO=1"
)


@pytest.fixture(scope="module")
def device(pytestconfig):
    path = pytestconfig.getoption("--ep-dll")
    if not path:
        pytest.skip("--ep-dll is required")
    alias = pytestconfig.getoption("--ep-name")
    capi = ort.capi._pybind_state
    capi.register_execution_provider_library(alias, os.path.realpath(path))
    return next(d for d in capi.get_ep_devices() if d.ep_name == alias)


def model(path, capacity, width, heads):
    ins = [
        h.make_tensor_value_info(n, T.FLOAT16, [1, "N", d])
        for n, d in [("query", heads * 256), ("key", 1024), ("value", 1024)]
    ]
    ins += [
        h.make_tensor_value_info(n, T.FLOAT16, [1, 4, capacity, width])
        for n in ["past_key", "past_value"]
    ]
    ins += [
        h.make_tensor_value_info("seqlens", T.INT32, [1]),
        h.make_tensor_value_info("total", T.INT32, []),
    ]
    outs = [h.make_tensor_value_info("output", T.FLOAT16, [1, "N", heads * 256])]
    outs += [
        h.make_tensor_value_info(n, T.FLOAT16, [1, 4, capacity, width])
        for n in ["present_key", "present_value"]
    ]
    node = h.make_node(
        "GroupQueryAttention",
        [x.name for x in ins],
        [x.name for x in outs],
        domain="com.microsoft",
        num_heads=heads,
        kv_num_heads=4,
        scale=1 / 16,
    )
    m = h.make_model(
        h.make_graph([node], "k8v8_codec_gqa", ins, outs),
        opset_imports=[h.make_opsetid("", 17), h.make_opsetid("com.microsoft", 1)],
    )
    m.ir_version = 13
    onnx.save(m, path)


def session(path, device, options=None):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    so.add_provider_for_devices(
        [device], {"artifact_format": "LLVM_IR", **(options or {})}
    )
    return ort.InferenceSession(str(path), sess_options=so)


def bf(x):
    return np.asarray(x, bfloat16).astype(np.float32)


def quantize(x):
    """Reference codec for FP16 values [4, L, 256] (BF16 intermediates)."""
    groups = np.asarray(x, np.float16).astype(np.float32).reshape(4, -1, 8, 32)
    lo, hi = groups.min(-1), groups.max(-1)
    scale = bf(np.maximum(bf(hi - lo), bf(1e-6)) / 255)
    zero = np.clip(-128 - np.rint(bf(lo / scale)), -128, 127).astype(np.int8)
    payload = np.clip(
        np.rint(bf(groups / scale[..., None])) + zero[..., None], -128, 127
    ).astype(np.int8)
    deq = (payload.astype(np.float32) - zero[..., None]) * scale[..., None]
    return deq.reshape(x.shape), payload.reshape(x.shape), scale.astype(bfloat16), zero


def unpack(cache, capacity):
    raw = np.ascontiguousarray(cache).view(np.uint8).reshape(-1)
    a, b = 4 * capacity * 256, 4 * capacity * 256 + 4 * capacity * 16
    return (
        raw[:a].view(np.int8).reshape(4, capacity, 256),
        raw[a:b].view(bfloat16).reshape(4, capacity, 8),
        raw[b:].view(np.int8).reshape(4, capacity, 8),
    )


def oracle(q, hist_k, hist_v, cur_k, cur_v, heads):
    """q [n, H, 256]; history [4, past, 256] (dequantized); current [n, 4, 256]."""
    n, past, hpg = q.shape[0], hist_k.shape[1], heads // 4
    out = np.empty((n, heads, 256))
    for head in range(heads):
        g = head // hpg
        k = np.concatenate([hist_k[g], cur_k[:, g]]).astype(np.float64)
        v = np.concatenate([hist_v[g], cur_v[:, g]]).astype(np.float64)
        s = q[:, head].astype(np.float64) @ k.T / 16
        s[np.arange(past + n)[None, :] > past + np.arange(n)[:, None]] = -np.inf
        w = np.exp(s - s.max(-1, keepdims=True))
        out[:, head] = (w / w.sum(-1, keepdims=True)) @ v
    return out.reshape(1, n, heads * 256)


class Shared:
    """Device cache buffers bound as both past and present (shared buffer)."""

    def __init__(self, sess, device, capacity, width=WIDTH):
        self.sess, self.capacity = sess, capacity
        memory = device.memory_info(ort.OrtDeviceMemoryType.DEFAULT)
        self.caches = [
            ort.OrtValue.ortvalue_from_shape_and_type(
                (1, 4, capacity, width), np.float16, memory_info=memory
            )
            for _ in range(2)
        ]

    def run(self, q, k, v, seqlen, present=None):
        b = self.sess.io_binding()
        for name, x in [
            ("query", q),
            ("key", k),
            ("value", v),
            ("seqlens", np.array([seqlen], np.int32)),
            ("total", np.array(self.capacity, np.int32)),
        ]:
            b.bind_cpu_input(name, x)
        for name, x in zip(["past_key", "past_value"], self.caches):
            b.bind_ortvalue_input(name, x)
        b.bind_output("output")
        for name, x in zip(["present_key", "present_value"], present or self.caches):
            b.bind_ortvalue_output(name, x)
        self.sess.run_with_iobinding(b)
        return b.copy_outputs_to_cpu()


@requires_kv_pareto
@pytest.mark.parametrize("heads", [24, 16])
@pytest.mark.parametrize(
    "capacity,schedule",
    [
        (128, [7, 1, 1, 13, 1, 33, 1]),
        (512, [256, 1, 1, 200, 1]),
        (64, [1, 1, 1]),
        (2304, [2200, 1, 1, 64, 1]),
    ],
)
def test_codec_attention_and_cache(device, tmp_path, heads, capacity, schedule):
    path = tmp_path / "gqa.onnx"
    model(path, capacity, WIDTH, heads)
    shared = Shared(session(path, device, CODEC), device, capacity)
    r = np.random.default_rng(capacity + heads)
    keys = np.zeros((4, 0, 256), np.float16)
    values = np.zeros((4, 0, 256), np.float16)
    for n in schedule:
        q, k, v = [
            (r.normal(size=(1, n, d)) * 0.5).astype(np.float16)
            for d in [heads * 256, 1024, 1024]
        ]
        output, kc, vc = shared.run(q, k, v, keys.shape[1] + n - 1)
        hist = [quantize(x)[0] for x in (keys, values)]
        expected = oracle(
            q.reshape(n, heads, 256),
            *hist,
            k.reshape(n, 4, 256),
            v.reshape(n, 4, 256),
            heads,
        )
        np.testing.assert_allclose(output, expected, atol=0.002, rtol=0.03)
        assert np.linalg.norm(output - expected) / np.linalg.norm(expected) < 0.005
        keys = np.concatenate([keys, k.reshape(n, 4, 256).transpose(1, 0, 2)], 1)
        values = np.concatenate([values, v.reshape(n, 4, 256).transpose(1, 0, 2)], 1)
        length = keys.shape[1]
        for cache, x in [(kc, keys), (vc, values)]:
            payload, scale, zero = unpack(cache, capacity)
            _, want_payload, want_scale, want_zero = quantize(x)
            np.testing.assert_array_equal(payload[:, :length], want_payload)
            np.testing.assert_array_equal(
                scale[:, :length].view(np.uint16),
                want_scale.reshape(4, -1, 8).view(np.uint16),
            )
            np.testing.assert_array_equal(zero[:, :length], want_zero.reshape(4, -1, 8))


@pytest.mark.skipif(ENABLED, reason="checks the default (opt-out) behavior")
def test_kv_pareto_disabled_by_default(device, tmp_path):
    """Without the flag a codec request fails; dense GQA is unchanged."""
    capacity, n = 64, 5
    packed = tmp_path / "packed.onnx"
    model(packed, capacity, WIDTH, 24)
    shared = Shared(session(packed, device, CODEC), device, capacity)
    zeros = [np.zeros((1, 1, d), np.float16) for d in [24 * 256, 1024, 1024]]
    with pytest.raises(Exception, match="inference_compute"):
        shared.run(*zeros, 0)
    del shared

    dense = tmp_path / "dense.onnx"
    model(dense, capacity, 256, 24)
    shared = Shared(session(dense, device), device, capacity, 256)
    r = np.random.default_rng(7)
    q, k, v = [
        (r.normal(size=(1, n, d)) * 0.5).astype(np.float16)
        for d in [24 * 256, 1024, 1024]
    ]
    output, kc, vc = shared.run(q, k, v, n - 1)
    empty = np.zeros((4, 0, 256), np.float32)
    expected = oracle(
        q.reshape(n, 24, 256),
        empty,
        empty,
        k.reshape(n, 4, 256),
        v.reshape(n, 4, 256),
        24,
    )
    np.testing.assert_allclose(output, expected, atol=0.002, rtol=0.03)
    np.testing.assert_array_equal(kc[0, :, :n], k.reshape(n, 4, 256).transpose(1, 0, 2))
    np.testing.assert_array_equal(vc[0, :, :n], v.reshape(n, 4, 256).transpose(1, 0, 2))


@requires_kv_pareto
def test_codec_rejections(device, tmp_path):
    zeros = [np.zeros((1, 4, d), np.float16) for d in [24 * 256, 1024, 1024]]
    packed = tmp_path / "packed.onnx"
    model(packed, 64, WIDTH, 24)
    dense = tmp_path / "dense.onnx"
    model(dense, 64, 256, 24)
    cases = [
        ("packed width without codec options", packed, None, False),
        ("codec options with dense width", dense, CODEC, False),
        (
            "unsupported codec identity",
            packed,
            {"kvCacheQuantizationBits": "4", "kv_cache_codec": "tq4_mse_qr_v1"},
            False,
        ),
        ("bit width without codec", packed, {"kvCacheQuantizationBits": "8"}, False),
        (
            "codec without bit width",
            packed,
            {"kv_cache_codec": "k8v8_g32_planar_v1"},
            False,
        ),
        ("unshared past/present", packed, CODEC, True),
    ]
    for label, path, options, unshared in cases:
        width = 256 if path == dense else WIDTH
        shared = Shared(session(path, device, options), device, 64, width)
        present = Shared(shared.sess, device, 64, width).caches if unshared else None
        with pytest.raises(Exception, match="inference_compute"):
            shared.run(*zeros, 3, present)
        del shared
