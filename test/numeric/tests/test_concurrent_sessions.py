#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Cross-session concurrency tests for the op-state slot mechanism.

The op-state slot design gives every ORT session its own array of
per-session op-state objects (constructed at ``inference_init``, torn down
at ``inference_cleanup``). Two sessions running concurrently must therefore
never share a per-session slot: each session resolves a slot into *its own*
state object, so concurrent ``inference_compute`` calls on different sessions
cannot corrupt one another. See docs/design/op-state-slots-design.md.

This harness spins up two ORT EP sessions on two threads and drives ``Run``
on both concurrently (synchronised by a barrier so the compute windows
overlap), then checks each thread's output against an ORT-CPU reference
computed up front.

Scenarios:

  * ``conv`` -- both sessions run a Conv whose MIOpen workspace lives in the
    per-session ``RuntimeState::conv_scratch`` field (one buffer per session,
    grown on demand, shared across that session's conv instances) but with
    *different shapes*, so they need *different* workspace sizes. Because each
    session has its own ``RuntimeState``, the two sessions cannot share the
    buffer; if they did, the workspace would belong to the wrong shape and the
    output would be wrong (or it would crash). This is the cross-session
    isolation assertion for conv's workspace.

  * ``qmoe`` -- both sessions run a QMoE whose device + pinned-host scratch
    lives in the per-session ``RuntimeState::qmoe_scratch`` /
    ``qmoe_host_scratch`` fields with *different shapes* (token count / hidden
    / intermediate / experts), so they need *different* scratch sizes. Each
    session has its own ``RuntimeState``, so the buffers cannot be shared
    across sessions; if they were, the transient sub-buffers would be mis-sized
    and the output would be wrong (or it would crash). This is the cross-session
    isolation assertion for qmoe's scratch.

  * ``matmul_nbits_asym`` -- both sessions run an asymmetric MatMulNBits (a
    per-session ``MatmulNbitsState`` slot owning the zero_points unpack cache)
    with *different* (N, K), so they need *different* unpack-cache buffer
    sizes. Per-instance isolation assertion for matmul_nbits's migrated cache
    (formerly the shared RuntimeState::zp_unpack_cache).

  * ``causal_conv_with_state`` -- both sessions run a depthwise
    CausalConvWithState (a per-session ``CausalConvState`` slot owning the
    MIOpen descriptor/algo cache) with *different* shapes, so they cache
    *different* MIOpen plans. Per-instance isolation assertion for the
    migrated cache (formerly the shared RuntimeState::causal_conv_cache).

  * ``multi_head_attention`` -- both sessions run a self-attention MHA (a
    per-session ``MhaState`` slot owning the hipBLASLt Score/Value GEMM
    descriptor cache) with *different* shapes, so they cache *different* GEMM
    descriptors. Per-instance isolation assertion for the migrated cache
    (formerly the shared RuntimeState::mha_gemm_cache). The identical mechanism
    backs GQA's GqaState (RuntimeState::gqa_gemm_cache); GQA itself is not a
    concurrency scenario here because its KV-cache present_key/value outputs hit
    the output-allocator passthrough limitation (unrelated to op-state slots).

The test reaches into the EP backend's discovered ``_ep_device`` /
``_provider_options`` to build long-lived sessions directly (the framework's
``Backend.run`` creates and destroys a session per call, which would not keep
two sessions alive at once). It skips cleanly when the selected backend is not
an ORT EP backend.
"""

from __future__ import annotations

import threading

import numpy as np
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes
from framework.ort_cpu_backend import OrtCpuBackend
from tests.test_causal_conv_with_state import _make_causal_conv_with_state_model
from tests.test_multi_head_attention import _make_mha_model, _qkv_inputs
from tests.test_qmoe import _make_qmoe_model

# Iterations each thread runs back-to-back. Enough that the two sessions'
# compute windows overlap heavily after the barrier release, so any
# cross-session state corruption has many chances to surface.
ITERS = 16


def _build_conv(in_c, out_c, k, spatial, seed):
    """fp16 2D Conv with weight+bias initializers; stride 1, same-size out."""
    pad = k // 2
    dtype = np.float16
    x_info = helper.make_tensor_value_info(
        "X", TensorProto.FLOAT16, [1, in_c, spatial, spatial]
    )
    y_info = helper.make_tensor_value_info(
        "Y", TensorProto.FLOAT16, [1, out_c, spatial, spatial]
    )
    rng = np.random.default_rng(seed)
    w = rng.uniform(-0.1, 0.1, [out_c, in_c, k, k]).astype(dtype)
    b = rng.uniform(-0.1, 0.1, [out_c]).astype(dtype)
    node = helper.make_node(
        "Conv",
        ["X", "weight", "bias"],
        ["Y"],
        group=1,
        kernel_shape=[k, k],
        strides=[1, 1],
        pads=[pad, pad, pad, pad],
        dilations=[1, 1],
    )
    model = make_model_from_nodes(
        [node],
        [x_info],
        [y_info],
        initializers=[
            numpy_helper.from_array(w, name="weight"),
            numpy_helper.from_array(b, name="bias"),
        ],
    )
    x = rng.uniform(-1.0, 1.0, [1, in_c, spatial, spatial]).astype(dtype)
    return model, [x]


def _build_matmul_nbits_asym(seq_len, K, N, block_size, seed):
    """fp16 MatMulNBits (com.microsoft) with 4-bit weights + packed-uint8
    zero_points -- the *asym* path, which is the only one that touches the
    per-instance MatmulNbitsState (its zero_points unpack cache). Different
    (N, K) -> different unpack-cache buffer sizes (need = N * ceil(K/bs)),
    so two specs need different per-session slot buffers.

    The harness computes the ORT-CPU reference from this same ONNX model, so
    the (arbitrary but spec-valid) packed weight/zp bytes define the truth;
    only the shapes/dtypes need to be spec-correct.
    """
    dtype = np.float16
    n_blocks = (K + block_size - 1) // block_size
    packed_zp_cols = (n_blocks + 1) // 2  # 4-bit: two zero points per byte

    x_info = helper.make_tensor_value_info("X", TensorProto.FLOAT16, [1, seq_len, K])
    y_info = helper.make_tensor_value_info("Y", TensorProto.FLOAT16, [1, seq_len, N])

    rng = np.random.default_rng(seed)
    q_weight = rng.integers(0, 256, [N, n_blocks, block_size // 2], dtype=np.uint8)
    scales = rng.uniform(-0.05, 0.05, [N, n_blocks]).astype(dtype)
    zero_points = rng.integers(0, 256, [N, packed_zp_cols], dtype=np.uint8)

    node = helper.make_node(
        "MatMulNBits",
        ["X", "weight_Q4", "weight_scales", "weight_zp"],
        ["Y"],
        domain="com.microsoft",
        K=K,
        N=N,
        bits=4,
        block_size=block_size,
        accuracy_level=4,
    )
    ms_opset = helper.make_opsetid("com.microsoft", 1)
    model = make_model_from_nodes(
        [node],
        [x_info],
        [y_info],
        initializers=[
            numpy_helper.from_array(q_weight, name="weight_Q4"),
            numpy_helper.from_array(scales, name="weight_scales"),
            numpy_helper.from_array(zero_points, name="weight_zp"),
        ],
        opset=21,
        extra_opsets=[ms_opset],
    )
    x = rng.uniform(-1.0, 1.0, [1, seq_len, K]).astype(dtype)
    return model, [x]


def _build_causal_conv(channels, seq_len, kernel_size, seed):
    """fp16 CausalConvWithState (com.microsoft) -- the depthwise causal conv
    whose MIOpen descriptor/algo cache moved into the per-instance
    CausalConvState slot (formerly the shared RuntimeState::causal_conv_cache).

    Different (channels, seq_len, kernel_size) -> different MIOpen descriptors,
    so two sessions cache different plans. If a session read another session's
    slot, it would run with the wrong descriptor and the output would be wrong
    (or it would crash).
    """
    model = _make_causal_conv_with_state_model(
        1, channels, seq_len, kernel_size, activation="silu"
    )
    rng = np.random.default_rng(seed)
    x = (rng.standard_normal((1, channels, seq_len)) * 0.5).astype(np.float16)
    state = (rng.standard_normal((1, channels, kernel_size - 1)) * 0.5).astype(
        np.float16
    )
    return model, [x, state]


def _build_mha(seq, num_heads, head_dim, seed):
    """fp16 MultiHeadAttention (com.microsoft) self-attention -- the decomposed
    Score + Value GEMM path whose hipBLASLt descriptor/algo cache moved into the
    per-instance MhaState slot (formerly the shared RuntimeState::mha_gemm_cache).

    Different (seq, num_heads, head_dim) -> different GEMM shapes, so two
    sessions cache different descriptors. If a session read another session's
    slot, it would run with the wrong GEMM descriptor and the output would be
    wrong (or it would crash). Self-attention (no present_key/value outputs)
    keeps the model inside the output-allocator-supported subset.
    """
    model = _make_mha_model(1, seq, seq, num_heads, head_dim)
    q, k, v = _qkv_inputs(1, seq, seq, num_heads, head_dim, seed=seed)
    return model, [q, k, v]


def _build_qmoe(seq_len, hidden, intermediate, num_experts, top_k, seed):
    """fp16 QMoE (com.microsoft) with 4-bit packed expert weights.

    Shapes drive the per-session RuntimeState::qmoe_scratch sizing (num_tokens
    * hidden / intermediate / experts), so two differently-shaped specs need
    different device + pinned-host buffer sizes.
    """
    model = _make_qmoe_model(
        1, seq_len, hidden, intermediate, num_experts, top_k, seed=seed
    )
    rng = np.random.default_rng(seed + 100)
    x = rng.uniform(-1.0, 1.0, [1, seq_len, hidden]).astype(np.float16)
    router = rng.standard_normal([seq_len, num_experts]).astype(np.float16)
    return model, [x, router]


# Two independent (model, inputs, atol) thread specs per scenario, each with
# different shapes so the two sessions need different buffer sizes -> proves
# per-session state is resolved per-session (no cross-session sharing). conv /
# qmoe use per-session RuntimeState scratch fields; the remaining scenarios use
# per-instance op-state slots.
_SCENARIOS = {
    "conv": lambda: [
        (*_build_conv(in_c=8, out_c=16, k=3, spatial=32, seed=1), 1e-2),
        (*_build_conv(in_c=16, out_c=8, k=5, spatial=16, seed=2), 1e-2),
    ],
    "qmoe": lambda: [
        (*_build_qmoe(8, 64, 128, 4, 2, seed=1), 5e-2),
        (*_build_qmoe(4, 128, 256, 8, 2, seed=2), 5e-2),
    ],
    "matmul_nbits_asym": lambda: [
        (*_build_matmul_nbits_asym(8, 128, 64, 32, seed=1), 5e-2),
        (*_build_matmul_nbits_asym(4, 256, 128, 32, seed=2), 5e-2),
    ],
    "causal_conv_with_state": lambda: [
        (*_build_causal_conv(channels=32, seq_len=128, kernel_size=4, seed=1), 5e-3),
        (*_build_causal_conv(channels=64, seq_len=16, kernel_size=3, seed=2), 5e-3),
    ],
    "multi_head_attention": lambda: [
        (*_build_mha(seq=128, num_heads=4, head_dim=16, seed=1), 2e-2),
        (*_build_mha(seq=8, num_heads=4, head_dim=32, seed=2), 2e-2),
    ],
}


def _make_ep_session(backend, model_path):
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    opts.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    opts.add_provider_for_devices([backend._ep_device], dict(backend._provider_options))
    return ort.InferenceSession(str(model_path), sess_options=opts)


class TestConcurrentSessions:
    @pytest.mark.parametrize("scenario", list(_SCENARIOS))
    def test_two_sessions_concurrent(self, model_runner, tmp_path, scenario):
        backend = model_runner.backend
        if not hasattr(backend, "_ep_device"):
            pytest.skip("--backend is not an ORT EP backend; concurrency N/A")

        cpu = OrtCpuBackend()
        thread_specs = _SCENARIOS[scenario]()

        # Materialise models + up-front CPU references (sequential, before
        # any threads start) so the concurrent section only does EP work.
        prepared = []
        for idx, (model, inputs, atol) in enumerate(thread_specs):
            model_path = tmp_path / f"{scenario}_t{idx}.onnx"
            model_path.write_bytes(model.SerializeToString())
            ref = cpu.run(str(model_path), inputs)
            prepared.append((model_path, inputs, ref, atol))

        barrier = threading.Barrier(len(prepared))
        results: dict[int, list[np.ndarray]] = {}
        errors: dict[int, BaseException] = {}

        def worker(tid, model_path, inputs):
            try:
                sess = _make_ep_session(backend, model_path)
                names = [i.name for i in sess.get_inputs()]
                feed = {n: v for n, v in zip(names, inputs)}
                # Release both threads at once so compute overlaps.
                barrier.wait()
                outs = None
                for _ in range(ITERS):
                    outs = sess.run(None, feed)
                results[tid] = outs
                del sess
            except BaseException as exc:  # noqa: BLE001 - report to main thread
                errors[tid] = exc

        threads = [
            threading.Thread(
                target=worker, args=(tid, mp, inp), name=f"{scenario}-t{tid}"
            )
            for tid, (mp, inp, _ref, _atol) in enumerate(prepared)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"worker thread(s) raised: {errors}"

        # Each session's final output must match its own CPU reference,
        # proving concurrent execution did not cross-contaminate state.
        for tid, (_mp, _inp, ref, atol) in enumerate(prepared):
            assert tid in results, f"thread {tid} produced no result"
            compare_outputs(results[tid], ref, atol=atol)
