#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Per-op correctness test for the Conv backend.

Builds a one-node ONNX `Conv` model in memory, runs it on the CPU EP and on
the MorphiZen EP, and compares outputs. Whichever Conv backend the EP picks
is decided by HIPDNN_EP_CONV:

    (env unset)                    pytest test/python/test_op_conv.py -v -s   # default: ck for fp16
    HIPDNN_EP_CONV=ck     pytest test/python/test_op_conv.py -v -s
    HIPDNN_EP_CONV=miopen pytest test/python/test_op_conv.py -v -s

The dispatch shim (lib/Runtime/real/conv_dispatch.cpp) reads HIPDNN_EP_CONV
ONCE per process via std::getenv. The model.dll is cached in %TEMP% across
sessions and process-shared once loaded -- so flipping the env var between
tests inside a single pytest invocation will NOT switch backends. Run
pytest twice with different env values to cover both paths.

dtype: fp16. CK ships precompiled WMMA cshufflev3 instances for fp16 NHWGC
on RDNA3+; fp32 NHWGC has no instances in TheRock's CK build because
CK_ENABLE_DL_KERNELS is undef'd. wrap_miopenConvolutionForward handles both
fp16 and fp32 via element_size_bytes.
"""

import gc
import io
import os
import pathlib
import subprocess
import sys
import tempfile

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent

# Re-use the conftest registration helper.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from conftest import register_morphizen_ep, cleanup  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic Conv-only ONNX builder.
# ---------------------------------------------------------------------------


def make_conv_only_onnx(N, C, H, W, K, ky, kx, stride, pad, dilation, group, *, seed=0):
    """Build a one-node Conv ONNX model (fp16).

    Returns (onnx_bytes, input_np, weight_np, expected_output_shape).
    Weights are baked in as an initializer so the model has only one input.
    """
    rng = np.random.default_rng(seed)
    in_shape = (N, C, H, W)
    w_shape = (K, C // group, ky, kx)

    inp = rng.standard_normal(in_shape, dtype=np.float32).astype(np.float16)
    weights = (rng.standard_normal(w_shape, dtype=np.float32) * 0.1).astype(np.float16)

    Ho = (H + 2 * pad - dilation * (ky - 1) - 1) // stride + 1
    Wo = (W + 2 * pad - dilation * (kx - 1) - 1) // stride + 1
    out_shape = (N, K, Ho, Wo)

    weight_init = numpy_helper.from_array(weights, name="W")
    inp_proto = helper.make_tensor_value_info("X", TensorProto.FLOAT16, in_shape)
    out_proto = helper.make_tensor_value_info("Y", TensorProto.FLOAT16, out_shape)

    conv_node = helper.make_node(
        "Conv",
        inputs=["X", "W"],
        outputs=["Y"],
        kernel_shape=[ky, kx],
        strides=[stride, stride],
        pads=[pad, pad, pad, pad],
        dilations=[dilation, dilation],
        group=group,
    )
    graph = helper.make_graph(
        [conv_node], "conv_only", [inp_proto], [out_proto], [weight_init]
    )
    opset = [helper.make_opsetid("", 17)]
    model = helper.make_model(graph, opset_imports=opset, ir_version=10)
    onnx.checker.check_model(model)
    return model.SerializeToString(), inp, weights, out_shape


# ---------------------------------------------------------------------------
# Session helpers (in-memory, parallel to conftest.create_*_session which
# expect a path).
# ---------------------------------------------------------------------------


def _ep_session_from_bytes(model_bytes):
    devices = register_morphizen_ep(REPO_ROOT)
    if not devices:
        pytest.skip("MorphiZen EP not found -- run python build.py first")
    so = ort.SessionOptions()
    so.add_provider_for_devices(devices, {})
    return ort.InferenceSession(model_bytes, sess_options=so)


def _cpu_session_from_bytes(model_bytes):
    return ort.InferenceSession(model_bytes, providers=["CPUExecutionProvider"])


# ---------------------------------------------------------------------------
# Test matrix. Shapes cover small smoke, ResNet-stem, 1x1, 3x3, depthwise,
# strided. group=64 depthwise is the trickiest (group == C).
# ---------------------------------------------------------------------------

CONV_SHAPES = [
    # (id,           N, C, H, W, K, ky, kx, stride, pad, dilation, group)
    #
    # One representative shape per workload class -- enough to catch
    # correctness regressions across the surface that real models hit,
    # without padding the test matrix with redundant variants. Mirrors
    # bench/bench_conv.py's SHAPES list (perf side); if you add a row
    # here, mirror it there and justify what behavior or workload class
    # it covers that the others don't.
    ("smoke_3x3",      1, 3, 32, 32, 16, 3, 3, 1, 1, 1, 1),
    ("rn50_stem",      1, 3, 224, 224, 64, 7, 7, 2, 3, 1, 1),
    ("rn50_s3_3x3",    1, 256, 14, 14, 256, 3, 3, 1, 1, 1, 1),
    ("1x1",            1, 16, 32, 32, 32, 1, 1, 1, 0, 1, 1),
    pytest.param("depthwise_3x3", 1, 16, 16, 16, 16, 3, 3, 1, 1, 1, 16,
                 marks=pytest.mark.xfail(
                     reason=("MIOpen rejects group == C depthwise with "
                             "'Invalid filter channel number'; CK handles it "
                             "fine. strict=False so xpass on the CK backend "
                             "still counts as success."),
                     strict=False)),
    ("vit_l14_224",    1, 3, 224, 224, 1024, 14, 14, 14, 0, 1, 1),
    ("sdxl_unet_1280", 1, 1280, 16, 16, 1280, 3, 3, 1, 1, 1, 1),
    ("2k_vit_b16",     1, 3, 2048, 2048, 768, 16, 16, 16, 0, 1, 1),
]


@pytest.mark.parametrize(
    "name,N,C,H,W,K,ky,kx,stride,pad,dilation,group", CONV_SHAPES
)
def test_conv_cpu_vs_ep(name, N, C, H, W, K, ky, kx, stride, pad, dilation, group):
    """CPU EP vs MorphiZen EP -- output must match within fp32 tolerance.

    The MorphiZen EP backend is whichever one HIPDNN_EP_CONV selects at
    process start (default: miopen). To cover both, run pytest twice with
    HIPDNN_EP_CONV=miopen and HIPDNN_EP_CONV=ck.
    """
    # Auto = "ck for fp16, miopen for fp32" (this test is fp16, so Auto -> ck).
    backend = os.environ.get("HIPDNN_EP_CONV") or "auto(ck for fp16)"
    print(f"\n[test_conv] shape={name} backend={backend}")

    model_bytes, inp, _w, expected_shape = make_conv_only_onnx(
        N, C, H, W, K, ky, kx, stride, pad, dilation, group, seed=42
    )

    cpu_sess = _cpu_session_from_bytes(model_bytes)
    cpu_out = cpu_sess.run(None, {"X": inp})[0]
    assert cpu_out.shape == expected_shape, (
        f"CPU output shape {cpu_out.shape} != expected {expected_shape}"
    )

    ep_sess = _ep_session_from_bytes(model_bytes)
    ep_out = ep_sess.run(None, {"X": inp})[0]
    assert ep_out.shape == cpu_out.shape

    # fp16 Conv with random inputs accumulates rounding across hundreds of
    # products. Loose tolerance to absorb fp16/fp32 mantissa differences;
    # tight enough to catch a real divergence (random outputs ~ N(0,1)).
    rtol = 1e-2
    atol = 5e-2 * max(1.0, float(np.abs(cpu_out).mean()))
    cpu_out = cpu_out.astype(np.float32)
    ep_out = ep_out.astype(np.float32)
    if not np.allclose(cpu_out, ep_out, rtol=rtol, atol=atol):
        max_abs = float(np.max(np.abs(cpu_out - ep_out)))
        max_rel = float(np.max(np.abs(cpu_out - ep_out) / (np.abs(cpu_out) + 1e-6)))
        pytest.fail(
            f"backend={backend} shape={name} output mismatch: "
            f"max_abs={max_abs:.3e}, max_rel={max_rel:.3e}, "
            f"cpu_mean={cpu_out.mean():.3e}, ep_mean={ep_out.mean():.3e}"
        )

    cleanup(ep_sess, cpu_sess)


@pytest.mark.skipif(
    not os.environ.get("RUN_PERF"),
    reason="set RUN_PERF=1 to enable the bench-style perf snapshot",
)
def test_conv_perf_resnet_stem():
    """Time 100 iterations of the ResNet-stem shape on the EP. Reports tok/s
    so that backend perf evolution is trackable; not a correctness gate."""
    import time

    backend = os.environ.get("HIPDNN_EP_CONV", "miopen")
    model_bytes, inp, _w, _ = make_conv_only_onnx(
        1, 3, 64, 64, 32, 7, 7, 2, 3, 1, 1, seed=42
    )
    sess = _ep_session_from_bytes(model_bytes)

    # Warmup (autotune, weight transpose cache, etc.).
    for _ in range(3):
        sess.run(None, {"X": inp})

    n = 100
    t0 = time.perf_counter()
    for _ in range(n):
        sess.run(None, {"X": inp})
    elapsed = time.perf_counter() - t0
    avg_ms = elapsed / n * 1000
    print(
        f"\n[test_conv perf] backend={backend} shape=resnet_stem "
        f"avg={avg_ms:.3f} ms/iter ({1000/avg_ms:.1f} it/s)"
    )
    cleanup(sess)
