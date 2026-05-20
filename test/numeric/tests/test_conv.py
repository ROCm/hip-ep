#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Numeric verification for the Conv op (fp16 NCHW).

Covers one representative shape per workload class -- smoke, ResNet stem,
ResNet stage-3 3x3, 1x1, depthwise 3x3, ViT-L/14 224, SDXL UNet 1280,
ViT-B/16 2K -- enough to catch correctness regressions across the surface
real models hit, without padding the matrix with redundant variants.
Mirrors the shape list in bench/bench_conv.py (perf side); when adding a
row here, mirror it there and justify the workload class it covers.

The Conv backend the EP picks at runtime is driven by HIPDNN_EP_CONV
(read once per process by lib/Runtime/real/conv_dispatch.cpp). The
default is "ck for fp16, miopen for fp32". Set the env var in your shell
before invoking pytest to switch -- the framework intentionally does not
expose a CLI flag for it (see test/numeric/README.md "EP-DLL-consumed
knobs"). Cached references live under cache/<test-name>/ keyed on
model+inputs, so switching the EP-side backend does not invalidate the
CPU reference.

dtype: fp16. CK ships precompiled WMMA cshufflev3 instances for fp16
NHWGC on RDNA3+; fp32 NHWGC has no instances in TheRock's CK build
because CK_ENABLE_DL_KERNELS is undef'd. wrap_miopenConvolutionForward
handles both fp16 and fp32 via element_size_bytes -- pick fp16 here so
both backends are exercised by the same shape.
"""

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes


def _make_conv_model(
    N: int,
    C: int,
    H: int,
    W: int,
    K: int,
    ky: int,
    kx: int,
    stride: int,
    pad: int,
    dilation: int,
    group: int,
):
    """Build a one-node Conv ONNX model (fp16, constant weight initializer)."""
    in_shape = [N, C, H, W]
    w_shape = [K, C // group, ky, kx]

    # Output spatial dim per ONNX Conv spec (square stride/pad/dilation).
    ho = (H + 2 * pad - dilation * (ky - 1) - 1) // stride + 1
    wo = (W + 2 * pad - dilation * (kx - 1) - 1) // stride + 1
    out_shape = [N, K, ho, wo]

    rng = np.random.default_rng(123)
    # 0.1 scale on weights keeps fp16 conv outputs in-range across the
    # largest receptive fields below (14x14, 16x16) without saturating.
    w_data = (rng.standard_normal(w_shape, dtype=np.float32) * 0.1).astype(np.float16)
    w_init = numpy_helper.from_array(w_data, name="W")

    X = helper.make_tensor_value_info("X", TensorProto.FLOAT16, in_shape)
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT16, out_shape)
    node = helper.make_node(
        "Conv",
        inputs=["X", "W"],
        outputs=["Y"],
        kernel_shape=[ky, kx],
        strides=[stride, stride],
        pads=[pad, pad, pad, pad],
        dilations=[dilation, dilation],
        group=group,
    )
    return make_model_from_nodes([node], [X], [Y], initializers=[w_init])


def _make_conv_input(N: int, C: int, H: int, W: int) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.standard_normal([N, C, H, W], dtype=np.float32).astype(np.float16)


# (id, N, C, H, W, K, ky, kx, stride, pad, dilation, group)
# Kept verbatim from the legacy test/python/test_op_conv.py matrix.

# Trivial CPU cost -- no point caching, reference="cpu" each run.
SMALL_SHAPES = [
    ("smoke_3x3", 1, 3, 32, 32, 16, 3, 3, 1, 1, 1, 1),
    ("conv1x1", 1, 16, 32, 32, 32, 1, 1, 1, 0, 1, 1),
]

# Expensive CPU references (rn50_s3_3x3 onwards is 100M+ MACs on CPU) --
# reference="cache" pays off here. First run computes once and persists;
# every subsequent run loads from disk.
LARGE_SHAPES = [
    ("rn50_s3_3x3", 1, 256, 14, 14, 256, 3, 3, 1, 1, 1, 1),
    ("rn50_stem", 1, 3, 224, 224, 64, 7, 7, 2, 3, 1, 1),
    ("vit_l14_224", 1, 3, 224, 224, 1024, 14, 14, 14, 0, 1, 1),
    ("sdxl_unet_1280", 1, 1280, 16, 16, 1280, 3, 3, 1, 1, 1, 1),
    ("2k_vit_b16", 1, 3, 2048, 2048, 768, 16, 16, 16, 0, 1, 1),
]


class TestConv:
    @pytest.mark.parametrize(
        "name,N,C,H,W,K,ky,kx,stride,pad,dilation,group",
        SMALL_SHAPES,
        ids=[row[0] for row in SMALL_SHAPES],
    )
    def test_conv_small(
        self,
        model_runner,
        name,
        N,
        C,
        H,
        W,
        K,
        ky,
        kx,
        stride,
        pad,
        dilation,
        group,
    ):
        """Small Conv sanity. CPU cost is trivial -- skip the cache."""
        model = _make_conv_model(N, C, H, W, K, ky, kx, stride, pad, dilation, group)
        x = _make_conv_input(N, C, H, W)
        actual, expected = model_runner.run_sample(model, [x], reference="cpu")
        compare_outputs(actual, expected, atol=5e-2, rtol=1e-2)

    @pytest.mark.parametrize(
        "name,N,C,H,W,K,ky,kx,stride,pad,dilation,group",
        LARGE_SHAPES,
        ids=[row[0] for row in LARGE_SHAPES],
    )
    def test_conv_large(
        self,
        model_runner,
        name,
        N,
        C,
        H,
        W,
        K,
        ky,
        kx,
        stride,
        pad,
        dilation,
        group,
    ):
        """Production-shape Conv. CPU reference cached -- first run is slow,
        subsequent runs are instant."""
        model = _make_conv_model(N, C, H, W, K, ky, kx, stride, pad, dilation, group)
        x = _make_conv_input(N, C, H, W)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=5e-2, rtol=1e-2)

    @pytest.mark.xfail(
        reason=(
            "MIOpen rejects group == C depthwise with 'Invalid filter channel "
            "number'; CK handles it fine. strict=False so xpass on the CK "
            "backend still counts as success."
        ),
        strict=False,
    )
    def test_conv_depthwise_3x3(self, model_runner):
        """Depthwise 3x3 where group == C -- the trickiest dispatch case."""
        N, C, H, W, K, ky, kx = 1, 16, 16, 16, 16, 3, 3
        stride, pad, dilation, group = 1, 1, 1, 16
        model = _make_conv_model(N, C, H, W, K, ky, kx, stride, pad, dilation, group)
        x = _make_conv_input(N, C, H, W)
        actual, expected = model_runner.run_sample(model, [x], reference="cpu")
        compare_outputs(actual, expected, atol=5e-2, rtol=1e-2)
