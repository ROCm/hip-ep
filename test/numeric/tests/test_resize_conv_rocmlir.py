#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Nearest integer upsample feeding a convolution, with rocMLIR forced on.

hip-fuse-rocmlir-anchor absorbs the resize into the rocMLIR convolution as an
expand_shape -> hip.expand -> collapse_shape view, so no resize kernel runs.
The reference is ONNX Runtime's CPU EP running the unfused Resize then Conv.
The 3D case builds a rank-8 view and the 2D case a rank-6 view.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from onnx import TensorProto, helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes

# Present only when LibHipCompiler is built with ENABLE_ROCMLIRTRITON.
# CompilerDriver.cpp emits it from the rocMLIR arm; a build without that
# arm ignores HIPDNN_EP_ROCMLIR and would pass on the ordinary kernels.
_ROCMLIR_BUILD_MARKER = b"rocMLIR kernel compilation failed"


def _make_resize_conv_model(
    in_shape: list[int],
    spatial_scale: int,
    out_channels: int,
    nearest_mode: str,
    use_sizes: bool,
    seed: int,
):
    rng = np.random.default_rng(seed)
    spatial = len(in_shape) - 2
    up_shape = in_shape[:2] + [d * spatial_scale for d in in_shape[2:]]
    kernel = [3] * spatial
    w = rng.uniform(-0.5, 0.5, [out_channels, in_shape[1], *kernel])
    b = rng.uniform(-0.5, 0.5, [out_channels])
    w_init = helper.make_tensor(
        "W", TensorProto.FLOAT16, w.shape, w.astype(np.float16).ravel()
    )
    b_init = helper.make_tensor(
        "B", TensorProto.FLOAT16, b.shape, b.astype(np.float16).ravel()
    )

    if use_sizes:
        resize_init = helper.make_tensor(
            "sizes", TensorProto.INT64, [len(up_shape)], up_shape
        )
        resize_inputs = ["X", "", "", "sizes"]
    else:
        scales = [1.0, 1.0] + [float(spatial_scale)] * spatial
        resize_init = helper.make_tensor(
            "scales", TensorProto.FLOAT, [len(scales)], scales
        )
        resize_inputs = ["X", "", "scales"]

    resize = helper.make_node(
        "Resize",
        resize_inputs,
        ["U"],
        mode="nearest",
        coordinate_transformation_mode="asymmetric",
        nearest_mode=nearest_mode,
    )
    conv = helper.make_node(
        "Conv",
        ["U", "W", "B"],
        ["Y"],
        kernel_shape=kernel,
        pads=[1] * (2 * spatial),
        strides=[1] * spatial,
    )
    out_shape = [in_shape[0], out_channels, *up_shape[2:]]
    x_info = helper.make_tensor_value_info("X", TensorProto.FLOAT16, in_shape)
    y_info = helper.make_tensor_value_info("Y", TensorProto.FLOAT16, out_shape)
    return make_model_from_nodes(
        [resize, conv],
        [x_info],
        [y_info],
        initializers=[resize_init, w_init, b_init],
        opset=13,
    )


def _dll_contains(path: Path, needle: bytes) -> bool:
    if not path.is_file():
        return False
    overlap = b""
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1 << 20)
            if not chunk:
                return False
            if needle in overlap + chunk:
                return True
            overlap = chunk[1 - len(needle) :]


def _require_rocmlir_build(request) -> None:
    dll = request.config.getoption("--ep-dll")
    if not dll or not _dll_contains(Path(dll), _ROCMLIR_BUILD_MARKER):
        pytest.skip(
            "EP was built without ENABLE_ROCMLIRTRITON; "
            "HIPDNN_EP_ROCMLIR would not run the absorb rewrite"
        )


def _assert_resize_absorbed(dump_dir: Path) -> None:
    dumps = sorted(dump_dir.glob("*.mlir"))
    if not dumps:
        pytest.fail(
            "HIPDNN_EP_IR_DUMP_PATH produced no IR, so the rocMLIR rewrite was not confirmed"
        )
    ir = "\n".join(path.read_text(encoding="utf-8", errors="replace") for path in dumps)
    # The tail dump starts after rocMLIR fusion. An absorbed upsample is a
    # hip.rocmlir dispatch (later wrap_rocmlir) and no resize op.
    if "hip.rocmlir" not in ir and "wrap_rocmlir" not in ir:
        pytest.fail(
            "compiled IR never dispatched rocMLIR, so the upsample absorb was not exercised"
        )
    if "hip.resize" in ir or "wrap_resize" in ir:
        pytest.fail(
            "compiled IR still contains a resize, so the upsample was not absorbed"
        )


@pytest.mark.parametrize(
    "in_shape,spatial_scale,nearest_mode,use_sizes",
    [
        # 2D, floor, explicit scales: rank-6 view.
        ([1, 8, 16, 16], 2, "floor", False),
        # 2D, round_prefer_floor at scale 2 agrees with floor on every sample.
        ([1, 8, 16, 16], 2, "round_prefer_floor", True),
        # 2D, scale 4 with floor.
        ([1, 4, 8, 8], 4, "floor", True),
        # 3D: rank-8 view.
        ([1, 4, 4, 8, 8], 2, "floor", False),
    ],
    ids=["2d_x2_floor", "2d_x2_round_prefer_floor", "2d_x4_floor", "3d_x2_floor"],
)
def test_nearest_upsample_conv_rocmlir(
    model_runner,
    monkeypatch,
    request,
    in_shape,
    spatial_scale,
    nearest_mode,
    use_sizes,
):
    _require_rocmlir_build(request)
    monkeypatch.setenv("HIPDNN_EP_ROCMLIR", "1")
    with tempfile.TemporaryDirectory(prefix="resize-conv-rocmlir-") as dump_dir:
        monkeypatch.setenv("HIPDNN_EP_IR_DUMP_PATH", str(Path(dump_dir) / "ir.mlir"))
        monkeypatch.setenv("HIPDNN_EP_IR_DUMP_AFTER_ONLY", "1")
        model = _make_resize_conv_model(
            in_shape,
            spatial_scale,
            out_channels=8,
            nearest_mode=nearest_mode,
            use_sizes=use_sizes,
            seed=11,
        )
        x = np.random.default_rng(5).uniform(-1, 1, in_shape).astype(np.float16)
        actual, expected = model_runner.run_sample(model, [x])
        _assert_resize_absorbed(Path(dump_dir))
    assert np.isfinite(actual[0]).all()
    compare_outputs(actual, expected, atol=2e-2, rtol=2e-2)
