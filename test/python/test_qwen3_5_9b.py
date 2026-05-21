#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Qwen3.5-9B vision dynshape micro-test.

Single-test placeholder for the full per-model test (which will follow the
ModelSpec / BaseORTTests / BaseOGATests pattern in conftest once the EP
covers the vision encoder's full op set — today blocked by Loop/If
support).

Synthesizes the specific ONNX pattern Qwen3.5 vision uses for its 2x2
patch merger — a `[num_patches, hidden]` Reshape into
`[num_patches/4, 4*hidden]` — and verifies the EP's DimSource SSA-origin
trace resolves the output dim via `mult=0.25` end-to-end:

  InferOnnxShapes  Reshape trace → DimOrigin{arg=0, dim=0, mult=0.25}
  C ABI             3-int64 triple, mult bit-cast through int64 slot
  metadata.proto    DimSource.mult = 0.25
  marshal_output    round(inputs[pixel_values].shape[0] * 0.25)

`num_patches` and `num_logical_patches` are deliberately distinct
dim_param strings (matches the real vision.onnx), so DimSource resolution
falls to priority-3 SSA trace — not name match. A regression in any link
of the chain produces a wrong output shape that the assertion catches.
"""

import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as nph

from conftest import REPO_ROOT, create_ep_session


def test_qwen_vision_patch_merger_dynshape():
    n_in, n_out = 64, 256  # outOther/inOther = 4 → mult = 0.25
    shape_const = nph.from_array(
        np.array([-1, n_out], dtype=np.int64), name="shape_const"
    )
    nodes = [
        oh.make_node(
            "Reshape",
            ["pixel_values", "shape_const"],
            ["image_features"],
            name="patch_merger",
        ),
    ]
    graph = oh.make_graph(
        nodes=nodes,
        name="qwen_vision_patch_merger_synthetic",
        inputs=[
            oh.make_tensor_value_info(
                "pixel_values",
                onnx.TensorProto.FLOAT16,
                ["num_patches", n_in],
            )
        ],
        outputs=[
            oh.make_tensor_value_info(
                "image_features",
                onnx.TensorProto.FLOAT16,
                ["num_logical_patches", n_out],
            )
        ],
        initializer=[shape_const],
    )
    model = oh.make_model(
        graph,
        opset_imports=[oh.make_opsetid("", 20)],
        ir_version=10,
        producer_name="test_qwen3_5_9b",
    )
    onnx.checker.check_model(model)

    tmpdir = Path(tempfile.mkdtemp(prefix="qwen35_dynshape_"))
    model_path = tmpdir / "patch_merger.onnx"
    onnx.save(model, str(model_path))

    sess = create_ep_session(str(model_path), REPO_ROOT)
    rng = np.random.RandomState(0)

    # Probe three input sizes that all match the Qwen patch-merger
    # invariant num_logical_patches = num_patches / 4.
    for n_patches in [16, 64, 256]:
        inputs = {
            "pixel_values": (rng.randn(n_patches, n_in) * 0.1).astype(np.float16),
        }
        out = sess.run(None, inputs)
        actual = out[0].shape[0]
        expected = n_patches // 4
        assert actual == expected, (
            f"Patch-merger divisor regression: input num_patches="
            f"{n_patches} expected output dim[0]={expected} "
            f"(num_patches/4), got {actual}. Likely cause: "
            f"`InferOnnxShapes` Reshape SSA-trace dropped the divide-by-K "
            f"branch, OR the C ABI bit-cast truncated `mult` to int, OR "
            f"`marshal_output_tensors` skipped `round(input * mult)`."
        )
