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
import pytest

from conftest import REPO_ROOT, create_ep_session


# Local model dir (vision.onnx + vision.onnx.data ~915 MB). Downloaded by
# huggingface_hub from amd/Qwen3.5-9B-rtn-int4-int8-128gs-fp16-onnx-gpu.
QWEN35_MODEL_DIR = (
    REPO_ROOT / "models" / "Qwen3.5-9B-rtn-int4-int8-128gs-fp16-onnx-gpu"
)


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


# ── Qwen3.5-9B vision encoder dynshape coverage ──────────────────────────
#
# vision.onnx is the Qwen-style image encoder for the Qwen3.5-9B VLM (2073
# nodes; one Conv patch-embedding + 27 onnx.Loop ops + LayerNorm/Gelu/Gemm
# transformer stack + 2x2 patch merger). Its variable input dim is
# `num_patches` (the flattened post-patchify sequence length); the
# accompanying `image_grid_thw` input is shape-fixed at `[1, 3]` (one image
# per Compute call). Output is `image_features [num_logical_patches, 4096]`
# where `num_logical_patches = num_patches / 4` from the 2x2 spatial merger
# (DimSource priority-3 SSA trace through the Reshape divide-by-K rule
# already exercised by the synthetic micro-test above).
#
# Tests mirror TestGemma3_4BVisionDynShape from test_gemma3_4b.py: one
# session, monotonically growing input sizes to exercise the pool grow path,
# plus a CPU cosine check. The "two images in one call" pattern from gemma3
# does NOT apply here — `image_grid_thw` is static at `[1, 3]`. Instead we
# vary `num_patches` between calls.


def _qwen35_grid_inputs(grid_thw, seed=0):
    """Synthesize (pixel_values, image_grid_thw) for a given [t, h, w] grid.

    `num_patches = t * h * w` (Qwen flattens the post-patchify grid). Values
    kept small-magnitude fp16 for numerical stability through 55 LayerNorms
    in the encoder.
    """
    t, h, w = grid_thw
    n_patches = t * h * w
    rng = np.random.default_rng(seed)
    return {
        "pixel_values": (rng.standard_normal((n_patches, 1536)) * 0.1).astype(
            np.float16
        ),
        "image_grid_thw": np.array([grid_thw], dtype=np.int64),
    }


class TestQwen3_5_9BVisionDynShape:
    """Single EP session for vision.onnx must handle any num_patches.

    Tests are ordered (pytest preserves declaration order within a class):
    each builds on the cached session, growing pool / input shapes
    monotonically so we exercise the grow-on-demand path explicitly.

    Reference expected output dim: `num_logical_patches = num_patches / 4`
    (Qwen 2x2 patch merger, identical to the synthetic test above).
    """

    @classmethod
    def setup_class(cls):
        import gc

        import onnxruntime as ort

        from conftest import register_morphizen_ep

        gc.collect()
        vision_path = QWEN35_MODEL_DIR / "vision.onnx"
        if not vision_path.exists():
            pytest.skip(f"vision.onnx not present at {vision_path}")
        devices = register_morphizen_ep(REPO_ROOT)
        if not devices:
            pytest.skip("MorphiZen EP not found - run build.py first")
        so = ort.SessionOptions()
        so.add_provider_for_devices(devices, {})
        cls.sess = ort.InferenceSession(str(vision_path), sess_options=so)
        # Two grid sizes used across tests. Both even-h, even-w so the 2x2
        # merger output dim is an integer.
        cls.grid_small = [2, 8, 8]  # num_patches=128 → num_logical=32
        cls.grid_large = [2, 12, 12]  # num_patches=288 → num_logical=72

    @classmethod
    def teardown_class(cls):
        import gc

        if hasattr(cls, "sess"):
            del cls.sess
        gc.collect()

    def _run(self, grid_thw, seed=0):
        return self.sess.run(None, _qwen35_grid_inputs(grid_thw, seed=seed))[0]

    def _expected_logical(self, grid_thw):
        t, h, w = grid_thw
        # spatial 2x2 merger only (temporal kept) → outputs = t * (h/2) * (w/2)
        # Conservative: assume Qwen-VL convention of /4 over total patches.
        return (t * h * w) // 4

    def test_vision_grid_small(self):
        out = self._run(self.grid_small)
        assert out.shape == (self._expected_logical(self.grid_small), 4096), (
            out.shape
        )
        assert out.dtype == np.float16

    def test_vision_grid_large_same_session(self):
        """Same compiled DLL, larger num_patches — pool must grow on demand."""
        out = self._run(self.grid_large)
        assert out.shape == (self._expected_logical(self.grid_large), 4096), (
            out.shape
        )
        assert out.dtype == np.float16

    def test_vision_re_run_grid_small_bit_identical(self):
        """After running larger grid, going back must give bit-identical
        output to the first small-grid call. Guards against cross-call state
        leakage (pool slot recycling without zero, autotune cache key
        omissions, etc.)."""
        first = self._run(self.grid_small)
        _ = self._run(self.grid_large)
        second = self._run(self.grid_small)
        assert first.shape == second.shape
        assert np.array_equal(first.view(np.uint16), second.view(np.uint16)), (
            "two identical num_patches calls through the same session must "
            "be bit-identical"
        )

    def test_vision_re_run_same_seed_bit_identical(self):
        """Two back-to-back calls with identical input must produce
        bit-identical output. Stronger than re-run-after-larger — catches
        pure determinism bugs (e.g. unzeroed scratch buffers read by a
        reduction)."""
        a = self._run(self.grid_small, seed=42)
        b = self._run(self.grid_small, seed=42)
        assert np.array_equal(a.view(np.uint16), b.view(np.uint16)), (
            "two identical-input calls must produce bit-identical output"
        )

    def test_vision_grid_small_matches_cpu_cosine(self):
        """EP output for grid_small must agree with CPU baseline within
        3 nines of cosine. The other tests pin shape and call-to-call
        bit-equality but never compare VALUE against CPU — a kernel that
        produces consistent-but-wrong outputs would slip past all of them."""
        import onnxruntime as ort

        vision_path = QWEN35_MODEL_DIR / "vision.onnx"
        so = ort.SessionOptions()
        cpu_sess = ort.InferenceSession(
            str(vision_path),
            sess_options=so,
            providers=["CPUExecutionProvider"],
        )
        try:
            inputs = _qwen35_grid_inputs(self.grid_small)
            ep_out = self.sess.run(None, inputs)[0]
            cpu_out = cpu_sess.run(None, inputs)[0]
        finally:
            del cpu_sess

        assert ep_out.shape == cpu_out.shape, (ep_out.shape, cpu_out.shape)
        ep_f = ep_out.astype(np.float32).flatten()
        cpu_f = cpu_out.astype(np.float32).flatten()
        finite = np.isfinite(ep_f) & np.isfinite(cpu_f)
        ep_f, cpu_f = ep_f[finite], cpu_f[finite]
        assert ep_f.size > 0, "no finite outputs to compare"
        cos = float(
            np.dot(ep_f, cpu_f) / (np.linalg.norm(ep_f) * np.linalg.norm(cpu_f))
        )
        assert cos >= 0.999, (
            f"vision EP output diverges from CPU (cosine={cos:.6f}); "
            "expected >= 0.999"
        )
