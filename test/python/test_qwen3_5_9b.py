#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Qwen3.5-9B test suite — vision encoder + OGA text + OGA text+image E2E.

Three layers of coverage:

  1. Vision encoder (ORT-direct, `TestQwen3_5_9BVisionDynShape` + the
     synthetic patch-merger micro-test below) — exercises the
     dynshape `vision.onnx` end-to-end on MorphiZenEP. Pre-existing.

  2. OGA text-only generation (`TestQwen3_5_9BOGAText`, inherited
     from `BaseOGATests`) — text decoder + embedding sub-model run via
     OGA's standard generator loop on MorphiZenEP. No image inputs; the
     vision sub-session is loaded but never invoked.

  3. OGA text + image E2E (`TestQwen3_5_9BMultimodal`) — the MS Build
     demo path: load all three sub-sessions (embedding, vision, text)
     on MorphiZenEP, run the Qwen multimodal processor over a
     synthesized RGB image, then generate. The test runs in a
     subprocess with `HIPDNN_EP_DEBUG=1` so stderr captures the
     `[REAL] wrap_*` runtime markers, and asserts that wrappers
     belonging to each of the three sub-sessions appear — i.e. all
     three were claimed by MorphiZenEP rather than silently falling
     back to CPU.

The synthetic patch-merger micro-test below verifies that
`InferOnnxShapes`'s Reshape SSA-trace resolves `num_logical_patches =
num_patches / 4` via the `mult=0.25` slot through the entire chain:

  InferOnnxShapes  Reshape trace → DimOrigin{arg=0, dim=0, mult=0.25}
  C ABI             3-int64 triple, mult bit-cast through int64 slot
  metadata.proto    DimSource.mult = 0.25
  marshal_output    round(inputs[pixel_values].shape[0] * 0.25)

`num_patches` and `num_logical_patches` are deliberately distinct
dim_param strings (matches the real vision.onnx), so DimSource resolution
falls to priority-3 SSA trace — not name match. A regression in any link
of the chain produces a wrong output shape that the assertion catches.
"""

import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as nph
import pytest

from conftest import (
    BaseOGATests,
    ModelSpec,
    REPO_ROOT,
    create_ep_session,
    register_model_fixtures,
)

# ruff: noqa: F811


# Local model dir (vision.onnx + vision.onnx.data ~915 MB). Downloaded by
# huggingface_hub from amd/Qwen3.5-9B-rtn-int4-int8-128gs-fp16-onnx-gpu.
QWEN35_MODEL_DIR = (
    REPO_ROOT / "models" / "Qwen3.5-9B-rtn-int4-int8-128gs-fp16-onnx-gpu-2"
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
        assert out.shape == (self._expected_logical(self.grid_small), 4096), out.shape
        assert out.dtype == np.float16

    def test_vision_grid_large_same_session(self):
        """Same compiled DLL, larger num_patches — pool must grow on demand."""
        out = self._run(self.grid_large)
        assert out.shape == (self._expected_logical(self.grid_large), 4096), out.shape
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
            f"vision EP output diverges from CPU (cosine={cos:.6f}); expected >= 0.999"
        )


# ── Qwen3.5-9B OGA E2E (text-only + text+image) ──────────────────────────
#
# The release/msbuild merge brings the three EP capabilities Qwen 3.5 OGA
# E2E depends on:
#   * VLM embedding sub-model bit-exact (Equal scalar broadcast / ordered
#     NonZero / Slice partial-fill / ScatterND count_ptr — five coupled
#     fixes in feat/qwen35-9b-embedding-and-text).
#   * Dynamic-shape vision encoder with hip.loop + strided memrefCopy +
#     FixLoopAccumulatorOffset (dynshapes_v3).
#   * Dynamic Range/Reshape + RelaxMultiDynExpandShape + INT64
#     wrap_miopenOpTensor for the text decoder's mrope path
#     (feat/qwen35-9b-embedding-and-text).
#
# OGA's `qwen3_5` model_type (introduced upstream in PR #2019, already
# present in our AMDmoore fork) loads all three sub-models as one
# multimodal pipeline. Each sub-session gets MorphiZenEP wired in via
# `patch_genai_config_for_morphizen`, so a regression that breaks
# graph-claim on ANY sub-session aborts under HIPDNN_EP_STRICT=1
# (conftest default) instead of silently falling back to CPU.


def _make_qwen35_oga_text_prompt():
    """Conservative prompt tokens for OGA text-only generation tests.

    No image, no chat template — just a short token list that lands in
    Qwen 3.5's vocabulary range. EOS is shared with BOS / PAD
    (248044) so picking it as the prompt prefix is fine; OGA only
    stops on EOS in the MODEL OUTPUT, not in the prompt.
    """
    # bos + arbitrary mid-vocab tokens. Vocab size is 248320, so any int
    # under 248044 is safe. The list isn't decoded — only fed into
    # `Generator.append_tokens` to drive an end-to-end forward pass.
    return [248044, 1234, 5678, 9012, 3456, 7890, 12345]


QWEN35 = ModelSpec(
    name="qwen3_5_9b",
    model_dir=QWEN35_MODEL_DIR,
    onnx_file="text.onnx",
    data_files=["text.onnx.data"],
    extra_data_files=[
        "embedding.onnx",
        "embedding.onnx.data",
        "vision.onnx",
        "vision.onnx.data",
    ],
    # Locally re-exported model; never auto-fetched. The on-disk files are
    # the authoritative artifacts the test runs against; the upstream HF
    # repo `amd/Qwen3.5-9B-rtn-int4-int8-128gs-fp16-onnx-gpu` ships an older
    # variant that does NOT match this directory. If `Qwen3.5-9B-rtn-int4-
    # int8-128gs-fp16-onnx-gpu-2/` is absent, every test in this file
    # pytest.skip's with a clear message.
    auto_download=False,
    # Architecture knobs — only consumed by BaseORTTests (which we do NOT
    # subclass here; Qwen 3.5's text.onnx uses inputs_embeds + 3-rank
    # mrope position_ids that the input builders in conftest don't model).
    # Kept as placeholders so the ModelSpec validator is happy.
    num_layers=32,
    num_kv_heads=4,
    head_dim=256,
    has_position_ids=True,
    # bos == eos == pad == 248044 in Qwen 3.5's config. Filler tokens are
    # arbitrary mid-vocab ints; we never decode them.
    bos_token=248044,
    filler_tokens=[1234, 5678, 9012, 3456, 7890, 12345],
    oga_files=[
        "chat_template.jinja",
        "config.json",
        "genai_config.json",
        "model_config.json",
        "processor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ],
    # All three sub-sessions (embedding / vision / text) are expected to
    # claim+compile on MorphiZenEP. STRICT=1 turns any compile failure
    # into a hard abort with a stack trace — the right diagnostic for a
    # regression. If a real kernel gap turns up here, flip to False AND
    # file an issue identifying the missing op.
    oga_strict=True,
    # 9B model + multimodal pipeline → don't reuse OGA Model across
    # tests (each test creates / tears down its own Model to bound the
    # peak working set at one session's worth of constants).
    reuse_oga_default_model=False,
    markers={
        # CPU baseline for chunked-prefill accuracy needs a full
        # CPU-EP run of the 9B text decoder + 9B embedding pipeline.
        # That's minutes per token and would dominate the CI budget,
        # for a check that's already covered by the LLM-family
        # chunked-prefill tests (Llama 8B etc.).
        "test_oga_ep_chunked_prefill": pytest.mark.skip(
            reason=(
                "CPU baseline for 9B + embedding sub-model is too slow to "
                "include in the standard test budget. Chunked-prefill "
                "correctness is covered by Llama-8B et al."
            )
        ),
    },
)


(
    dynamic_model_path,
    fixed_decode_path,
    fixed_prefill_128_path,
    ep_dynamic_session,
    ep_fixed_decode_session,
    ep_fixed_prefill_128_session,
    oga_default_model,
) = register_model_fixtures(QWEN35)


class TestQwen3_5_9BOGAText(BaseOGATests):
    """Qwen 3.5 9B text-only OGA E2E on MorphiZenEP.

    Inherits the canonical 4 OGA tests (chunked_prefill is skipped above).
    No images — the vision sub-session is loaded by OGA's multimodal
    pipeline but never invoked by `append_tokens(...) → generate_next_token`.
    Exercise of the embedding sub-session is implicit: OGA routes
    `input_ids → embedding.onnx → text.onnx` even when no image_features
    are supplied (the embedding graph short-circuits the scatter path on
    empty image_features).
    """

    spec = QWEN35


# ── Qwen 3.5 OGA text + image E2E ────────────────────────────────────────
#
# The MS Build demo path. Runs in a subprocess for two reasons:
#  (1) HIPDNN_EP_DEBUG=1 is a process-startup-time decision in the EP DLL
#      (`hipdnn_ep_debug_enabled()` is a `static const bool` evaluated on
#      first call). A test that flips the env var mid-process can't enable
#      debug logging — and the EP DLL has already been loaded by the
#      conftest module-scope ORT/OGA setup. Spawning a fresh subprocess
#      with the env var pre-set is the only way to capture the
#      `[REAL] wrap_*` markers we use as proof of GPU dispatch.
#  (2) The 9B + multimodal pipeline holds ~10 GB resident at peak. Putting
#      it in a subprocess means the parent pytest process recovers all of
#      that memory the moment the test finishes, instead of leaking into
#      subsequent tests in the file.


_QWEN35_MULTIMODAL_WORKER = textwrap.dedent(
    r"""
    # -*- coding: utf-8 -*-
    import os, sys, json, tempfile, traceback
    from pathlib import Path

    MODEL_DIR = Path(sys.argv[1])
    EP_DLL = Path(sys.argv[2])
    IMAGE_PATH = Path(sys.argv[3])
    MAX_NEW = int(sys.argv[4])

    # Add EP + ROCm DLL dirs to PATH so the model.dll lld-link step finds
    # amdhip64.lib / MIOpen.lib at compile time. (The parent test already
    # set these but a child process inherits a snapshot; we re-add as a
    # defensive belt-and-braces.)
    repo_root = MODEL_DIR.parent.parent
    for d in (repo_root / "install" / "dist" / "bin", repo_root / "install" / "therock" / "bin"):
        if d.exists() and str(d) not in os.environ.get("PATH", ""):
            os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")

    try:
        import onnxruntime_genai as og
    except ImportError as e:
        print("WORKER_SKIP onnxruntime-genai missing:", e, file=sys.stderr)
        sys.exit(2)

    if not hasattr(og, "register_execution_provider_library"):
        print("WORKER_SKIP OGA build lacks register_execution_provider_library", file=sys.stderr)
        sys.exit(2)

    og.register_execution_provider_library("MorphiZenEP", str(EP_DLL))

    try:
        model = og.Model(str(MODEL_DIR))
    except RuntimeError as e:
        print(f"WORKER_FAIL og.Model load: {e}", file=sys.stderr)
        sys.exit(3)

    processor = model.create_multimodal_processor()
    tokenizer = og.Tokenizer(model)
    tokenizer_stream = processor.create_stream()
    images = og.Images.open(str(IMAGE_PATH))

    # Use the official chat template via apply_chat_template — pattern from
    # the upstream Olive recipe (microsoft/olive-recipes Qwen-Qwen3.5-9B
    # builtin/inference.py). The recipe-provided chat_template.jinja
    # appends "<think>\n" so the model starts in reasoning mode (Qwen 3.5
    # is a thinking-mode VLM). Hand-built templates omit this and put the
    # model in a degraded mode.
    messages = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text",
         "text": "Describe this image in one short sentence."},
    ]}]
    full_prompt = tokenizer.apply_chat_template(
        json.dumps(messages), add_generation_prompt=True)
    inputs = processor(full_prompt, images=images)

    params = og.GeneratorParams(model)
    # Cap total length so a runaway generation can't hang the test budget.
    params.set_search_options(max_length=512, do_sample=False)
    generator = og.Generator(model, params)
    generator.set_inputs(inputs)

    generated = []
    decoded = ""
    while not generator.is_done() and len(generated) < MAX_NEW:
        generator.generate_next_token()
        tok = int(generator.get_next_tokens()[0])
        generated.append(tok)
        try:
            decoded += tokenizer_stream.decode(tok)
        except Exception:
            pass  # decoder may stall on partial UTF-8; tokens are the truth

    # Emit a structured result line on stdout for the parent to parse.
    print("WORKER_OK " + json.dumps({"tokens": generated, "decoded": decoded}))
    """
)


# Real test image: photo of the Eiffel Tower. Used for the CPU baseline
# content assertion ("must produce 'eiffel' in the decoded text") AND for
# the EP-vs-CPU token-match test. A real photo is preferable to a
# synthesized one because:
#   * synthetic flat-color images push the model into a degenerate output
#     regime (it tends to either describe colors literally or hallucinate
#     a "cone-shaped object" — neither is a meaningful CPU baseline);
#   * the Eiffel Tower is unambiguous enough that an "identifies the
#     subject" assertion (case-insensitive substring match on "eiffel")
#     is a robust regression signal — a vision encoder that silently
#     produces NaN or all-zero patch tokens cannot accidentally land on
#     that word.
QWEN35_TEST_IMAGE = REPO_ROOT / "test" / "python" / "images" / "tower.jpg"


class TestQwen3_5_9BMultimodal:
    """Qwen 3.5 9B text + image E2E via OGA + MorphiZenEP.

    Skipped when the model files aren't present (gated AMD repo). Runs in
    a subprocess so HIPDNN_EP_DEBUG=1 takes effect at EP-DLL load time
    and we can grep the captured stderr for `[REAL] wrap_*` runtime
    dispatch markers as proof that the embedding, vision, AND text
    sub-sessions all executed kernels on the GPU (rather than silently
    falling back to CPU).
    """

    spec = QWEN35

    @pytest.fixture(scope="class")
    def workspace(self, tmp_path_factory, repo_root):
        # Required model files. Skip cleanly if any are missing — they're
        # multi-GB and only fetched explicitly by the user (the OGA suite
        # downloads them on first OGA test, but if you're cherry-picking
        # only this multimodal test we don't want a 12 GB surprise).
        for f in (
            self.spec.model_dir / x
            for x in (
                "text.onnx",
                "text.onnx.data",
                "embedding.onnx",
                "embedding.onnx.data",
                "vision.onnx",
                "vision.onnx.data",
                "genai_config.json",
                "chat_template.jinja",
                "tokenizer.json",
                "processor_config.json",
            )
        ):
            if not f.exists():
                pytest.skip(f"Qwen 3.5 9B file missing: {f.name}")

        from conftest import (
            patch_genai_config_for_morphizen,
            restore_genai_config,
            setup_oga_ep,
        )

        # Register the EP in the parent process so the subprocess inherits
        # the cached DLL location + PATH additions. Returns the absolute
        # ep_dll path the subprocess needs.
        _, ep_dll = setup_oga_ep(repo_root)
        patch_genai_config_for_morphizen(self.spec.model_dir, ep_dll)
        tmp_dir = tmp_path_factory.mktemp("qwen35_multimodal")
        if not QWEN35_TEST_IMAGE.exists():
            pytest.skip(f"test image missing: {QWEN35_TEST_IMAGE}")
        yield {
            "tmp_dir": tmp_dir,
            "image_path": QWEN35_TEST_IMAGE,
            "ep_dll": ep_dll,
            "repo_root": repo_root,
        }
        restore_genai_config(self.spec.model_dir)

    def _run_worker(self, workspace, max_new, *, debug=False):
        worker_py = workspace["tmp_dir"] / "worker.py"
        # Explicit UTF-8 — the worker template ships with a coding
        # declaration but Python's default text-write encoding on Windows
        # is cp1252, which will mangle any non-ASCII char in a comment
        # before the worker can even parse the declaration.
        worker_py.write_text(_QWEN35_MULTIMODAL_WORKER, encoding="utf-8")
        env = os.environ.copy()
        # STRICT=1 makes any sub-session graph-claim-then-compile-failure
        # a hard abort with a stack trace — the right signal for a real
        # regression instead of a silent CPU fallback.
        env["HIPDNN_EP_STRICT"] = "1"
        if debug:
            env["HIPDNN_EP_DEBUG"] = "1"
        else:
            env.pop("HIPDNN_EP_DEBUG", None)
        # THEROCK_DIST must be set without trailing whitespace — see the
        # CLAUDE.md "cmd.exe set quoting" gotcha. os.environ.copy()
        # already gives clean values; just ensure it's set.
        therock = workspace["repo_root"] / "install" / "therock"
        env.setdefault("THEROCK_DIST", str(therock))
        # Bound by a generous timeout: cold-cache EP compile of text.onnx
        # (~6.7 GB ONNX) plus generation can take 10+ min on a clean tree.
        # Warm cache: ~30 s.
        result = subprocess.run(
            [
                sys.executable,
                str(worker_py),
                str(self.spec.model_dir),
                str(workspace["ep_dll"]),
                str(workspace["image_path"]),
                str(max_new),
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=1200,
        )
        return result

    def test_oga_multimodal_runs(self, workspace):
        """Smoke test: image+text prompt produces at least a few tokens
        and the worker exits 0. No HIPDNN_EP_DEBUG — keeps stderr quiet
        so a failure here is unambiguous (any cosine drift / silent CPU
        fallback shows up as a slow run or as a strict-mode abort).
        """
        result = self._run_worker(workspace, max_new=5, debug=False)
        if result.returncode == 2:
            pytest.skip(result.stderr.strip().splitlines()[-1])
        assert result.returncode == 0, (
            f"Worker exited {result.returncode}.\n"
            f"--- stderr (tail) ---\n{result.stderr[-2000:]}"
        )
        # Parse the WORKER_OK JSON line. There may be other stdout noise
        # from OGA / ORT init — scan for the marker.
        ok_lines = [
            ln for ln in result.stdout.splitlines() if ln.startswith("WORKER_OK ")
        ]
        assert ok_lines, (
            f"Worker did not emit WORKER_OK line.\n"
            f"--- stdout (tail) ---\n{result.stdout[-2000:]}\n"
            f"--- stderr (tail) ---\n{result.stderr[-2000:]}"
        )
        import json as _json

        payload = _json.loads(ok_lines[-1][len("WORKER_OK ") :])
        assert len(payload["tokens"]) >= 1, (
            f"No tokens generated. Decoded: {payload['decoded']!r}"
        )

    def test_cpu_baseline_identifies_eiffel(self):
        """CPU baseline: ORT CPU EP + OGA + Qwen 3.5 9B must identify the
        Eiffel Tower in tower.jpg. NO MorphiZenEP in this test, no
        subprocess — runs in-process like
        `install/olive-recipes/Qwen-Qwen3.5-9B/builtin/inference.py`.

        Purpose: gate the rest of the suite on a known-good content-level
        baseline. If this fails, the issue is in the ORT / OGA / model
        files / pip environment — NOT in MorphiZenEP.

        Wall-clock: aligned with the Olive recipe (~25 s of CPU at
        ~15 tok/s for ~330 tokens of `<think>` reasoning + the final
        sentence, plus a few seconds of model load). Does NOT use the
        `workspace` fixture (which would load MorphiZenEP, patch
        genai_config for EP, and spawn a worker subprocess — each step
        adds seconds we don't need for a pure-CPU check).

        Assertion: case-insensitive substring match on "eiffel". The
        tower is the only globally-recognizable subject in the photo;
        a vision encoder producing NaN / all-zero patch tokens cannot
        accidentally hit that word.
        """
        import gc
        import json
        import shutil

        try:
            import onnxruntime_genai as og
        except ImportError:
            pytest.skip("onnxruntime-genai not installed")

        model_dir = self.spec.model_dir
        for fname in (
            "text.onnx",
            "text.onnx.data",
            "embedding.onnx",
            "embedding.onnx.data",
            "vision.onnx",
            "vision.onnx.data",
            "genai_config.json",
            "chat_template.jinja",
            "tokenizer.json",
            "processor_config.json",
        ):
            if not (model_dir / fname).exists():
                pytest.skip(f"Qwen 3.5 9B file missing: {fname}")
        if not QWEN35_TEST_IMAGE.exists():
            pytest.skip(f"test image missing: {QWEN35_TEST_IMAGE}")

        # The repo's genai_config.json may have been patched by an earlier
        # EP test (or by `patch_genai_config_for_morphizen` running for
        # another fixture in this file). Force CPU EP for this test by
        # zeroing provider_options on every sub-session, and restore the
        # original file in `finally` so subsequent EP tests are unaffected.
        cfg_path = model_dir / "genai_config.json"
        cpu_bak = model_dir / "genai_config.json.cpu_baseline_bak"
        shutil.copy2(cfg_path, cpu_bak)
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            for sub_key in ("decoder", "embedding", "vision"):
                sub = cfg["model"].get(sub_key)
                if isinstance(sub, dict):
                    sub.setdefault("session_options", {})["provider_options"] = []
            cfg_path.write_text(json.dumps(cfg, indent=4), encoding="utf-8")

            # Mirror the Olive inference.py flow exactly: load, build
            # chat-template prompt, processor, generator, decode.
            model = og.Model(str(model_dir))
            try:
                processor = model.create_multimodal_processor()
                tokenizer = og.Tokenizer(model)
                images = og.Images.open(str(QWEN35_TEST_IMAGE))
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {
                                "type": "text",
                                "text": "Describe this image in one short sentence.",
                            },
                        ],
                    }
                ]
                full_prompt = tokenizer.apply_chat_template(
                    json.dumps(messages), add_generation_prompt=True
                )
                inputs = processor(full_prompt, images=images)

                params = og.GeneratorParams(model)
                # max_length must exceed the image-tokenized prompt
                # (~319 tokens for tower.jpg) plus the expected ~350-token
                # thinking-mode response. 1024 leaves comfortable headroom.
                params.set_search_options(max_length=1024, do_sample=False)
                generator = og.Generator(model, params)
                generator.set_inputs(inputs)

                tokens = []
                while not generator.is_done():
                    generator.generate_next_token()
                    tokens.append(int(generator.get_next_tokens()[0]))

                decoded = tokenizer.decode(tokens)
            finally:
                # Release the 9 GB model before the next test runs.
                # `del locals()[name]` does not work — locals() is not
                # writable in CPython. Rebind each name to None so the
                # OGA C++ objects are released on the next gc cycle.
                model = processor = tokenizer = images = None
                inputs = params = generator = None
                gc.collect()
        finally:
            shutil.move(str(cpu_bak), str(cfg_path))

        print(f"\n  CPU baseline tokens: {len(tokens)}")
        print(f"  CPU baseline decoded:\n{decoded}")
        assert len(tokens) >= 1, (
            "OGA+CPU produced zero tokens — model load or generation "
            "step failed silently."
        )
        assert not all(t == 0 for t in tokens), (
            "CPU produced all-zero token sequence — likely NaN/zero "
            "logits. ORT setup is broken; gate the rest of the suite "
            "on this before debugging EP-side issues."
        )
        assert "eiffel" in decoded.lower(), (
            "CPU baseline did not identify the Eiffel Tower in tower.jpg. "
            "Either the model files are wrong, the processor/chat-template "
            "is misconfigured, or the vision sub-session produced garbage "
            "embeddings on CPU.\n"
            f"--- decoded ---\n{decoded}"
        )

    def _run_oga_in_provider_mode(
        self,
        workspace,
        provider,
        *,
        with_image,
        max_new,
        max_length=None,
    ):
        """Spawn an OGA worker patched to use either CPU EP or MorphiZenEP,
        with or without the test image, and return the generated token list.

        Used by both `test_oga_multimodal_tokens_match_cpu_xfail` (semantic
        comparison) and any future logit-level probes that want a controlled
        provider pin per pass.
        """
        worker_src = textwrap.dedent(r"""
            # -*- coding: utf-8 -*-
            import os, sys, json, shutil
            from pathlib import Path
            MODEL_DIR = Path(sys.argv[1]); EP_DLL = Path(sys.argv[2])
            IMAGE_PATH = Path(sys.argv[3]); MAX_NEW = int(sys.argv[4])
            PROVIDER = sys.argv[5]; WITH_IMAGE = sys.argv[6] == "1"
            MAX_LENGTH = int(sys.argv[7])

            # Re-add PATH defensively (child inherits a snapshot only).
            repo_root = MODEL_DIR.parent.parent
            for d in (repo_root / "install" / "dist" / "bin",
                      repo_root / "install" / "therock" / "bin"):
                if d.exists() and str(d) not in os.environ.get("PATH", ""):
                    os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")

            cfg_path = MODEL_DIR / "genai_config.json"
            backup = MODEL_DIR / "genai_config.json.match_cpu_bak"
            shutil.copy2(cfg_path, backup)
            cfg = json.load(open(cfg_path))
            if PROVIDER == "cpu":
                for sub_key in ("decoder", "embedding", "vision"):
                    sub = cfg["model"].get(sub_key)
                    if isinstance(sub, dict):
                        sub.setdefault("session_options", {})["provider_options"] = []
            else:
                opts = [{"MorphiZenEP": {}}]
                for sub_key in ("decoder", "embedding", "vision"):
                    sub = cfg["model"].get(sub_key)
                    if isinstance(sub, dict):
                        sub.setdefault("session_options", {})["provider_options"] = opts
            json.dump(cfg, open(cfg_path, "w"), indent=4)

            try:
                import onnxruntime_genai as og
                og.register_execution_provider_library("MorphiZenEP", str(EP_DLL))
                model = og.Model(str(MODEL_DIR))
                processor = model.create_multimodal_processor()
                tokenizer = og.Tokenizer(model)
                # Use the official chat template via apply_chat_template — this
                # is the pattern from the upstream Olive recipe
                # (microsoft/olive-recipes/Qwen-Qwen3.5-9B/builtin/inference.py).
                # The hand-built prompt strings we used previously were missing
                # the trailing "<think>\n" the model expects (Qwen 3.5 is a
                # thinking-mode model — the template appends 248068=<think>
                # and a newline so the model starts in reasoning mode).
                if WITH_IMAGE:
                    messages = [{"role": "user", "content": [
                        {"type": "image"},
                        {"type": "text",
                         "text": "Describe this image in one short sentence."},
                    ]}]
                else:
                    messages = [{"role": "user",
                                 "content": "Write one short sentence about Paris."}]
                full_prompt = tokenizer.apply_chat_template(
                    json.dumps(messages), add_generation_prompt=True)
                if WITH_IMAGE:
                    images = og.Images.open(str(IMAGE_PATH))
                    inputs = processor(full_prompt, images=images)
                else:
                    inputs = processor(full_prompt)
                params = og.GeneratorParams(model)
                params.set_search_options(max_length=MAX_LENGTH, do_sample=False)
                gen = og.Generator(model, params)
                gen.set_inputs(inputs)
                tokens = []
                while not gen.is_done() and len(tokens) < MAX_NEW:
                    gen.generate_next_token()
                    tokens.append(int(gen.get_next_tokens()[0]))
                try:
                    decoded = tokenizer.decode(tokens)
                except Exception as e:
                    decoded = f"<decode-error: {e}>"
                print("WORKER_OK " + json.dumps({"tokens": tokens,
                                                 "decoded": decoded,
                                                 "device_type": model.device_type}))
            finally:
                shutil.move(backup, cfg_path)
            """)
        worker_py = workspace["tmp_dir"] / f"worker_{provider}_{with_image}.py"
        worker_py.write_text(worker_src, encoding="utf-8")
        env = os.environ.copy()
        # STRICT=1 only matters for the EP run, but harmless on CPU.
        env["HIPDNN_EP_STRICT"] = "1"
        env.pop("HIPDNN_EP_DEBUG", None)
        env.setdefault(
            "THEROCK_DIST", str(workspace["repo_root"] / "install" / "therock")
        )
        # Pad max_length above max_new so the image-tokenized prompt
        # (~319 tokens for tower.jpg, less for text-only) still fits with
        # room for the requested generation. 768 is a comfortable cushion;
        # callers that want a tighter bound can pass `max_length` explicitly.
        eff_max_length = max_length if max_length is not None else max_new + 768
        result = subprocess.run(
            [
                sys.executable,
                str(worker_py),
                str(self.spec.model_dir),
                str(workspace["ep_dll"]),
                str(workspace["image_path"]),
                str(max_new),
                provider,
                "1" if with_image else "0",
                str(eff_max_length),
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=1200,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Worker (provider={provider} with_image={with_image}) "
                f"exited {result.returncode}.\n"
                f"--- stderr (tail) ---\n{result.stderr[-2000:]}"
            )
        ok_lines = [
            ln for ln in result.stdout.splitlines() if ln.startswith("WORKER_OK ")
        ]
        if not ok_lines:
            raise RuntimeError(
                f"Worker (provider={provider} with_image={with_image}) "
                f"did not emit WORKER_OK.\n"
                f"--- stdout (tail) ---\n{result.stdout[-2000:]}"
            )
        import json as _json

        payload = _json.loads(ok_lines[-1][len("WORKER_OK ") :])
        return payload["tokens"], payload.get("decoded", ""), payload["device_type"]

    # Greedy-decode token equivalence: CPU vs EP must match byte-for-byte
    # within the 5-token window. Was an `xfail` capturing the empty-image
    # regression (text-only produced all-zero tokens, VLM locked on one
    # special token from decode step 1). Root cause was Slice/ScatterND
    # treating an empty input (e.g. `image_features` shape[0]==0 on
    # text-only calls) as a hard null-arg failure rather than the no-op
    # ONNX prescribes; both wrappers now handle the empty case correctly
    # (slice → zero output, scatter_nd → output := data). See CLAUDE.md
    # "VLM embedding sub-model — Slice/ScatterND empty-input no-op" gotcha.
    def test_oga_multimodal_tokens_match_cpu(self, workspace):
        """Greedy-decode token sequences from CPU and MorphiZenEP must match
        (or be very close — we accept ≥ 80% match rate on 5 tokens to absorb
        fp16/quant noise around argmax ties).

        Tests both VLM and text-only modes. The CPU side is the slow one
        (~30 s on gfx1151 for 5 decode tokens), but we don't cache here —
        this test is the canonical "is the accuracy regression still
        present?" diagnostic and cache hides regression in the CPU run
        path itself.
        """
        cpu_tokens_vlm, _, cpu_dev = self._run_oga_in_provider_mode(
            workspace,
            "cpu",
            with_image=True,
            max_new=5,
        )
        assert cpu_dev.lower() == "cpu", f"CPU provider returned device_type={cpu_dev}"
        ep_tokens_vlm, _, ep_dev = self._run_oga_in_provider_mode(
            workspace,
            "ep",
            with_image=True,
            max_new=5,
        )
        assert ep_dev == "MorphiZenEP", f"EP provider returned device_type={ep_dev}"

        cpu_tokens_text, _, _ = self._run_oga_in_provider_mode(
            workspace,
            "cpu",
            with_image=False,
            max_new=5,
        )
        ep_tokens_text, _, _ = self._run_oga_in_provider_mode(
            workspace,
            "ep",
            with_image=False,
            max_new=5,
        )

        def _match(a, b):
            n = min(len(a), len(b))
            return sum(1 for i in range(n) if a[i] == b[i]) / n if n else 0.0

        rate_vlm = _match(cpu_tokens_vlm, ep_tokens_vlm)
        rate_text = _match(cpu_tokens_text, ep_tokens_text)
        print(
            f"\n  Token match (vlm):       CPU={cpu_tokens_vlm}  "
            f"EP={ep_tokens_vlm}  rate={rate_vlm:.0%}"
        )
        print(
            f"  Token match (text-only): CPU={cpu_tokens_text}  "
            f"EP={ep_tokens_text}  rate={rate_text:.0%}"
        )
        # Post-fix both modes match CPU exactly for these 5 tokens; assert
        # 100% so any future regression that flips even one argmax is
        # caught. If fp16-tie noise ever causes legitimate single-token
        # flips on this hardware, relax to >= 0.80 and add a note here
        # explaining which token went unstable.
        assert rate_vlm == 1.0, (
            f"VLM token match rate {rate_vlm:.0%} != 100% "
            f"(CPU={cpu_tokens_vlm} vs EP={ep_tokens_vlm})"
        )
        assert rate_text == 1.0, (
            f"Text-only token match rate {rate_text:.0%} != 100% "
            f"(CPU={cpu_tokens_text} vs EP={ep_tokens_text})"
        )

    def test_oga_multimodal_gpu_dispatch(self, workspace):
        """Run with HIPDNN_EP_DEBUG=1; assert that `[REAL] wrap_*` markers
        from kernel wrappers exercised by ALL THREE sub-sessions show up
        on stderr. Proof that the embedding (Equal/NonZero/ScatterND),
        vision (Conv/LayerNorm/MatMul/Loop), and text (matmul_nbits/GQA)
        all dispatched on the GPU rather than silently falling back to
        CPU — the canonical CLAUDE.md hygiene gotcha.

        Just 3 generated tokens — we're looking for the kernel dispatch
        fingerprint, not perf.
        """
        result = self._run_worker(workspace, max_new=3, debug=True)
        if result.returncode == 2:
            pytest.skip(result.stderr.strip().splitlines()[-1])
        assert result.returncode == 0, (
            f"Worker exited {result.returncode} under HIPDNN_EP_DEBUG=1.\n"
            f"--- stderr (tail) ---\n{result.stderr[-3000:]}"
        )
        stderr = result.stderr
        # Per-sub-session fingerprints. Each marker is chosen because it
        # appears in EXACTLY ONE sub-session's runtime dispatch trace, so a
        # positive count proves that sub-session is on the GPU:
        #   * embedding: scatter_nd is the image-placeholder substitution's
        #     final write. nonzero precedes it. Neither vision nor text
        #     dispatches them.
        #   * vision: wrap_gelu (Qwen vision MLP) and wrap_layer_normalization
        #     (the unfused-LN form vision uses). Text's LN is the
        #     wrap_skip_simplified_layer_norm fused form; text's activation
        #     is wrap_miopenActivationForward(sigmoid) for SwiGLU.
        #     PatchEmbedConvToGemm rewrites the patch embed to wrap_gemm
        #     (which is why "Conv" markers do NOT appear — kept off this
        #     list).
        #   * text: wrap_matmul_nbits (AWQ INT4) is the per-layer hot path
        #     and is text-only — vision and embedding use fp16 wrap_gemm.
        embedding_markers = ("wrap_scatter_nd", "wrap_nonzero")
        vision_markers = ("wrap_gelu", "wrap_layer_normalization")
        text_markers = ("wrap_matmul_nbits",)

        def _count(markers):
            return sum(stderr.count(f"[REAL] {m}") for m in markers)

        emb_n = _count(embedding_markers)
        vis_n = _count(vision_markers)
        txt_n = _count(text_markers)
        print("\n  Sub-session GPU dispatch counts:")
        print(f"    embedding (NonZero/ScatterND):                  {emb_n}")
        print(f"    vision    (Gelu/LayerNormalization):            {vis_n}")
        print(f"    text      (matmul_nbits):                       {txt_n}")
        # All three must show GPU activity. A zero count means the
        # sub-session silently fell back to CPU even though we patched
        # genai_config for MorphiZenEP on every sub-session — the
        # canonical CLAUDE.md silent-fallback gotcha.
        assert emb_n > 0, (
            f"No embedding sub-session GPU dispatch (looked for {embedding_markers}). "
            f"Embedding ran on CPU? --- stderr (tail) ---\n{stderr[-3000:]}"
        )
        assert vis_n > 0, (
            f"No vision sub-session GPU dispatch (looked for {vision_markers}). "
            f"Vision ran on CPU? --- stderr (tail) ---\n{stderr[-3000:]}"
        )
        assert txt_n > 0, (
            f"No text sub-session GPU dispatch (looked for {text_markers}). "
            f"Text ran on CPU? --- stderr (tail) ---\n{stderr[-3000:]}"
        )
