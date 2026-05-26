#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Gemma3-4B VLM test suite.

Model: amd/gemma3-4b-it-rtn-int4-128gs-fp16-onnx-gpu (gated AMD repo).
RTN INT4 g128, 34 layers, 4 KV heads (HPG=2), head_dim=256, hidden=2560.
Files live under an `Upload gemma3-4b-it ONNX files/` subdir which we flatten
via ModelSpec.hf_subdir.

Gemma3 is a multi-session VLM pipeline (embedding.onnx + vision.onnx +
text.onnx). The ORT-direct accuracy tests exercise text.onnx ONLY — input
to the text decoder is `inputs_embeds` (random fp16), NOT `input_ids`. The
two callable overrides on the spec (`prefill_input_fn`, `decode_input_fn`)
plug those builders into `BaseORTTests` so this file is just a spec
declaration — no custom test class.

Vision (~840 MB) is downloaded but not exercised — OGA's gemma3 model type
fails to load if any pipeline file referenced by genai_config is missing.
`_normalize_gemma3_embedding` strips the dangling `image_features` initializer
from embedding.onnx so ORT binds it as a graph input at runtime (matches
upstream MS Gemma-3-ONNX behavior).

Decode tests use zero past KV (see `make_zero_kv_cache` in conftest) — no
gemma3-specific fixture needed any more.

Test coverage is provided by `BaseORTTests` (5 tests) + `BaseOGATests`
(4 tests) in conftest.py — see CLAUDE.md "Python Performance Tests".
"""

import numpy as np
import pytest

from conftest import (
    BaseOGATests,
    BaseORTTests,
    ModelSpec,
    REPO_ROOT,
    register_model_fixtures,
)

# ruff: noqa: F811


HIDDEN_SIZE = 2560
NUM_LAYERS = 34
NUM_KV_HEADS = 4
HEAD_DIM = 256


def _normalize_gemma3_embedding(spec):
    """Drop the empty `image_features` initializer from embedding.onnx.

    The AMD repo ships embedding.onnx with `image_features` declared as BOTH
    a graph input AND an empty initializer (dims=[0,0,2560], no raw data).
    OGA's gemma3 model type interprets the initializer as a constant default
    and skips binding `image_features` at runtime. The MorphiZen EP still
    exposes `image_features` as a graph input, so ORT errors out at Compute()
    with "Model input was not found: image_features". Upstream MS
    onnxruntime/Gemma-3-ONNX uses graph-input-only — we match that pattern by
    removing the initializer. Idempotent.

    NOTE: vision.onnx is NOT modified here. The AS-SHIPPED form ships output
    `image_features` with dim_params `[num_image_tokens,
    MatMulimage_features_dim_1, 2560]` that don't match the input
    `num_images` dim_param. The EP handles this via the
    `InferOnnxShapes` pre-lowering pass + `DimSource.static_value` channel
    (compiler tells the EP the refined static shape for symbolic dims that
    don't resolve via input lookup). No on-disk or in-memory model
    modification is needed.
    """
    import onnx

    embed_path = spec.model_dir / "embedding.onnx"
    m = onnx.load(str(embed_path), load_external_data=False)
    init_names = {t.name for t in m.graph.initializer}
    input_names = {i.name for i in m.graph.input}
    overlap = init_names & input_names
    if not overlap:
        return

    new_initializers = [t for t in m.graph.initializer if t.name not in overlap]
    del m.graph.initializer[:]
    m.graph.initializer.extend(new_initializers)
    onnx.save(m, str(embed_path), save_as_external_data=False)
    print(
        f"  Normalized {embed_path.name}: dropped initializers {sorted(overlap)} "
        "(graph-input-only — matches MS upstream)"
    )


# ── Gemma3 inputs_embeds-based input builders ──────────────────────────────
#
# Both functions match the signatures BaseORTTests passes through
# `spec.build_*` — drop-in replacements for the default input_ids builders.


def _gemma3_prefill_input_fn(cfg, prefill_len, max_seq_len):
    """Random fp16 `inputs_embeds` of `prefill_len` positions + zero past KV.

    Seeded so CPU and EP get bit-identical inputs. Magnitudes scaled by
    0.1 to keep activations through 34 fp16 layers with q_norm/k_norm in
    a numerically stable range.
    """
    rng = np.random.default_rng(0)
    inputs = {
        "inputs_embeds": (
            rng.standard_normal((1, prefill_len, HIDDEN_SIZE)) * 0.1
        ).astype(np.float16),
        "attention_mask": np.zeros((1, max_seq_len), dtype=np.int64),
        "position_ids": np.arange(prefill_len, dtype=np.int64).reshape(1, -1),
    }
    inputs["attention_mask"][0, :prefill_len] = 1
    for i in range(cfg.num_kv_layers):
        inputs[f"past_key_values.{i}.key"] = np.zeros(
            (1, cfg.num_kv_heads, max_seq_len, cfg.head_dim), dtype=np.float16
        )
        inputs[f"past_key_values.{i}.value"] = np.zeros(
            (1, cfg.num_kv_heads, max_seq_len, cfg.head_dim), dtype=np.float16
        )
    return inputs


def _gemma3_decode_input_fn(cfg, position, kv_cache, max_seq_len):
    """One-token `inputs_embeds` decode at `position`, with past from kv_cache."""
    rng = np.random.default_rng(position + 1)  # position-stable seed
    inputs = {
        "inputs_embeds": (rng.standard_normal((1, 1, HIDDEN_SIZE)) * 0.1).astype(
            np.float16
        ),
        "attention_mask": np.zeros((1, max_seq_len), dtype=np.int64),
        "position_ids": np.array([[position]], dtype=np.int64),
    }
    inputs["attention_mask"][0, : position + 1] = 1
    for i in range(cfg.num_kv_layers):
        inputs[f"past_key_values.{i}.key"] = kv_cache[(i, "key")]
        inputs[f"past_key_values.{i}.value"] = kv_cache[(i, "value")]
    return inputs


GEMMA3 = ModelSpec(
    name="gemma3_4b",
    model_dir=REPO_ROOT / "models" / "gemma3-4b-it-rtn-int4-128gs-fp16-onnx-gpu",
    onnx_file="text.onnx",
    data_files=["text.onnx.data"],
    extra_data_files=[
        "embedding.onnx",
        "embedding.onnx.data",
        # Vision (~840 MB) is not exercised but must be present — OGA's
        # gemma3 model type fails to load if any pipeline file referenced by
        # genai_config is missing.
        "vision.onnx",
        "vision.onnx.data",
    ],
    hf_repo="amd/gemma3-4b-it-rtn-int4-128gs-fp16-onnx-gpu",
    hf_subdir="Upload gemma3-4b-it ONNX files",
    num_layers=NUM_LAYERS,
    num_kv_heads=NUM_KV_HEADS,
    head_dim=HEAD_DIM,
    has_position_ids=True,
    bos_token=2,
    filler_tokens=[1234, 5678, 9012, 3456, 7890, 12345],
    oga_files=[
        "added_tokens.json",
        "chat_template.jinja",
        "genai_config.json",
        "processor_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
    ],
    normalize_onnx_hook=_normalize_gemma3_embedding,
    prefill_input_fn=_gemma3_prefill_input_fn,
    decode_input_fn=_gemma3_decode_input_fn,
    # OGA gemma3 model_type loads embedding+vision+text as one multi-session
    # pipeline. Sub-sessions claim+fail by design (NonZero / ScatterND etc.
    # have no HIP converters) so MorphiZenEP provides HipDataTransferImpl
    # while ORT falls back to CPU for compute. HIPDNN_EP_STRICT=1 would
    # abort on those failures — OGA tests opt out via oga_strict=False.
    # ORT-direct tests are unaffected (only text.onnx → STRICT=1 stays).
    oga_strict=False,
    markers={
        "test_oga_ep_chunked_prefill": pytest.mark.skip(
            reason=(
                "Chunked-prefill accuracy needs a CPU baseline that runs the full "
                "embedding+text pipeline. The shared run_cpu_reference_generation "
                "helper uses input_ids on a single ONNX, which doesn't apply to "
                "gemma3 (decoder takes inputs_embeds, embedding lookup is in a "
                "separate ONNX). The OGA-CPU alternative would be ~minutes for a "
                "4B model and exceed test budget."
            )
        )
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
) = register_model_fixtures(GEMMA3)


class TestGemma3_4BORT(BaseORTTests):
    spec = GEMMA3


class TestGemma3_4BOGA(BaseOGATests):
    spec = GEMMA3


# ── Gemma3 vision encoder dyn-shape coverage ─────────────────────────────
#
# vision.onnx is a separate ONNX file (the SigLIP image encoder + multimodal
# projector) loaded by OGA's gemma3 pipeline alongside text.onnx. Its sole
# symbolic input dim is `num_images`; H/W are fixed at 896. The
# `RefineReshapeOutputType` + companion shape-inference patterns (added for
# this model family in CLAUDE.md "Dynamic-shape ViT / vision encoder
# support") let MorphiZenEP compile vision.onnx end-to-end and reuse the
# same compiled DLL for any num_images batch size at runtime — verified
# below.
#
# Output values are currently all-NaN vs. ORT CPU finite (known correctness
# bug, see CLAUDE.md). We assert SHAPE behavior + bit-pattern stability —
# the dyn-shape compile path is independent of the correctness issue, and
# regressing the shape behavior would silently break a future fix.


class TestGemma3_4BVisionDynShape:
    """Single EP session for vision.onnx must handle any `num_images` batch.

    Tests are ordered (pytest preserves declaration order within a class):
    each builds on the cached session, growing pool / input shapes
    monotonically so we exercise the grow-on-demand path explicitly.
    """

    spec = GEMMA3

    @classmethod
    def setup_class(cls):
        import gc

        import onnxruntime as ort

        from conftest import REPO_ROOT, register_morphizen_ep

        gc.collect()
        devices = register_morphizen_ep(REPO_ROOT)
        if not devices:
            pytest.skip("MorphiZen EP not found - run build.py first")
        vision_path = cls.spec.model_dir / "vision.onnx"
        if not vision_path.exists():
            pytest.skip(f"vision.onnx not present at {vision_path}")
        # Ensure the dim_param normalization has run (BaseORTTests fixtures
        # invoke `normalize_onnx_hook` on model setup; this class skips that
        # path so apply the hook directly).
        if cls.spec.normalize_onnx_hook is not None:
            cls.spec.normalize_onnx_hook(cls.spec)
        so = ort.SessionOptions()
        so.add_provider_for_devices(devices, {})
        cls.sess = ort.InferenceSession(str(vision_path), sess_options=so)
        # Two deterministic fp16 images (small magnitude for fp16 stability).
        rng = np.random.default_rng(0)
        cls.img1 = (rng.standard_normal((1, 3, 896, 896)) * 0.1).astype(np.float16)
        cls.img2 = (rng.standard_normal((1, 3, 896, 896)) * 0.1).astype(np.float16)

    @classmethod
    def teardown_class(cls):
        import gc

        if hasattr(cls, "sess"):
            del cls.sess
        gc.collect()

    def _run(self, pixel_values):
        return self.sess.run(None, {"pixel_values": pixel_values})[0]

    def test_vision_num_images_1(self):
        """Output shape must adapt to num_images=1."""
        out = self._run(self.img1)
        assert out.shape == (1, 256, 2560), out.shape
        assert out.dtype == np.float16

    def test_vision_num_images_2_same_session(self):
        """Same compiled DLL, larger batch — pool must grow on demand."""
        both = np.concatenate([self.img1, self.img2], axis=0)
        out = self._run(both)
        assert out.shape == (2, 256, 2560), out.shape
        assert out.dtype == np.float16

    def test_vision_re_run_num_images_1_bit_identical(self):
        """After running at num_images=2, going back to num_images=1 must
        give bit-identical output to the first num_images=1 call. Guards
        against cross-batch state leakage (e.g. pool slot recycling
        without zeroing, autotune cache key omissions, etc.)."""
        first = self._run(self.img1)
        # Force the pool to have grown (re-run num_images=2 first).
        _ = self._run(np.concatenate([self.img1, self.img2], axis=0))
        second = self._run(self.img1)
        assert first.shape == second.shape == (1, 256, 2560)
        # Use bit-pattern equality so NaN==NaN works (current state).
        assert np.array_equal(first.view(np.uint16), second.view(np.uint16)), (
            "two identical num_images=1 runs through the same session must be bit-identical"
        )

    def test_vision_num_images_2_twin_input_rows_identical(self):
        """num_images=2 with TWO COPIES of the same image must produce two
        identical output rows. Guards against batch-axis bugs (e.g. an op
        that confuses batch and channel strides)."""
        twin = np.concatenate([self.img1, self.img1], axis=0)
        out = self._run(twin)
        assert out.shape == (2, 256, 2560), out.shape
        row0 = out[0].view(np.uint16)
        row1 = out[1].view(np.uint16)
        assert np.array_equal(row0, row1), (
            "two identical input images must produce identical output rows"
        )

    def test_vision_cross_batch_determinism(self):
        """row 0 of a num_images=2 run with [img1, img2] must equal the
        full output of a num_images=1 run on img1 alone. Stronger than
        twin-row identity — checks that batch processing does not leak
        information across rows."""
        single = self._run(self.img1)
        both = np.concatenate([self.img1, self.img2], axis=0)
        batched = self._run(both)
        assert np.array_equal(single[0].view(np.uint16), batched[0].view(np.uint16)), (
            "num_images=2 row 0 must match num_images=1 output for the same image"
        )

    def test_vision_num_images_1_matches_cpu_cosine(self):
        """EP output for num_images=1 must agree with CPU baseline within
        3 nines of cosine. The other tests in this class pin shape and
        cross-batch bit-equality but never compare VALUE against CPU —
        a kernel that produces consistent-but-wrong outputs (e.g.
        a Reshape decomposition that loses elements, a broadcast-Div
        that reads OOB and lands on a stable garbage pattern) would slip
        past all of them. Cosine threshold 0.999 chosen vs the measured
        0.999987 on this hardware so 3 nines of headroom absorb future
        fp16 / hipBLASLt drift without false failures.
        """
        import onnxruntime as ort

        vision_path = self.spec.model_dir / "vision.onnx"
        # Standalone CPU session — the class's `cls.sess` is the EP one;
        # tearing it down would invalidate the other tests in this class.
        so = ort.SessionOptions()
        cpu_sess = ort.InferenceSession(
            str(vision_path), sess_options=so, providers=["CPUExecutionProvider"]
        )
        try:
            ep_out = self._run(self.img1)
            cpu_out = cpu_sess.run(None, {"pixel_values": self.img1})[0]
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
            "expected >= 0.999 (measured baseline ~0.999987)"
        )
