<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Whisper Variant Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the existing Whisper-large-v3 EP support to large-v3-turbo and the small openai sizes (tiny/base/small/medium) by parameterizing model sourcing, shape config, and the greedy harness — with one per-variant EP-vs-CPU smoke test each.

**Architecture:** All Whisper variants share one architecture and one decoder surgery. Per-variant differences are only: layer/head counts, `n_mels` (80 for small sizes, 128 for turbo/large-v3), `n_vocab`, and special-token IDs. `head_dim` is 64 and `n_text_ctx` is 448 for every variant. Shapes are derived from each model's `transformers.WhisperConfig`; start/eot tokens are derived from its tokenizer (the large-v3 51866 layout shifts task tokens by one vs the 51865 multilingual layout). A small registry pins only `(hf_model_id, revision)`.

**Tech Stack:** Python, pytest, onnx, onnxruntime, transformers, the OGA DirectML model builder (isolated venv, already wired), the MorphiZen/AMDGPU-umbrella EP.

## Global Constraints

- This work lives entirely in **test infra + scripts + docs** — no compiler/runtime/kernel changes. (Test files are exempt from the "no model-specific names in code comments" rule; scripts are not compiler code.)
- All new Python files start with the MIT license header (3-line `Copyright (C) 2026 Advanced Micro Devices, Inc. ... Licensed under the MIT License.`) — copy the exact block from `test/python/whisper/whisper_infer.py:1-4`.
- Format every changed file with `lintrunner -a` (ruff) before each commit.
- Model-dir naming convention is fixed: `models/whisper-{variant}-onnx` (fp32) and `models/whisper-{variant}-onnx-fp16` (fp16). This string convention is duplicated in `scripts/build_whisper_models.py` and `test/python/conftest.py` — both carry a comment that they MUST match.
- `head_dim == 64`, `n_text_ctx == 448`, `n_audio_ctx == 1500` for ALL Whisper variants (verified). Code derives them but these are the invariants.
- Variants in scope: `large-v3-turbo`, `tiny`, `base`, `small`, `medium`. `large-v3` behavior must stay byte-for-byte unchanged.
- Offline-unit-testable pure functions get real TDD here. GPU/build/network integration tasks (build, setup-download, smoke) cannot be exercised offline; their "run" steps are marked **[gfx1151 host]** and the offline deliverable is collection/`--list`/import success.

---

### Task 1: Variant registry + config/token derivation (conftest, offline-TDD)

**Files:**
- Modify: `test/python/conftest.py` (add registry + 2 pure helpers + `WhisperVariant` + `resolve_whisper_variant`, near the existing `WhisperModelConfig` at line 557)
- Test: `test/python/whisper/test_whisper_variants_unit.py` (create)

**Interfaces:**
- Consumes: existing `WhisperModelConfig` dataclass (conftest.py:557).
- Produces:
  - `WHISPER_VARIANTS: dict[str, tuple[str, str]]` — `name -> (hf_model_id, revision)`.
  - `whisper_model_config_from_hf_config(c) -> WhisperModelConfig` — `c` is any object with attrs `d_model, encoder_layers, encoder_attention_heads, max_source_positions, decoder_layers, decoder_attention_heads, max_target_positions, num_mel_bins, vocab_size`.
  - `WHISPER_SPECIAL_TOKENS: tuple[str, ...]`.
  - `whisper_start_tokens(tokenizer) -> tuple[list[int], int]` — returns `(start_tokens, eot)`; `tokenizer` has `convert_tokens_to_ids(str) -> int`.
  - `WhisperVariant` dataclass with fields `name, hf_model_id, revision, cfg, start_tokens, eot` and properties `n_layers, n_heads, head_dim, self_kv_slots`.
  - `resolve_whisper_variant(name) -> WhisperVariant` (uses transformers; network/cache).

- [ ] **Step 1: Write the failing tests**

Create `test/python/whisper/test_whisper_variants_unit.py`:

```python
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Offline unit tests for the Whisper variant config/token derivation helpers.

No network, no GPU, no built models — these exercise the pure mapping functions
that turn a transformers WhisperConfig / tokenizer into our shape config and
forced-start tokens.
"""

import pathlib
import sys
from types import SimpleNamespace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from conftest import (  # noqa: E402
    WHISPER_VARIANTS,
    WhisperModelConfig,
    whisper_model_config_from_hf_config,
    whisper_start_tokens,
)


class _FakeTok:
    def __init__(self, mapping):
        self._m = mapping

    def convert_tokens_to_ids(self, t):
        return self._m[t]


def test_registry_has_target_variants():
    for name in ("large-v3", "large-v3-turbo", "tiny", "base", "small", "medium"):
        assert name in WHISPER_VARIANTS
        hf_id, rev = WHISPER_VARIANTS[name]
        assert hf_id.startswith("openai/whisper")
        assert isinstance(rev, str) and rev


def test_model_config_from_hf_config_small():
    c = SimpleNamespace(
        d_model=384, encoder_layers=4, encoder_attention_heads=6,
        max_source_positions=1500, decoder_layers=4, decoder_attention_heads=6,
        max_target_positions=448, num_mel_bins=80, vocab_size=51865,
    )
    cfg = whisper_model_config_from_hf_config(c)
    assert isinstance(cfg, WhisperModelConfig)
    assert cfg.n_audio_state == 384 and cfg.n_text_state == 384
    assert cfg.n_text_layer == 4 and cfg.n_text_head == 6
    assert cfg.n_mels == 80 and cfg.n_vocab == 51865
    assert cfg.n_text_ctx == 448 and cfg.n_audio_ctx == 1500
    # head_dim invariant
    assert cfg.n_text_state // cfg.n_text_head == 64


def test_model_config_from_hf_config_turbo():
    c = SimpleNamespace(
        d_model=1280, encoder_layers=32, encoder_attention_heads=20,
        max_source_positions=1500, decoder_layers=4, decoder_attention_heads=20,
        max_target_positions=448, num_mel_bins=128, vocab_size=51866,
    )
    cfg = whisper_model_config_from_hf_config(c)
    assert cfg.n_audio_layer == 32 and cfg.n_text_layer == 4
    assert cfg.n_mels == 128 and cfg.n_vocab == 51866


def test_start_tokens_large_v3_layout():
    # large-v3 (51866 vocab): transcribe/notimestamps at the high IDs.
    m = {
        "<|startoftranscript|>": 50258, "<|en|>": 50259,
        "<|transcribe|>": 50360, "<|notimestamps|>": 50364,
        "<|endoftext|>": 50257,
    }
    start, eot = whisper_start_tokens(_FakeTok(m))
    assert start == [50258, 50259, 50360, 50364]
    assert eot == 50257


def test_start_tokens_multilingual_shift():
    # smaller variants (51865 vocab): one fewer language -> transcribe/
    # notimestamps shifted DOWN by one vs large-v3. The helper must read the
    # actual IDs, never hardcode the large-v3 ones.
    m = {
        "<|startoftranscript|>": 50258, "<|en|>": 50259,
        "<|transcribe|>": 50359, "<|notimestamps|>": 50363,
        "<|endoftext|>": 50257,
    }
    start, eot = whisper_start_tokens(_FakeTok(m))
    assert start == [50258, 50259, 50359, 50363]
    assert eot == 50257
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest test/python/whisper/test_whisper_variants_unit.py -v`
Expected: FAIL — `ImportError: cannot import name 'WHISPER_VARIANTS'` (and the helpers).

- [ ] **Step 3: Implement registry + helpers in conftest.py**

Insert immediately AFTER the `WhisperModelConfig` dataclass (after conftest.py:575, before the `WHISPER_HF_REPO_FP32` constants at line 584):

```python
# ── Whisper variant registry + config/token derivation ───────────────────────
#
# Per-variant shape params (state, layers, heads, n_mels, vocab) are DERIVED from
# each model's transformers WhisperConfig (whisper_model_config_from_hf_config) —
# we do NOT maintain a hardcoded dimension table. The registry pins only the HF
# model id + revision so the source weights are reproducible. head_dim is 64,
# n_text_ctx is 448, n_audio_ctx is 1500 for every Whisper variant.
WHISPER_VARIANTS = {
    # name            (hf_model_id,                    revision)
    "large-v3":       ("openai/whisper-large-v3",       "06f233fe06e710322aca913c1bc4249a0d71fce1"),
    "large-v3-turbo": ("openai/whisper-large-v3-turbo", "main"),
    "tiny":           ("openai/whisper-tiny",           "main"),
    "base":           ("openai/whisper-base",           "main"),
    "small":          ("openai/whisper-small",          "main"),
    "medium":         ("openai/whisper-medium",         "main"),
}


def whisper_model_config_from_hf_config(c) -> "WhisperModelConfig":
    """Map a transformers WhisperConfig (or any object exposing the same attrs)
    to our WhisperModelConfig. This is the single source of per-variant shape
    params — the OGA builder reads the same WhisperConfig, so the ONNX and our
    test config can never disagree."""
    return WhisperModelConfig(
        n_audio_state=c.d_model,
        n_audio_layer=c.encoder_layers,
        n_audio_head=c.encoder_attention_heads,
        n_audio_ctx=c.max_source_positions,
        n_text_state=c.d_model,
        n_text_layer=c.decoder_layers,
        n_text_head=c.decoder_attention_heads,
        n_text_ctx=c.max_target_positions,
        n_mels=c.num_mel_bins,
        n_vocab=c.vocab_size,
    )


# Forced-start specials for transcription (English, no timestamps). The IDs are
# READ from the tokenizer per variant — large-v3's 51866 layout added a language
# vs the 51865 multilingual layout, which shifts <|transcribe|>/<|notimestamps|>
# by one, so these MUST NOT be hardcoded.
WHISPER_SPECIAL_TOKENS = (
    "<|startoftranscript|>",
    "<|en|>",
    "<|transcribe|>",
    "<|notimestamps|>",
)


def whisper_start_tokens(tokenizer):
    """Derive (start_tokens, eot) from a tokenizer exposing
    convert_tokens_to_ids(str) -> int. Returns (list[int], int)."""
    start = [tokenizer.convert_tokens_to_ids(t) for t in WHISPER_SPECIAL_TOKENS]
    eot = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    return start, eot


@dataclass
class WhisperVariant:
    """A fully-resolved Whisper variant: pinned source + derived shape config +
    derived forced-start tokens. head_dim / n_layers / n_heads / self_kv_slots
    are read off cfg so callers never recompute them."""

    name: str
    hf_model_id: str
    revision: str
    cfg: "WhisperModelConfig"
    start_tokens: list
    eot: int

    @property
    def n_layers(self) -> int:
        return self.cfg.n_text_layer

    @property
    def n_heads(self) -> int:
        return self.cfg.n_text_head

    @property
    def head_dim(self) -> int:
        return self.cfg.n_text_state // self.cfg.n_text_head

    @property
    def self_kv_slots(self) -> int:
        return self.cfg.n_text_ctx


def resolve_whisper_variant(name) -> "WhisperVariant":
    """Resolve a variant name to a WhisperVariant. Reads the HF WhisperConfig +
    tokenizer (network/cache) for shapes and forced-start tokens. Raises KeyError
    for an unknown name."""
    if name not in WHISPER_VARIANTS:
        raise KeyError(
            f"unknown Whisper variant {name!r}; known: {sorted(WHISPER_VARIANTS)}"
        )
    hf_id, rev = WHISPER_VARIANTS[name]
    from transformers import WhisperConfig, WhisperTokenizer

    cfg = whisper_model_config_from_hf_config(
        WhisperConfig.from_pretrained(hf_id, revision=rev)
    )
    start, eot = whisper_start_tokens(
        WhisperTokenizer.from_pretrained(hf_id, revision=rev)
    )
    return WhisperVariant(name, hf_id, rev, cfg, start, eot)
```

Note: `dataclass` is already imported in conftest (used by `WhisperModelConfig`). If a `flake`/ruff run complains about the forward-ref string types, they are fine (`WhisperModelConfig` is defined above this block).

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest test/python/whisper/test_whisper_variants_unit.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Format + commit**

```bash
lintrunner -a
git add test/python/conftest.py test/python/whisper/test_whisper_variants_unit.py
git commit -m "test(whisper): variant registry + config/token derivation helpers"
```

---

### Task 2: Output-dir mapping + `build_whisper_models.py` `--variant`/`--list`

**Files:**
- Modify: `scripts/build_whisper_models.py`
- Test: `test/python/whisper/test_whisper_build_paths.py` (create)

**Interfaces:**
- Consumes: `WHISPER_VARIANTS` source-of-truth is in conftest, but the builder stays standalone (no conftest import). It carries its OWN per-variant `(hf_model_id, revision)` table — a small intentional duplication, gated by a comment that it MUST match `conftest.WHISPER_VARIANTS`.
- Produces:
  - `whisper_output_dirs(variant) -> tuple[Path, Path]` — `(fp32_dir, fp16_dir)`.
  - `VARIANT_SOURCES: dict[str, tuple[str, str]]` — `name -> (hf_model_id, revision)`.
  - `DEFAULT_VARIANTS: list[str]` — `["large-v3-turbo", "tiny", "base", "small", "medium"]`.
  - CLI: `--variant` (repeatable / comma-list), `--list`.

- [ ] **Step 1: Write the failing test**

Create `test/python/whisper/test_whisper_build_paths.py`:

```python
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Offline tests for build_whisper_models path/variant resolution (no OGA run)."""

import importlib.util
import pathlib

_SCRIPTS = pathlib.Path(__file__).resolve().parents[3] / "scripts"
_spec = importlib.util.spec_from_file_location(
    "build_whisper_models", _SCRIPTS / "build_whisper_models.py"
)
bwm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bwm)


def test_output_dirs_convention():
    fp32, fp16 = bwm.whisper_output_dirs("tiny")
    assert fp32.name == "whisper-tiny-onnx"
    assert fp16.name == "whisper-tiny-onnx-fp16"
    assert fp32.parent.name == "models"


def test_default_variants_cover_scope():
    for v in ("large-v3-turbo", "tiny", "base", "small", "medium"):
        assert v in bwm.DEFAULT_VARIANTS
        assert v in bwm.VARIANT_SOURCES


def test_variant_sources_match_conftest():
    # The builder duplicates the (hf_id, revision) table for standalone-ness; it
    # MUST stay in sync with conftest.WHISPER_VARIANTS.
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    from conftest import WHISPER_VARIANTS

    for name, src in bwm.VARIANT_SOURCES.items():
        assert WHISPER_VARIANTS[name] == src
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest test/python/whisper/test_whisper_build_paths.py -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'whisper_output_dirs'`.

- [ ] **Step 3: Refactor build_whisper_models.py to a variant table**

Replace the pinned-input + PRECISIONS block (current lines ~38-60) with:

```python
# ---------------------------------------------------------------------------
# Pinned per-variant sources (same idiom as build.py's version pins).
# DUPLICATED from conftest.WHISPER_VARIANTS so this builder stays standalone
# (no test-infra import). test_whisper_build_paths.py asserts the two match.
# ---------------------------------------------------------------------------
VARIANT_SOURCES = {
    # name            (hf_model_id,                    revision)
    "large-v3":       ("openai/whisper-large-v3",       "06f233fe06e710322aca913c1bc4249a0d71fce1"),
    "large-v3-turbo": ("openai/whisper-large-v3-turbo", "main"),
    "tiny":           ("openai/whisper-tiny",           "main"),
    "base":           ("openai/whisper-base",           "main"),
    "small":          ("openai/whisper-small",          "main"),
    "medium":         ("openai/whisper-medium",         "main"),
}

# Default build set when --variant is omitted (turbo + the small sizes; large-v3
# is built/downloaded via its own AMD-HF path in conftest, so it is not default).
DEFAULT_VARIANTS = ["large-v3-turbo", "tiny", "base", "small", "medium"]

# Exact builder deps. onnxruntime-genai-directml is the STOCK builder package;
# torch CPU is fine (the builder only reads HF weights). These live ONLY in the
# isolated venv, never the dev env.
BUILDER_REQUIREMENTS = [
    "onnxruntime-genai-directml==0.13.1",
    "torch==2.12.0",
    "transformers==4.49.0",
    "onnx",
    "onnx_ir==0.2.1",
]


def whisper_output_dirs(variant):
    """(fp32_dir, fp16_dir) under models/ for a variant. MUST match the naming
    convention in conftest (models/whisper-{variant}-onnx[-fp16])."""
    return (
        MODELS / f"whisper-{variant}-onnx",
        MODELS / f"whisper-{variant}-onnx-fp16",
    )
```

Delete the old module-level `WHISPER_HF_MODEL` / `WHISPER_HF_REVISION` / `PRECISIONS` constants. Update `build_one` to take `(py, model_id, revision, precision, out_dir)` and pass `model_id`/`revision` into the subprocess argv (the `BUILDER_SNIPPET` already reads `model_id, revision, precision, out_dir, cache_dir = sys.argv[1:6]` — keep that). Update `main()`:

```python
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build reproducible Whisper fp32 + fp16 ONNX via the pinned "
        "OGA DirectML model builder (isolated venv)."
    )
    ap.add_argument(
        "--variant",
        action="append",
        default=None,
        help="variant to build (repeatable, or comma-separated). "
        f"Known: {sorted(VARIANT_SOURCES)}. Default: {DEFAULT_VARIANTS}.",
    )
    ap.add_argument(
        "--precision", choices=["fp32", "fp16", "both"], default="both",
        help="which precision(s) to build (default: both)",
    )
    ap.add_argument(
        "--list", action="store_true",
        help="print the resolved variant -> output-dir mapping and exit",
    )
    args = ap.parse_args()

    # Resolve --variant (repeatable AND comma-list) -> flat list; default set.
    if args.variant:
        variants = []
        for v in args.variant:
            variants.extend(p.strip() for p in v.split(",") if p.strip())
    else:
        variants = list(DEFAULT_VARIANTS)
    for v in variants:
        if v not in VARIANT_SOURCES:
            ap.error(f"unknown variant {v!r}; known: {sorted(VARIANT_SOURCES)}")

    if args.list:
        for v in variants:
            fp32, fp16 = whisper_output_dirs(v)
            print(f"{v}: fp32={fp32}  fp16={fp16}  src={VARIANT_SOURCES[v]}")
        return 0

    py = ensure_builder_venv()
    for v in variants:
        model_id, revision = VARIANT_SOURCES[v]
        fp32_dir, fp16_dir = whisper_output_dirs(v)
        for prec, out_dir in (("fp32", fp32_dir), ("fp16", fp16_dir)):
            if args.precision in (prec, "both"):
                build_one(py, model_id, revision, prec, out_dir)

    print("[whisper-build] DONE.")
    return 0
```

And change `build_one`'s signature/body:

```python
def build_one(py: Path, model_id: str, revision: str, precision: str, out_dir: Path) -> None:
    """Build one (variant, precision) via the OGA builder in the isolated venv.
    Idempotent (skips if the raw encoder+decoder already exist)."""
    if (out_dir / "encoder.onnx").exists() and (out_dir / "decoder.onnx").exists():
        print(f"[whisper-build] {out_dir.name}: already built")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = BUILDER_VENV / "_hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"[whisper-build] building {precision} {model_id} -> {out_dir}")
    subprocess.run(
        [str(py), "-c", BUILDER_SNIPPET, model_id, revision, precision,
         str(out_dir), str(cache_dir)],
        check=True,
    )
```

Update the module docstring to say "Whisper variants" instead of "Whisper-large-v3".

- [ ] **Step 4: Run test + smoke the CLI**

Run: `pytest test/python/whisper/test_whisper_build_paths.py -v`
Expected: PASS (3 tests).

Run: `python scripts/build_whisper_models.py --list`
Expected: prints 5 default variants with their fp32/fp16 dirs; exit 0.

Run: `python scripts/build_whisper_models.py --variant tiny --list`
Expected: prints only the tiny row.

- [ ] **Step 5: Format + commit**

```bash
lintrunner -a
git add scripts/build_whisper_models.py test/python/whisper/test_whisper_build_paths.py
git commit -m "feat(whisper): parameterize model builder over variants (--variant/--list)"
```

---

### Task 3: Generalize setup to `setup_whisper_variant` (conftest)

**Files:**
- Modify: `test/python/conftest.py` (lines ~588-714: `_ensure_whisper_raw_downloaded`, `setup_whisper_model_dir`, `_apply_whisper_surgery_and_fix_shapes`, `setup_whisper_fp16_model_dir`)
- Test: `test/python/whisper/test_whisper_setup_dispatch.py` (create)

**Interfaces:**
- Consumes: `resolve_whisper_variant` (Task 1), `inject_seqlens_k`, `fix_shapes` (existing conftest).
- Produces:
  - `whisper_model_dir(name, precision) -> Path` — model dir for `(variant, precision)`; same convention as `whisper_output_dirs`.
  - `setup_whisper_variant(name, precision="fp16") -> tuple[Path, WhisperVariant]` — ensures raw bundle, runs surgery + fix_shapes with the variant's `n_text_ctx`, returns `(model_dir, variant)`.
  - `_apply_whisper_surgery_and_fix_shapes(model_dir, n_text_ctx=448)` — `n_text_ctx` now a parameter.
  - Unchanged public behavior: `setup_whisper_model_dir(model_dir)`, `setup_whisper_fp16_model_dir(model_dir)` (large-v3 wrappers).

- [ ] **Step 1: Write the failing test**

Create `test/python/whisper/test_whisper_setup_dispatch.py`:

```python
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Offline tests for setup_whisper_variant dispatch (no download, no surgery).

We monkeypatch the raw-ensure + surgery steps so the test verifies ONLY the
routing: correct model_dir per (variant, precision), n_text_ctx threaded from the
resolved variant, and large-v3 wrappers delegating unchanged.
"""

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import conftest  # noqa: E402


def test_whisper_model_dir_convention():
    p = conftest.whisper_model_dir("tiny", "fp16")
    assert p.name == "whisper-tiny-onnx-fp16"
    p32 = conftest.whisper_model_dir("tiny", "fp32")
    assert p32.name == "whisper-tiny-onnx"


def test_setup_variant_threads_n_text_ctx(monkeypatch):
    calls = {}

    # Fake a resolved variant with a sentinel n_text_ctx so we can assert it is
    # the value threaded into surgery+fix_shapes.
    fake_cfg = conftest.WhisperModelConfig(n_text_ctx=448, n_vocab=51865)
    fake_var = conftest.WhisperVariant(
        name="tiny", hf_model_id="openai/whisper-tiny", revision="main",
        cfg=fake_cfg, start_tokens=[1, 2, 3, 4], eot=0,
    )
    monkeypatch.setattr(conftest, "resolve_whisper_variant", lambda n: fake_var)
    monkeypatch.setattr(
        conftest, "_ensure_whisper_raw", lambda name, model_dir, precision: None
    )
    # Pretend the raw files exist so the guard passes.
    monkeypatch.setattr(conftest.pathlib.Path, "exists", lambda self: True)

    def _fake_surgery(model_dir, n_text_ctx=448):
        calls["model_dir"] = model_dir
        calls["n_text_ctx"] = n_text_ctx

    monkeypatch.setattr(
        conftest, "_apply_whisper_surgery_and_fix_shapes", _fake_surgery
    )

    model_dir, var = conftest.setup_whisper_variant("tiny", "fp16")
    assert model_dir.name == "whisper-tiny-onnx-fp16"
    assert var is fake_var
    assert calls["n_text_ctx"] == 448
    assert calls["model_dir"].name == "whisper-tiny-onnx-fp16"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest test/python/whisper/test_whisper_setup_dispatch.py -v`
Expected: FAIL — `AttributeError: module 'conftest' has no attribute 'whisper_model_dir'`.

- [ ] **Step 3: Implement the generalized setup in conftest.py**

Add near the `WHISPER_HF_REPO_FP32/FP16` constants:

```python
def whisper_model_dir(name, precision):
    """models/whisper-{name}-onnx[-fp16] for a (variant, precision). MUST match
    build_whisper_models.whisper_output_dirs."""
    suffix = "-fp16" if precision == "fp16" else ""
    return REPO_ROOT / "models" / f"whisper-{name}-onnx{suffix}"


def _ensure_whisper_raw(name, model_dir, precision):
    """Ensure model_dir/{encoder,decoder}.onnx exist. large-v3 downloads from its
    AMD HF repo (existing behavior); all other variants are LOCAL-BUILD primary
    (no AMD repo) — raise a build hint if absent."""
    if (model_dir / "encoder.onnx").exists() and (model_dir / "decoder.onnx").exists():
        return
    if name == "large-v3":
        repo = WHISPER_HF_REPO_FP16 if precision == "fp16" else WHISPER_HF_REPO_FP32
        _ensure_whisper_raw_downloaded(model_dir, repo)
        return
    raise FileNotFoundError(
        f"Whisper {name} {precision} raw model not found at {model_dir}.\n"
        f"  Build it locally: python scripts/build_whisper_models.py "
        f"--variant {name} --precision {precision}\n"
        f"  (or: python build.py --build-whisper-models)"
    )


def setup_whisper_variant(name, precision="fp16"):
    """Idempotent prep for ANY Whisper variant. Ensures the raw OGA bundle (local
    build primary; large-v3 may download from AMD HF), then surgery + fix_shapes
    using the variant's n_text_ctx. Returns (model_dir, WhisperVariant)."""
    var = resolve_whisper_variant(name)
    model_dir = whisper_model_dir(name, precision)
    _ensure_whisper_raw(name, model_dir, precision)
    for fname in ("encoder.onnx", "decoder.onnx"):
        if not (model_dir / fname).exists():
            raise FileNotFoundError(
                f"Whisper {name} {precision} incomplete at {model_dir} "
                f"(missing {fname}); build with "
                f"python scripts/build_whisper_models.py --variant {name}"
            )
    _apply_whisper_surgery_and_fix_shapes(model_dir, var.cfg.n_text_ctx)
    return model_dir, var
```

Change `_apply_whisper_surgery_and_fix_shapes` signature + the two `fix_shapes` calls to use `n_text_ctx`:

```python
def _apply_whisper_surgery_and_fix_shapes(model_dir, n_text_ctx=448):
    ...
    decoder_surgery = model_dir / "decoder_surgery.onnx"
    inject_seqlens_k(model_dir / "decoder.onnx", decoder_surgery)

    encoder_fixed = model_dir / "encoder_fixed.onnx"
    if not encoder_fixed.exists():
        fix_shapes(model_dir / "encoder.onnx", encoder_fixed, {"batch_size": 1})

    prefill_fixed = model_dir / "decoder_fixed_prefill.onnx"
    if not prefill_fixed.exists():
        fix_shapes(
            decoder_surgery, prefill_fixed,
            {"batch_size": 1, "sequence_length": 4,
             "past_sequence_length": n_text_ctx, "total_sequence_length": n_text_ctx},
        )

    decode_fixed = model_dir / "decoder_fixed_decode.onnx"
    if not decode_fixed.exists():
        fix_shapes(
            decoder_surgery, decode_fixed,
            {"batch_size": 1, "sequence_length": 1,
             "past_sequence_length": n_text_ctx, "total_sequence_length": n_text_ctx},
        )
```

Keep `setup_whisper_model_dir` / `setup_whisper_fp16_model_dir` working: their bodies already call `_apply_whisper_surgery_and_fix_shapes(model_dir)` (now defaulting `n_text_ctx=448`), so they are unchanged and large-v3 stays byte-identical. (Do NOT route them through `setup_whisper_variant` — large-v3 keeps its existing AMD-repo path via `_ensure_whisper_raw_downloaded`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest test/python/whisper/test_whisper_setup_dispatch.py -v`
Expected: PASS.

Also re-run Task 1 tests to confirm no regression:
Run: `pytest test/python/whisper/test_whisper_variants_unit.py -v`
Expected: PASS.

- [ ] **Step 5: Format + commit**

```bash
lintrunner -a
git add test/python/conftest.py test/python/whisper/test_whisper_setup_dispatch.py
git commit -m "feat(whisper): setup_whisper_variant — variant-generic surgery+fix_shapes"
```

---

### Task 4: Thread variant through the greedy harness (whisper_infer.py)

**Files:**
- Modify: `test/python/whisper/whisper_infer.py`
- Modify: `test/python/whisper/test_whisper.py` (update imports if shared helpers move)
- Test: `test/python/whisper/test_whisper_runtime_params.py` (create)

**Interfaces:**
- Consumes: `WhisperVariant` (Task 1).
- Produces:
  - `_runtime_params(variant) -> tuple` — `(n_layers, n_heads, head_dim, self_kv_slots, start_tokens, eot, n_vocab)`; `variant=None` returns the large-v3 module constants.
  - `encoder_cross_kv(enc, audio, dtype=np.float32, variant=None)` — variant-aware layer count.
  - `zeroed_self_past(dtype=np.float32, variant=None)`.
  - `greedy_decode_cpu(session_factory, audio, max_length=200, timings=None, dtype=np.float32, variant=None)`.
  - `greedy_decode_morphizen(..., variant=None)` (+ `_greedy_decode_iobinding` / `_greedy_decode_numpy` gain `variant=None`).
  - `decode_text(tokens, tokenizer_id="openai/whisper-large-v3", eot=EOT)`.

- [ ] **Step 1: Write the failing test**

Create `test/python/whisper/test_whisper_runtime_params.py`:

```python
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Offline test: _runtime_params resolves large-v3 defaults and variant overrides."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import whisper_infer  # noqa: E402
from conftest import WhisperModelConfig, WhisperVariant  # noqa: E402


def test_runtime_params_default_is_large_v3():
    nl, nh, hd, slots, start, eot, vocab = whisper_infer._runtime_params(None)
    assert nl == 32 and nh == 20 and hd == 64 and slots == 448
    assert start == whisper_infer.START_TOKENS and eot == whisper_infer.EOT
    assert vocab == 51866


def test_runtime_params_variant_override():
    cfg = WhisperModelConfig(
        n_audio_state=384, n_audio_layer=4, n_audio_head=6,
        n_text_state=384, n_text_layer=4, n_text_head=6,
        n_text_ctx=448, n_mels=80, n_vocab=51865,
    )
    var = WhisperVariant(
        name="tiny", hf_model_id="openai/whisper-tiny", revision="main",
        cfg=cfg, start_tokens=[50258, 50259, 50359, 50363], eot=50257,
    )
    nl, nh, hd, slots, start, eot, vocab = whisper_infer._runtime_params(var)
    assert nl == 4 and nh == 6 and hd == 64 and slots == 448
    assert start == [50258, 50259, 50359, 50363] and eot == 50257
    assert vocab == 51865
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest test/python/whisper/test_whisper_runtime_params.py -v`
Expected: FAIL — `AttributeError: module 'whisper_infer' has no attribute '_runtime_params'`.

- [ ] **Step 3: Add `_runtime_params` and thread it through whisper_infer**

Add after the module constants (after whisper_infer.py:63):

```python
def _runtime_params(variant):
    """Resolve per-variant runtime params. variant=None -> the large-v3 module
    constants (so existing callers are byte-for-byte unchanged)."""
    if variant is None:
        return (
            N_LAYERS, N_HEADS, HEAD_DIM, SELF_KV_SLOTS,
            list(START_TOKENS), EOT, CFG.n_vocab,
        )
    return (
        variant.n_layers, variant.n_heads, variant.head_dim, variant.self_kv_slots,
        list(variant.start_tokens), variant.eot, variant.cfg.n_vocab,
    )
```

Then, in EACH of `encoder_cross_kv`, `zeroed_self_past`, `greedy_decode_cpu`, `greedy_decode_morphizen`, `_greedy_decode_iobinding`, `_greedy_decode_numpy`:
1. Add `variant=None` to the signature.
2. At the top of the body, unpack:
   `n_layers, n_heads, head_dim, slots, start_tokens, eot, n_vocab = _runtime_params(variant)`
3. Replace every use of the module globals `N_LAYERS`/`N_HEADS`/`HEAD_DIM`/`SELF_KV_SLOTS`/`START_TOKENS`/`EOT`/`CFG.n_vocab` in that function body with the local `n_layers`/`n_heads`/`head_dim`/`slots`/`start_tokens`/`eot`/`n_vocab`.
4. In `greedy_decode_morphizen`, forward `variant=variant` into the `impl(...)` call.

Example — `encoder_cross_kv` becomes:

```python
def encoder_cross_kv(enc_session, audio_fp, dtype=np.float32, variant=None):
    n_layers, *_ = _runtime_params(variant)
    feats = audio_fp.astype(dtype)
    names = [o.name for o in enc_session.get_outputs()]
    out = enc_session.run(None, {"audio_features": feats})
    omap = dict(zip(names, out))
    cross = {}
    for i in range(n_layers):
        cross[f"past_key_cross_{i}"] = omap[f"present_key_cross_{i}"].astype(dtype)
        cross[f"past_value_cross_{i}"] = omap[f"present_value_cross_{i}"].astype(dtype)
    return omap["hidden_states"], cross
```

`zeroed_self_past`:

```python
def zeroed_self_past(dtype=np.float32, variant=None):
    n_layers, n_heads, head_dim, slots, *_ = _runtime_params(variant)
    kv = {}
    for i in range(n_layers):
        kv[f"past_key_self_{i}"] = np.zeros((1, n_heads, slots, head_dim), dtype=dtype)
        kv[f"past_value_self_{i}"] = np.zeros((1, n_heads, slots, head_dim), dtype=dtype)
    return kv
```

In `greedy_decode_cpu` / `_greedy_decode_numpy` / `_greedy_decode_iobinding`, the per-step loops reference `START_TOKENS`, `EOT`, `SELF_KV_SLOTS`, `N_LAYERS`, `N_HEADS`, `HEAD_DIM`, `CFG.n_vocab`, and call `encoder_cross_kv(...)` / `zeroed_self_past(...)` — pass `variant=variant` into those two calls and use the unpacked locals everywhere else. `s0 = len(start_tokens)`.

`decode_text` becomes variant-aware (default large-v3 so existing callers unchanged):

```python
def decode_text(tokens, tokenizer_id="openai/whisper-large-v3", eot=EOT):
    from transformers import WhisperTokenizer

    tok = WhisperTokenizer.from_pretrained(tokenizer_id)
    body = [t for t in tokens if t < eot]
    return tok.decode(body, skip_special_tokens=True).strip()
```

- [ ] **Step 4: Run the test + the existing offline whisper unit tests**

Run: `pytest test/python/whisper/test_whisper_runtime_params.py -v`
Expected: PASS (2 tests).

Run: `pytest test/python/whisper/test_inject_seqlens_k.py -v`
Expected: PASS (no regression — surgery untouched).

- [ ] **Step 5: [gfx1151 host] Regression-check large-v3 greedy is unchanged**

Run on the GPU host: `pytest test/python/whisper/test_whisper.py::test_e2e_transcription_greedy -v -s`
Expected: PASS for both fp16 and fp32 (verbatim JFK quote) — proves the `variant=None` default path is byte-for-byte the old behavior. (Skips cleanly if the EP/model is absent on the implementer's box; the reviewer runs it on the gfx1151 host.)

- [ ] **Step 6: Format + commit**

```bash
lintrunner -a
git add test/python/whisper/whisper_infer.py test/python/whisper/test_whisper_runtime_params.py
git commit -m "refactor(whisper): thread WhisperVariant through the greedy harness"
```

---

### Task 5: Per-variant EP-vs-CPU smoke test

**Files:**
- Modify: `test/python/whisper/whisper_infer.py` (move `_CaptureFD` + `_assert_compiled_on_gpu` here so the smoke test and `test_whisper.py` share one copy)
- Modify: `test/python/whisper/test_whisper.py` (import the two moved helpers from `whisper_infer` instead of defining them)
- Create: `test/python/whisper/test_whisper_variant_smoke.py`

**Interfaces:**
- Consumes: `setup_whisper_variant` (Task 3), `greedy_decode_cpu` / `greedy_decode_morphizen` / `make_cpu_session_factory` / `make_morphizen_session_factory` (Task 4), `make_whisper_inputs` (conftest).
- Produces:
  - In `whisper_infer.py`: `class CaptureFD` (renamed from `_CaptureFD`, now public-ish/shared) and `assert_compiled_on_gpu(stderr_text)`.
  - `test/python/whisper/test_whisper_variant_smoke.py` with `test_variant_smoke[<variant>]` parametrized over turbo + small sizes.

- [ ] **Step 1: Move the shared FD-capture + compile tripwire into whisper_infer.py**

Cut `_CaptureFD` (test_whisper.py:1084-1130) and `_assert_compiled_on_gpu` (test_whisper.py:296-314) out of `test_whisper.py`. Paste into `whisper_infer.py` near the end (before `decode_text`), renamed to `CaptureFD` and `assert_compiled_on_gpu` (drop the leading underscore — they are now a shared API). They have NO pytest dependency. Add `import os`, `import sys`, `import tempfile` if not already imported in whisper_infer (os/sys are; add `tempfile` lazily inside `CaptureFD.__enter__` as the original does).

In `test_whisper.py`, replace both definitions with an import and local aliases (keep the underscore names the rest of the file uses):

```python
from whisper_infer import (  # noqa: E402
    CaptureFD as _CaptureFD,
    assert_compiled_on_gpu as _assert_compiled_on_gpu,
    ...  # existing imports
)
```

- [ ] **Step 2: Verify the move did not break test_whisper collection**

Run: `pytest test/python/whisper/test_whisper.py --collect-only -q`
Expected: collection succeeds (all the existing test ids listed, no ImportError).

- [ ] **Step 3: Write the smoke test**

Create `test/python/whisper/test_whisper_variant_smoke.py`:

```python
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Per-variant EP correctness smoke test for Whisper turbo + small sizes.

ORACLE: same-variant MorphiZen-GPU greedy tokens == ORT-CPU greedy tokens. NOT a
verbatim-text bar — small models (tiny/base) have low transcription accuracy, so
an absolute-text assertion would test model quality, not EP correctness. Comparing
EP vs CPU on the SAME variant isolates the EP. large-v3's full correctness +
verbatim-JFK matrix stays in test_whisper.py.

Each variant is built/setup on demand (fp16); a missing build SKIPS cleanly, so a
box without the OGA builder / GPU degrades gracefully. The GPU run is wrapped in a
bounded FD capture to assert the graph compiled on GPU (no silent CPU fallback —
otherwise EP==CPU would be trivially true).
"""

import gc
import os
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from conftest import (  # noqa: E402
    REPO_ROOT,
    make_whisper_inputs,
    setup_jfk_sample,
    setup_whisper_variant,
)
import whisper_infer  # noqa: E402

# turbo (128-mel, 4 decoder layers) + the 80-mel small sizes. large-v3 is covered
# by the full matrix in test_whisper.py, so it is intentionally absent here.
_SMOKE_VARIANTS = ["large-v3-turbo", "tiny", "base", "small", "medium"]

_WHISPER_DATA = REPO_ROOT / "test" / "python" / "data" / "whisper"
_AUDIO = _WHISPER_DATA / "jfk.wav"


@pytest.fixture(scope="module", autouse=True)
def _audio():
    therock = REPO_ROOT / "install" / "therock"
    if therock.exists():
        os.environ["THEROCK_DIST"] = str(therock)
    if not setup_jfk_sample(_WHISPER_DATA):
        pytest.skip("jfk.wav unavailable (network fetch failed, no local cache)")


@pytest.fixture(autouse=True)
def _gc():
    yield
    gc.collect()


@pytest.mark.parametrize("variant_name", _SMOKE_VARIANTS)
def test_variant_smoke(variant_name):
    # Build + surgery + fix_shapes on demand (fp16). Missing build -> skip.
    try:
        model_dir, var = setup_whisper_variant(variant_name, precision="fp16")
    except FileNotFoundError as e:
        pytest.skip(f"{variant_name} fp16 not built: {e}")

    dtype = np.float16
    audio = make_whisper_inputs(_AUDIO, var.cfg)["audio_features"].astype(dtype)

    # CPU reference (dynamic graph, same variant).
    cpu_factory = whisper_infer.make_cpu_session_factory(model_dir)
    cpu_tokens = whisper_infer.greedy_decode_cpu(
        cpu_factory, audio, variant=var, dtype=dtype
    )
    gc.collect()

    # GPU under test. make_morphizen_session_factory raises if the EP is absent.
    try:
        mz_factory = whisper_infer.make_morphizen_session_factory(REPO_ROOT, model_dir)
    except RuntimeError as e:
        pytest.skip(f"MorphiZen EP unavailable: {e}")

    with whisper_infer.CaptureFD() as cap:
        mz_tokens = whisper_infer.greedy_decode_morphizen(
            mz_factory, audio, variant=var, dtype=dtype
        )
    # No silent CPU fallback — else EP==CPU is meaningless (both CPU).
    whisper_infer.assert_compiled_on_gpu(cap.text)

    txt = whisper_infer.decode_text(mz_tokens, tokenizer_id=var.hf_model_id, eot=var.eot)
    print(f"\n[{variant_name}] GPU greedy: {txt!r}")
    assert mz_tokens == cpu_tokens, (
        f"{variant_name}: EP greedy tokens != CPU greedy tokens\n"
        f"  cpu_len={len(cpu_tokens)} gpu_len={len(mz_tokens)}"
    )
```

- [ ] **Step 4: Verify the smoke test collects (offline)**

Run: `pytest test/python/whisper/test_whisper_variant_smoke.py --collect-only -q`
Expected: 5 parametrized ids collected (`test_variant_smoke[large-v3-turbo]` … `[medium]`), no ImportError.

- [ ] **Step 5: [gfx1151 host] Build + run the smoke suite for real**

Run on the GPU host (builds models on first call — slow):
```bash
python build.py --build-whisper-models --variant large-v3-turbo,tiny
pytest test/python/whisper/test_whisper_variant_smoke.py -v -s -k "turbo or tiny"
```
Expected: `test_variant_smoke[large-v3-turbo]` and `[tiny]` PASS (EP tokens == CPU tokens), with the compile tripwire confirming GPU dispatch. This validates BOTH the 128-mel (turbo) and 80-mel (tiny) families and the surgery node-name match on a 4-decoder-layer model. (Risk items 1 + 2 from the spec are discharged here.)

- [ ] **Step 6: Format + commit**

```bash
lintrunner -a
git add test/python/whisper/whisper_infer.py test/python/whisper/test_whisper.py \
        test/python/whisper/test_whisper_variant_smoke.py
git commit -m "test(whisper): per-variant EP-vs-CPU greedy smoke test (turbo + small)"
```

---

### Task 6: `scripts/setup_whisper_model.py --variant`

**Files:**
- Modify: `scripts/setup_whisper_model.py`
- Test: `test/python/whisper/test_setup_script_cli.py` (create)

**Interfaces:**
- Consumes: `setup_whisper_variant` (Task 3).
- Produces: CLI `--variant <name>` (default `large-v3`) + `--fp32` precision flag; a `--list` that prints the resolved model dir without running setup.

- [ ] **Step 1: Write the failing test**

Create `test/python/whisper/test_setup_script_cli.py`:

```python
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Offline CLI-parse test for scripts/setup_whisper_model.py (no setup run)."""

import importlib.util
import pathlib

_SCRIPTS = pathlib.Path(__file__).resolve().parents[3] / "scripts"
_spec = importlib.util.spec_from_file_location(
    "setup_whisper_model", _SCRIPTS / "setup_whisper_model.py"
)
swm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(swm)


def test_resolve_target_default_large_v3():
    name, precision = swm._resolve_args([])
    assert name == "large-v3" and precision == "fp16"


def test_resolve_target_variant_fp32():
    name, precision = swm._resolve_args(["--variant", "tiny", "--fp32"])
    assert name == "tiny" and precision == "fp32"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest test/python/whisper/test_setup_script_cli.py -v`
Expected: FAIL — `AttributeError: module ... has no attribute '_resolve_args'`.

- [ ] **Step 3: Rewrite setup_whisper_model.py around setup_whisper_variant**

Replace the body with:

```python
#!/usr/bin/env python
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Prepare an already-built Whisper variant ONNX for the MorphiZen EP.

Ensures the raw OGA bundle is present (large-v3 auto-downloads from AMD HF; other
variants are local-build primary — `python scripts/build_whisper_models.py
--variant <name>`), then applies the decoder surgery + fix_shapes to emit the
static-shape variants the EP compiles. Default variant is large-v3, default
precision fp16.

    python scripts/setup_whisper_model.py                       # large-v3 fp16
    python scripts/setup_whisper_model.py --variant tiny        # tiny fp16
    python scripts/setup_whisper_model.py --variant small --fp32
"""

import argparse
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "test" / "python"))

from conftest import setup_whisper_variant  # noqa: E402


def _resolve_args(argv):
    ap = argparse.ArgumentParser(
        description="Prepare a Whisper variant ONNX for the EP."
    )
    ap.add_argument("--variant", default="large-v3", help="variant name")
    ap.add_argument(
        "--fp32", action="store_true", help="select fp32 (default: fp16)"
    )
    ns = ap.parse_args(argv)
    return ns.variant, ("fp32" if ns.fp32 else "fp16")


def main() -> int:
    name, precision = _resolve_args(sys.argv[1:])
    print(f"[whisper-setup] target: {name} ({precision})")
    try:
        model_dir, _var = setup_whisper_variant(name, precision)
    except (FileNotFoundError, KeyError) as e:
        print(f"[whisper-setup] ERROR: {e}")
        return 1
    expected = [
        "encoder_fixed.onnx", "decoder_surgery.onnx",
        "decoder_fixed_prefill.onnx", "decoder_fixed_decode.onnx",
    ]
    missing = [f for f in expected if not (model_dir / f).exists()]
    if missing:
        print(f"[whisper-setup] ERROR: missing expected files: {missing}")
        return 1
    print(f"[whisper-setup] DONE — {name} {precision} ready at:\n    {model_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test + smoke the help**

Run: `pytest test/python/whisper/test_setup_script_cli.py -v`
Expected: PASS (2 tests).

Run: `python scripts/setup_whisper_model.py --help`
Expected: usage text showing `--variant` and `--fp32`; exit 0.

- [ ] **Step 5: Format + commit**

```bash
lintrunner -a
git add scripts/setup_whisper_model.py test/python/whisper/test_setup_script_cli.py
git commit -m "feat(whisper): setup_whisper_model.py --variant (any Whisper size)"
```

---

### Task 7: Docs — quick-start + CLAUDE.md

**Files:**
- Modify: `docs/whisper_quick_start.md`
- Modify: `CLAUDE.md`

**Interfaces:** none (documentation).

- [ ] **Step 1: Update `docs/whisper_quick_start.md`**

Add a "Variants" section near the top documenting the supported set and the build/setup recipe:

```markdown
## Supported variants

The EP supports these Whisper variants (all share one architecture + decoder
surgery; only shape params, n_mels, vocab, and special-token IDs differ):

| Variant         | n_mels | enc / dec layers | heads | vocab  | source                         |
|-----------------|--------|------------------|-------|--------|--------------------------------|
| large-v3        | 128    | 32 / 32          | 20    | 51866  | AMD HF (download) / local OGA  |
| large-v3-turbo  | 128    | 32 / 4           | 20    | 51866  | local OGA build                |
| medium          | 80     | 24 / 24          | 16    | 51865  | local OGA build                |
| small           | 80     | 12 / 12          | 12    | 51865  | local OGA build                |
| base            | 80     | 6 / 6            | 8     | 51865  | local OGA build                |
| tiny            | 80     | 4 / 4            | 6     | 51865  | local OGA build                |

head_dim is 64 and the decoder context is 448 for every variant.

Build + prepare a non-large-v3 variant:

    python build.py --build-whisper-models --variant large-v3-turbo   # or tiny/base/small/medium
    python scripts/setup_whisper_model.py --variant large-v3-turbo    # surgery + fix_shapes

Per-variant EP correctness is covered by
`test/python/whisper/test_whisper_variant_smoke.py` (EP-vs-CPU greedy token match,
on-demand build, skips cleanly if a variant is not built).
```

(Verify the layer/head/vocab numbers against the actual built `config.json` during the gfx1151 build in Task 5; correct any cell that differs.)

- [ ] **Step 2: Update `CLAUDE.md`**

In the Whisper model row / notes area (the Test Models section + the Whisper notes), add a sentence noting multi-variant support and the 80-mel family. Minimal, factual addition — e.g. after the existing Whisper-large-v3 row note:

```markdown
**Whisper variants (turbo + tiny/base/small/medium) are supported** via the same
architecture + decoder surgery — only shape params, `n_mels` (80 for the small
sizes, 128 for turbo/large-v3), vocab, and special-token IDs differ (`head_dim=64`,
decoder context 448 for all). Shapes are derived from each model's
`transformers.WhisperConfig`; forced-start tokens from its tokenizer (large-v3's
51866 layout shifts the task tokens by one vs the 51865 multilingual layout).
Sourcing: large-v3 downloads from AMD HF; the others are local OGA builds
(`python build.py --build-whisper-models --variant <name>`). Per-variant EP
correctness: `test/python/whisper/test_whisper_variant_smoke.py` asserts EP-GPU
greedy tokens == ORT-CPU greedy tokens on the SAME variant (NOT verbatim text —
small models are low-accuracy, so EP-vs-CPU isolates EP correctness). large-v3
keeps its full correctness + verbatim-JFK matrix in `test_whisper.py`.
```

- [ ] **Step 3: Commit**

```bash
git add docs/whisper_quick_start.md CLAUDE.md
git commit -m "docs(whisper): document multi-variant support (turbo + small sizes)"
```

---

## Self-Review

**Spec coverage:**
- n_mels=80 family → Task 1 (config derivation) + Task 4 (variant-aware encoder/audio) + Task 5 (smoke covers tiny=80-mel). ✓
- Variant registry + config-derived shapes → Task 1. ✓
- Local OGA build primary → Task 2 (builder `--variant`) + Task 3 (`_ensure_whisper_raw` local-build-primary for non-large-v3). ✓
- EP-vs-CPU smoke oracle → Task 5. ✓
- build_whisper_models.py / conftest setup / make_whisper_inputs / test_whisper / setup_whisper_model.py / docs → Tasks 2,3,4(make_whisper_inputs already variant-aware via cfg arg),5,6,7. ✓
- Risk verification (graph layout, n_mels conv, small-model accuracy) → Task 5 Step 5 [gfx1151]. ✓
- large-v3 unchanged → Task 3 keeps large-v3 wrappers on the old path; Task 4 `variant=None` default; Task 4 Step 5 regression check. ✓

Note: `make_whisper_inputs(audio, cfg)` already takes a config and uses `WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3")`. For 80-mel variants the FE must produce 80 mels. **Gap fix folded into Task 4:** make `make_whisper_inputs` choose the FE by mel count — add a `fe_id` derived from `cfg.n_mels` (128 → `openai/whisper-large-v3`, 80 → `openai/whisper-tiny`; both are canonical FE configs for their mel count). Add to Task 4 Step 3:

```python
# in conftest.make_whisper_inputs — pick the feature extractor by mel count, so
# 80-mel variants get an 80-mel log-mel and 128-mel (large-v3/turbo) get 128.
fe_id = "openai/whisper-large-v3" if cfg.n_mels == 128 else "openai/whisper-tiny"
fe = WhisperFeatureExtractor.from_pretrained(fe_id)
```
and a unit assertion in `test_whisper_variants_unit.py` is not possible offline (needs transformers FE); covered functionally by Task 5's tiny run. This edit is in `conftest.make_whisper_inputs`, so add it to Task 4's file list (Modify: `test/python/conftest.py`) and commit message scope.

**Placeholder scan:** No TBD/TODO/"handle edge cases". Revisions for new variants are `"main"` (a real, working value; commit-pinning is a documented follow-up, not a placeholder). ✓

**Type consistency:** `_runtime_params` 7-tuple order is identical in Task 4 definition, test, and the unpacking instructions. `setup_whisper_variant` returns `(model_dir, WhisperVariant)` consistently across Tasks 3/5/6. `whisper_output_dirs` (build script) vs `whisper_model_dir` (conftest) are distinct names with the same convention, both comment-gated. ✓
