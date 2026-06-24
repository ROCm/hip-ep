# Whisper Variant Support — Design

**Date:** 2026-06-23
**Branch / worktree:** `whisper-variants` (off `fix/whisper-ci-memory-pressure`)
**Status:** Approved design — ready for implementation plan

## Goal

Extend the existing Whisper-large-v3 support to additional Whisper variants:

- **large-v3-turbo** (128-mel; 32 encoder layers + **4** decoder layers)
- **small sizes**: tiny / base / small / medium (all **80-mel**)

large-v3 itself stays exactly as-is.

## Key architectural insight

Every Whisper variant shares the **same architecture and the same decoder
surgery**. The MorphiZen decoder surgery (`inject_seqlens_k`) keys on node-name
patterns produced by the OGA DirectML model builder — `MultiHeadAttention`,
`past_key_self_*` in slot 6, `embed_positions.weight` Slice — and on
`n_text_ctx = 448`, which is **constant across all Whisper sizes**. The surgery
loops over all nodes, so per-variant layer-count differences (e.g. turbo's 4
decoder layers) are absorbed automatically.

Therefore this is a **parameterization**, not a rewrite. Only two things differ
per variant:

1. **Shape parameters** — `n_audio_state` / `n_text_state`, encoder & decoder
   layer counts, head counts, `n_vocab`, and **`n_mels` (80 for the small
   sizes, 128 for turbo/large-v3)**. `n_mels` is the only genuinely *new*
   dimension family introduced by this work.
2. **Model source** — HF model id + pinned revision.

## Chosen approach: variant registry + config-derived shapes

Rejected alternatives:
- **Per-variant copies of the setup functions** — heavy duplication, rots
  quickly. Rejected.
- **Fully config-driven with zero registry** — most robust to params but needs
  more upfront plumbing and still needs *somewhere* to pin HF id + revision.

**Selected:** a small `WHISPER_VARIANTS` registry holding only `{hf_model_id,
pinned_revision}` per variant, plus **deriving the shape parameters from the
built model's `config.json` / `genai_config.json`** instead of maintaining a
brittle hardcoded dimension table. The `WhisperModelConfig` dataclass remains as
the large-v3 default / fallback.

### Decisions locked for this design

1. **Smoke-test oracle = "same-variant EP-GPU greedy tokens == ORT-CPU greedy
   tokens".** NOT "verbatim equals the JFK quote". Small models (tiny/base) have
   low transcription accuracy and may not emit the exact JFK text, so an
   absolute-text assertion would be a model-quality test, not an EP-correctness
   test. Comparing EP vs CPU on the *same* variant isolates EP correctness and is
   immune to model accuracy. (large-v3's existing verbatim-JFK e2e assertion is
   left unchanged.)
2. **Shapes are derived from the built config**, not hardcoded.
3. **`build_whisper_models.py` default builds turbo + all small sizes** when
   `--variant` is omitted; `--variant` accepts an explicit subset.

## Model source

**Local OGA build is primary.** No AMD-hosted ONNX repos exist for
turbo/tiny/base/small/medium, so these are built locally by the existing pinned
OGA DirectML builder (`scripts/build_whisper_models.py`) from
`openai/whisper-{variant}`. large-v3 keeps its AMD-HF-download-with-local-backup
behavior. The builder snippet itself is unchanged (same
`AutoModelForSpeechSeq2Seq` path works for all sizes).

## Per-file changes

### `scripts/build_whisper_models.py`
- Introduce a per-variant table: `variant → (hf_model_id, pinned_revision)`.
- Add `--variant` (accepts one or more; default = turbo + tiny/base/small/medium).
- Output dirs: `models/whisper-{variant}-onnx[-fp16]`.
- Builder snippet unchanged; still builds fp32 + fp16 per variant.

### `test/python/conftest.py`
- Add `WHISPER_VARIANTS` registry (`name → {hf_model_id, pinned_revision}`).
- Add a helper that reads the built `config.json` / `genai_config.json` to
  populate a `WhisperModelConfig` (n_mels, state, enc/dec layers, heads, vocab).
  Keep the dataclass defaults as the large-v3 fallback.
- Generalize the setup path into `setup_whisper_variant(variant, precision)`:
  resolve `model_dir` → ensure raw bundle (local build primary; large-v3 may
  download from AMD) → `inject_seqlens_k` surgery → `fix_shapes` with the
  invariant `n_text_ctx = 448`. The existing
  `setup_whisper_model_dir` / `setup_whisper_fp16_model_dir` become thin
  large-v3 wrappers over it (no behavior change for large-v3).
- `make_whisper_inputs`: pull the `WhisperFeatureExtractor` from the variant's
  own HF id so it emits the correct `[1, n_mels, 3000]` (80 vs 128).

### `test/python/whisper/test_whisper.py`
- Add a `@pytest.mark.parametrize` smoke test over turbo + small sizes.
  - Ensures the variant is built/setup, runs EP-GPU greedy decode and ORT-CPU
    greedy decode on the same variant + same audio, asserts token-sequence
    equality (the oracle above).
  - Reuse the silent-CPU-fallback tripwire (`_assert_compiled_on_gpu`) so a
    fallback can't pass as a CPU-vs-CPU match.
- large-v3's full matrix (encoder/prefill/decode cosine + verbatim-JFK e2e +
  perf, fp16 + fp32) is untouched.

### `scripts/setup_whisper_model.py`
- Add `--variant` so the standalone prep wrapper can target any variant.

### Docs (same PR — repo rule: docs must not go stale)
- `docs/whisper_quick_start.md` — supported-variant matrix + build/setup recipe.
- `CLAUDE.md` — update the Whisper rows / notes to reflect multi-variant support
  and the 80-mel family.

## Risks to verify during implementation

1. **OGA builder graph layout consistency** for turbo/small — the surgery's
   node-name matching must still hold. Verify by actually building + surgering at
   least one small size **and** turbo, and confirming `inject_seqlens_k` reports
   a non-zero self-attn count and a replaced position-embedding Slice.
2. **n_mels=80 encoder rank-3 Conv** correctness on the EP — should reuse the
   existing conv1d path, but confirm via the smoke test's EP-vs-CPU match.
3. **Small-model accuracy** — already mitigated by the EP-vs-CPU oracle.

## Out of scope

- Pushing built ONNX to AMD HF repos.
- Downloading community pre-built ONNX (not OGA-built / un-surgered).
- Per-variant perf benchmarking and full correctness matrices (large-v3 only).
- distil-whisper and the full openai size spectrum beyond the chosen set.

## Success criteria

- `python build.py --build-whisper-models --variant <v>` (or default set)
  produces raw fp32 + fp16 bundles for turbo + the small sizes.
- `setup_whisper_variant` surgeries + fixes shapes for each, with
  `inject_seqlens_k` reporting a non-zero self-attn count.
- The parametrized smoke test passes for turbo + small sizes: EP-GPU greedy
  tokens match ORT-CPU greedy tokens on the same variant, with the GPU-dispatch
  tripwire confirming no silent CPU fallback.
- large-v3's existing tests remain green and unchanged.
- `docs/whisper_quick_start.md` and `CLAUDE.md` reflect the new support.
