#!/usr/bin/env python
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Standalone builder for Whisper variant ONNX models (fp32 + fp16).

Both precisions are produced by the SAME pinned OGA DirectML model builder, so
pinning the builder deps + the HF model revision makes both models reproducible.
This REPLACES the old external fp32 download (tonythethompson/whisper-large-v3-genai),
which was non-reproducible.

Isolation: the OGA model builder needs the STOCK onnxruntime-genai-directml, which
shadows the OGA FORK that build.py --build-oga installs (both import as
`onnxruntime_genai`). To keep the dev env's fork intact, this script builds the
models in a DEDICATED, PINNED venv (install/whisper-builder-venv/) and runs the
builder via subprocess in that venv. The dev env is never touched.

Default precision is fp16: the added variants (turbo + small sizes) only need
fp16 (all variant tests run fp16), which halves build + disk + test work. fp32
is still available via --precision (large-v3 keeps its fp32-vs-fp32 cross-backend
benchmark, sourced by download in conftest).

Run:
    conda activate hip-ep
    python scripts/build_whisper_models.py            # build default variants (fp16)
    python scripts/build_whisper_models.py --precision both   # fp32 + fp16
    python scripts/build_whisper_models.py --list     # show what would be built
    python scripts/build_whisper_models.py --variant tiny --variant base  # specific variants

Outputs (raw OGA bundles; conftest then surgeries + fix_shapes them):
    models/whisper-{variant}-onnx/        fp32: encoder/decoder.onnx (+.data) + configs
    models/whisper-{variant}-onnx-fp16/   fp16: same layout, fp16 body + fp32 lm_head
"""

import argparse
import subprocess
import venv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS = REPO_ROOT / "models"
BUILDER_VENV = REPO_ROOT / "install" / "whisper-builder-venv"

# ---------------------------------------------------------------------------
# Pinned per-variant sources (same idiom as build.py's version pins).
# DUPLICATED from conftest.WHISPER_VARIANTS so this builder stays standalone
# (no test-infra import). test_whisper_build_paths.py asserts the two match.
# ---------------------------------------------------------------------------
VARIANT_SOURCES = {
    # name            (hf_model_id,                    revision)
    "large-v3": ("openai/whisper-large-v3", "06f233fe06e710322aca913c1bc4249a0d71fce1"),
    "large-v3-turbo": ("openai/whisper-large-v3-turbo", "main"),
    "tiny": ("openai/whisper-tiny", "main"),
    "base": ("openai/whisper-base", "main"),
    "small": ("openai/whisper-small", "main"),
    "medium": ("openai/whisper-medium", "main"),
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


def whisper_output_dirs(variant: str) -> "tuple[Path, Path]":
    """Return (fp32_dir, fp16_dir) under models/ for a variant.

    Naming convention MUST match conftest (models/whisper-{variant}-onnx[-fp16]).
    """
    return (
        MODELS / f"whisper-{variant}-onnx",
        MODELS / f"whisper-{variant}-onnx-fp16",
    )


def _venv_python(vdir: Path) -> Path:
    # Windows venv layout: Scripts/python.exe
    return vdir / "Scripts" / "python.exe"


def ensure_builder_venv() -> Path:
    """Create install/whisper-builder-venv/ with the pinned deps. Idempotent
    (a .ok sentinel records that the pinned requirements were installed)."""
    sentinel = BUILDER_VENV / ".deps.ok"
    py = _venv_python(BUILDER_VENV)
    if sentinel.exists() and py.exists():
        print(f"[whisper-build] builder venv ready: {BUILDER_VENV}")
        return py

    if not py.exists():
        print(f"[whisper-build] creating builder venv at {BUILDER_VENV} ...")
        venv.create(str(BUILDER_VENV), with_pip=True)

    print(f"[whisper-build] installing pinned builder deps: {BUILDER_REQUIREMENTS}")
    subprocess.run([str(py), "-m", "pip", "install", "--upgrade", "pip"], check=True)
    subprocess.run([str(py), "-m", "pip", "install", *BUILDER_REQUIREMENTS], check=True)
    sentinel.touch()
    print("[whisper-build] builder venv deps installed.")
    return py


# The build runs INSIDE the venv via `python -c <BUILDER_SNIPPET>` so the stock
# OGA import resolves there, not in the dev env. The qwen-class stub makes OGA's
# eager builders/__init__ import succeed under transformers 4.49.0 (the Whisper
# builder never uses those classes). This snippet is the SINGLE source of the OGA
# `create_model(... precision=..., execution_provider="dml" ...)` invocation for
# both precisions (the old test/python/whisper/build_fp16_model.py was retired).
_NUM_STUBS = 10
BUILDER_SNIPPET = r"""
import re, sys
import transformers
_STUBBED = set()
def _import_with_stubs(importer):
    # OGA's builders/__init__ eagerly imports qwen.py (refs transformers classes
    # absent in the pinned transformers); register empty stubs as each missing
    # name surfaces. Used for BOTH the create_model import and the later
    # builders.whisper import (the asymmetric-layer patch) since either can be
    # the first to trigger the eager qwen import.
    for _ in range(%d):
        try:
            return importer()
        except ImportError as e:
            m = re.search(r"cannot import name '([^']+)'", str(e))
            # A non-"cannot import name" ImportError (e.g. OGA itself not
            # installed in the venv) is a real failure — surface it, don't stub.
            if not m: raise
            name = m.group(1)
            # If we already force-stubbed this exact name yet the import still
            # fails, something else is wrong -> re-raise instead of looping.
            # (Do NOT key on hasattr: transformers' _LazyModule reports
            # hasattr==True for names declared in its _import_structure that it
            # nonetheless cannot actually import, which would make us give up on
            # the very first missing class.)
            if name in _STUBBED: raise
            _STUBBED.add(name)
            # Stub onto sys.modules["transformers"], NOT the module captured by
            # `import transformers` at the top: transformers' _LazyModule
            # rebinds itself in sys.modules when first materialized (which the
            # revision-wrapping block above does via getattr), and `from
            # transformers import <name>` resolves against the sys.modules entry.
            # Setting the attr on the stale top-level binding silently no-ops for
            # the from-import on retry.
            setattr(sys.modules["transformers"], name, type(name, (), {}))
    raise RuntimeError(
        "could not satisfy OGA builder imports after %d transformers stubs")
def _imp_create_model():
    from onnxruntime_genai.models.builder import create_model  # noqa
    return create_model
model_id, revision, precision, out_dir, cache_dir = sys.argv[1:6]

# --- Pin the HF revision so the weights are byte-stable across runs. ----------
# WHICH CALL PATH the OGA Whisper builder actually uses (verified against the
# stock onnxruntime-genai 0.13.1 source):
#   create_model -> transformers.AutoConfig.from_pretrained                (config)
#                -> base.Model.make_model -> AutoModelForSpeechSeq2Seq.from_pretrained  (WEIGHTS)
#                -> make_genai_config    -> AutoConfig / GenerationConfig.from_pretrained
#                -> save_processing      -> AutoTokenizer.from_pretrained   (tokenizer)
# ALL of these go through transformers -> `cached_file` -> hf_hub_download PER FILE.
# NONE of them call huggingface_hub.snapshot_download, and none thread a
# `revision=` kwarg, and there is no env var that hf_hub honors for the default
# revision (huggingface_hub hardcodes DEFAULT_REVISION="main"). So the ONLY
# reliable way to pin the revision is to default it into the from_pretrained
# calls the builder makes, plus hf_hub_download as a per-file catch-all.
import huggingface_hub
import transformers

def _wrap_revision(fn):
    def _wrapped(*a, **k):
        k.setdefault("revision", revision)
        return fn(*a, **k)
    return _wrapped

# from_pretrained classmethods are the primary hooks (config/weights/tokenizer).
for _cls_name in (
    "AutoConfig",
    "AutoModelForSpeechSeq2Seq",  # Whisper weights
    "AutoTokenizer",              # save_processing
    "AutoProcessor",              # belt-and-suspenders (not used by Whisper today)
    "GenerationConfig",           # make_genai_config
):
    _cls = getattr(transformers, _cls_name, None)
    if _cls is not None and hasattr(_cls, "from_pretrained"):
        _cls.from_pretrained = _wrap_revision(_cls.from_pretrained)

# Per-file catch-all: every transformers `cached_file` lands here eventually.
huggingface_hub.hf_hub_download = _wrap_revision(huggingface_hub.hf_hub_download)
# Harmless belt-and-suspenders (the builder does not actually call this).
huggingface_hub.snapshot_download = _wrap_revision(huggingface_hub.snapshot_download)

# Import create_model first. This is also what makes the bare `builders` package
# importable: onnxruntime_genai puts its models/ dir on sys.path, so builder.py's
# module-level `from builders import (...)` loads the builders package a SECOND
# time under the top-level name `builders` (distinct module object from
# `onnxruntime_genai.models.builders`, with DISTINCT class objects). create_model
# uses the bare `builders.whisper.WhisperEncoder`, so the asymmetric-layer patch
# below must target THAT class, not the qualified copy. (Doing this import here
# also resolves OGA's eager qwen import via _import_with_stubs.)
create_model = _import_with_stubs(_imp_create_model)

# --- Fix OGA 0.13.1 asymmetric-layer Whisper (e.g. large-v3-turbo). -----------
# The stock WhisperEncoder builds the cross-attention KV cache outputs
# (present_key_cross_i / present_value_cross_i, and the Reshape/Transpose
# postprocessing that feeds them) with `range(self.num_layers)`, where
# num_layers is the ENCODER layer count (it sets config.num_hidden_layers =
# config.encoder_layers in __init__). But cross-attention KV is produced once
# per DECODER layer and consumed by the decoder's past_key_cross_i inputs, so
# the count MUST be config.decoder_layers. For symmetric variants
# (tiny/base/small/medium/large-v3) encoder_layers == decoder_layers, so this is
# a no-op; for the pruned-decoder turbo (32 enc / 4 dec) the stock code
# IndexErrors at decoder.layers[4] ("index 4 is out of range"). Every use of
# self.num_layers inside make_inputs_and_outputs / make_postprocessing_nodes of
# the ENCODER is the cross-KV count, so temporarily swapping it to the captured
# decoder-layer count around each call is a minimal, exact fix. decoder_layers
# is captured at __init__ time because base.Model does not store self.config and
# the decoder builder later runs with the same config object.
try:
    # Patch every loaded WhisperEncoder that shares the builder source file:
    # both the bare `builders.whisper` (the copy create_model actually uses) and
    # the qualified `onnxruntime_genai.models.builders.whisper`, since either may
    # be present in sys.modules and they are distinct class objects.
    _enc_classes = []
    for _modname in ("builders.whisper",
                     "onnxruntime_genai.models.builders.whisper"):
        _m = sys.modules.get(_modname)
        if _m is not None and hasattr(_m, "WhisperEncoder"):
            _enc_classes.append(_m.WhisperEncoder)
    if not _enc_classes:
        raise RuntimeError("no loaded WhisperEncoder class found to patch")

    def _with_cross_kv_count(orig):
        def _wrapped(self):
            n_cross = getattr(self, "_cross_kv_layers", None)
            if n_cross is None or n_cross == self.num_layers:
                return orig(self)  # symmetric (or unknown) -> stock behavior
            saved = self.num_layers
            self.num_layers = n_cross
            try:
                return orig(self)
            finally:
                self.num_layers = saved
        return _wrapped

    def _make_init(orig_init):
        def _patched(self, config, *a, **k):
            # decoder_layers stays intact on config (only num_hidden_layers is
            # mutated by __init__), so capture it before delegating.
            n_cross = getattr(config, "decoder_layers", None)
            orig_init(self, config, *a, **k)
            self._cross_kv_layers = n_cross
        return _patched

    for _Enc in _enc_classes:
        _Enc.__init__ = _make_init(_Enc.__init__)
        _Enc.make_inputs_and_outputs = _with_cross_kv_count(
            _Enc.make_inputs_and_outputs)
        _Enc.make_postprocessing_nodes = _with_cross_kv_count(
            _Enc.make_postprocessing_nodes)
except Exception as _e:  # pragma: no cover - defensive; never block symmetric builds
    print("[builder-subproc] WARN: asymmetric-Whisper patch not applied:", _e)

create_model(
    model_name=model_id, input_path="", output_dir=out_dir,
    precision=precision, execution_provider="dml", cache_dir=cache_dir,
)
print("[builder-subproc] done:", precision, "->", out_dir)
""" % (_NUM_STUBS, _NUM_STUBS)


def build_one(
    py: Path, model_id: str, revision: str, precision: str, out_dir: Path
) -> None:
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
        [
            str(py),
            "-c",
            BUILDER_SNIPPET,
            model_id,
            revision,
            precision,
            str(out_dir),
            str(cache_dir),
        ],
        check=True,
    )


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
        "--precision",
        choices=["fp32", "fp16", "both"],
        default="fp16",
        help="which precision(s) to build (default: fp16). The added variants "
        "(turbo + small sizes) only need fp16; pass fp32/both for large-v3's "
        "cross-backend fp32-vs-fp32 benchmark.",
    )
    ap.add_argument(
        "--list",
        action="store_true",
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


if __name__ == "__main__":
    raise SystemExit(main())
