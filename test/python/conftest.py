#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Shared fixtures and helpers for Python performance tests."""

import contextlib
import gc
import hashlib
import itertools
import json
import os
import pathlib
import shutil
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, ClassVar, Optional

import numpy as np
import onnxruntime as ort
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent

# Default to strict EP mode for all tests so any silent CPU-fallback
# regression aborts loudly with a stack trace (CLAUDE.md "HIPDNN_EP_STRICT=1"
# gotcha). Multimodal-OGA scenarios that intentionally have sub-session
# graph-claim-then-fail behaviour (currently: gemma3 OGA tests, which load
# embedding/vision/speech alongside the text decoder) opt out per ModelSpec
# via `oga_strict=False` — the OGA Model load there is wrapped in
# `hipdnn_strict_for_oga(spec)` which temporarily unsets the env. Set
# externally to override (e.g. `HIPDNN_EP_STRICT=0 pytest ...`).
os.environ.setdefault("HIPDNN_EP_STRICT", "1")

NUM_WARMUP = 1
NUM_RUNS = 3

# The EP is now reached through the AMD GPU umbrella EP (amdgpu-ep.dll), which
# loads hip-backend.dll → hipgpu.dll (the renamed MorphiZen EP). The umbrella
# selects the backend via the "profile" provider option (see EP_PROVIDER_OPTIONS).
EP_DLL_NAME = "amdgpu-ep.dll"
EP_REGISTRATION_NAME = "AMDGPUExecutionProvider"
# OGA registers the umbrella under its short name (genai_config key).
EP_OGA_NAME = "AMDGPU"
# Provider option the umbrella forwards to pick the hipgpu backend.
EP_PROVIDER_OPTIONS = {"profile": "llm"}

# Optional artifact-format override (escape hatch). Production / CI use the
# default in-process LLVM-IR (bitcode) JIT, so this is UNSET by default and the
# tests run on bitcode. Set HIPEP_ARTIFACT_FORMAT=NATIVE to compile each model to
# a per-model DLL (lld-link + LoadLibrary) instead. Normally not needed — build
# the EP against a Tier-1 `llvm-install` (see docs/whisper_quick_start.md §1) and
# bitcode works; NATIVE is just a fallback. The value rides through as a raw
# `ep.hipgpu.*` session-config entry (NOT a provider option: ORT validates those
# against the umbrella's declared set and rejects an unknown key; the umbrella
# forwards non-umbrella-prefixed config entries to the backend verbatim). Mirrors
# CI's `-C ep.hipgpu.artifact_format|NATIVE`.
ARTIFACT_FORMAT_ENV = "HIPEP_ARTIFACT_FORMAT"


def apply_artifact_format(so):
    """Apply the optional HIPEP_ARTIFACT_FORMAT override to a SessionOptions.

    No-op (bitcode default) when the env var is unset/empty.
    """
    fmt = os.environ.get(ARTIFACT_FORMAT_ENV, "").strip()
    if fmt:
        so.add_session_config_entry("ep.hipgpu.artifact_format", fmt)


_morphizen_registered = False


def _ep_runtime_dirs(repo_root):
    """Return (ep_bin, therock_bin) for locating the EP DLL + ROCm runtime DLLs.

    Honours an out-of-tree install layout via env vars (set them when you build
    with a custom --install_dir / --build_dir, e.g. the quick-start $ROOT layout):
      HIPEP_EP_BIN      -- dir holding the AMD GPU umbrella EP chain
                           (amdgpu-ep.dll + hip-backend.dll + hipgpu.dll)
                           (default <repo>/install/dist/bin)
      THEROCK_DIST      -- TheRock SDK root; its bin/ holds amdhip64*.dll etc.
                           (default <repo>/install/therock)
    Falls back to the legacy in-repo layout so existing setups keep working.
    The legacy MORPHIZEN_EP_BIN name is still honoured as a fallback so older
    local setups / scripts keep working during the rename.
    """
    ep_env = os.environ.get("HIPEP_EP_BIN") or os.environ.get("MORPHIZEN_EP_BIN")
    ep_bin = pathlib.Path(ep_env) if ep_env else repo_root / "install" / "dist" / "bin"
    therock_env = os.environ.get("THEROCK_DIST")
    therock_bin = (
        pathlib.Path(therock_env) / "bin"
        if therock_env
        else repo_root / "install" / "therock" / "bin"
    )
    return ep_bin, therock_bin


# ── Model download / shape fixing ────────────────────────────────────────────


def download(url: str, dest: pathlib.Path) -> None:
    print(f"  Downloading {url}")
    urllib.request.urlretrieve(url, str(dest))
    mb = dest.stat().st_size / (1024 * 1024)
    print(f"  Saved {dest.name} ({mb:.1f} MB)")


def setup_jfk_sample(data_dir: pathlib.Path) -> bool:
    """Idempotently fetch ``jfk.wav`` (the JFK inaugural-address excerpt) into
    ``data_dir/jfk.wav``.

    Source: the whisper.cpp repo's ``samples/jfk.wav`` — a 16 kHz mono 16-bit PCM
    WAV mirror of the clip openai/whisper ships only as FLAC. We download rather
    than commit the binary so the PR stays free of test-data blobs (the file is
    gitignored). Provenance / license is documented in
    ``test/python/data/whisper/README.md``.

    Returns ``True`` if the file is present (already cached or freshly fetched),
    ``False`` if the network fetch failed and nothing is cached (caller should
    ``pytest.skip``). Never raises on a fetch failure.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    dest = data_dir / "jfk.wav"
    if dest.exists():
        return True  # idempotent — already cached
    url = "https://github.com/ggml-org/whisper.cpp/raw/master/samples/jfk.wav"
    try:
        download(url, dest)
    except Exception as e:  # network / 404 / partial write
        print(f"  [jfk] fetch failed ({e!r})")
        # Clean up any partial file so the existence gate above stays honest.
        dest.unlink(missing_ok=True)
        return False
    return True


def fix_shapes(src: pathlib.Path, dst: pathlib.Path, dim_map: dict) -> None:
    import onnx

    m = onnx.load(str(src), load_external_data=False)
    for tensor in list(m.graph.input) + list(m.graph.output) + list(m.graph.value_info):
        if not tensor.type.tensor_type.HasField("shape"):
            continue
        for d in tensor.type.tensor_type.shape.dim:
            if d.dim_param in dim_map:
                v = dim_map[d.dim_param]
                d.Clear()
                d.dim_value = v
    onnx.save(m, str(dst), save_as_external_data=False)
    print(f"  Fixed-shape model -> {dst.name}")


def _prune_dead_nodes(graph) -> int:
    """Erase nodes whose every output is neither consumed by another node nor a
    graph output. Iterates to a fixed point so a dead chain (e.g.
    Shape→Gather→Add→Unsqueeze) is removed end-to-end: deleting the tail
    Unsqueeze makes the Add dead on the next pass, then the Gather, then the
    Shape. Returns the number of nodes removed. Initializers/graph inputs are
    never touched — only graph.node entries. Behavior-preserving: a node deleted
    here is by definition unreachable from any graph output.
    """
    removed = 0
    graph_out = {o.name for o in graph.output}
    changed = True
    while changed:
        changed = False
        consumed = {i for n in graph.node for i in n.input if i}
        for node in list(graph.node):
            if all(o not in consumed and o not in graph_out for o in node.output):
                graph.node.remove(node)
                removed += 1
                changed = True
    return removed


def inject_seqlens_k(src: pathlib.Path, dst: pathlib.Path) -> None:
    """Inject the ``past_sequence_length`` input into all self-attention
    ``MultiHeadAttention`` nodes in a Whisper decoder ONNX. Cross-attention MHAs
    (identified by an empty or ``past_key_cross_*`` ``input[6]``) are untouched.

    Self-attn detection: ``input[6].startswith("past_key_self_")``. We pad
    ``node.input`` to length 8 (empty strings for missing slots — bias /
    key_padding_mask / attention_bias are optional in MHA) and append
    ``input[8] = "past_sequence_length"`` per the MHA spec.

    The ``past_sequence_length`` input (MHA spec slot 9) is the ONLY signal the
    MorphiZen converter needs to recognise post-surgery decoder self-attn
    (``MultiHeadAttentionConversion.cpp`` branch 2 keys on the operand's
    presence). We deliberately do NOT add a ``past_present_share_buffer``
    attribute: that attribute is NOT in ORT's ``com.microsoft.MultiHeadAttention``
    schema (it lives on ``GroupQueryAttention``/``Attention``), so emitting it
    makes ORT reject the graph at session load with "Unrecognized attribute:
    past_present_share_buffer for operator MultiHeadAttention" — before the EP
    ever sees the model. Threading the input alone keeps the graph ORT-valid
    while still driving the shared-buffer KV path.

    Idempotency contract: this helper short-circuits when ``dst`` exists.
    Callers MUST delete ``dst`` when ``src`` changes, otherwise stale output
    will be reused silently.
    """
    if dst.exists():
        return  # idempotent
    import onnx
    from onnx import TensorProto, helper

    m = onnx.load(str(src), load_external_data=False)
    # Add new graph input (skip if a prior run already added it)
    if "past_sequence_length" not in {i.name for i in m.graph.input}:
        m.graph.input.append(
            helper.make_tensor_value_info(
                "past_sequence_length", TensorProto.INT32, [1]
            )
        )
    self_count = 0
    for node in m.graph.node:
        if node.op_type != "MultiHeadAttention":
            continue
        # Self-attn nodes have past_key_self_* in slot 6; cross-attn have
        # either no slot 6 (3-input form) or a non-self past_key reference.
        if len(node.input) < 7 or not node.input[6].startswith("past_key_self_"):
            continue
        # Pad to 8 inputs (bias / key_padding_mask / attention_bias may be
        # absent), then append slot 8 = past_sequence_length. The while-loop
        # above guarantees len(node.input) == 8 on exit, so a plain append
        # always lands at index 8 — no need for a separate overwrite branch.
        while len(node.input) < 8:
            node.input.append("")
        node.input.append("past_sequence_length")
        # NOTE: do NOT add a past_present_share_buffer attribute here — ORT's
        # com.microsoft.MultiHeadAttention schema does not recognise it and
        # would reject the graph at load. The slot-8 past_sequence_length input
        # is the sole signal the MorphiZen converter keys on (see
        # MultiHeadAttentionConversion.cpp branch 2).
        self_count += 1

    # ── Replace the position-embedding Slice with a Gather(position_ids) ───────
    #
    # The decoder's position-embedding preprocessing computes a slice offset into
    # ``decoder.embed_positions.weight`` from ``Shape(past_key_self_0)[2]`` (the
    # past-KV seq dim) and does ``Slice(embed_positions, [offset], [offset+S])``.
    # In the original DYNAMIC model the past-KV seq dim == the real past length
    # (0 at prefill, growing). But shared-buffer ``fix_shapes`` pins that dim to
    # the STATIC 448-slot buffer, so the offset becomes a constant 448 → the
    # Slice reads ``embed_positions[448:452]`` → empty {0,1280} → "Add
    # Incompatible dimensions" at load.
    #
    # The OBVIOUS fix (re-source the offset from ``past_sequence_length`` and
    # rebuild the Slice bounds) does NOT work on the MorphiZen GPU path: a Slice
    # with RUNTIME-data-dependent start/end falls through to the ``hip.slice``
    # runtime op, whose start/end come from int64 SCALAR arithmetic
    # (Cast/Gather/Add/Sub) that the GPU lowering mishandles — int64 scalar Add
    # lowers to ``miopenOpTensor`` and fails ("descriptor cache creation
    # failed"), so the bounds resolve to 0, ``wrap_slice`` computes extent 0 and
    # aborts ("derived output extent != IR output_shape"), and the
    # position-embedding output is left ZERO. Zeroed position embeddings corrupt
    # the hidden state going into layer 0 → wrong present_key_self_0 → broken
    # logits (prefill argmax 50360 instead of 400). This was the LAST greedy
    # blocker, and it is a GPU-runtime limitation (runtime-bounds Slice +
    # int64-scalar arith), not a graph-semantics bug — the SAME surgered graph
    # run on the ORT CPU EP produces the correct embedding (cosine 0.99999).
    #
    # Fix: eliminate the runtime-bounds Slice AND the scalar arithmetic entirely.
    # Add a ``position_ids`` int64 [sequence_length] graph input and replace the
    # Slice with ``Gather(embed_positions.weight, position_ids, axis=0)``. A
    # Gather with a runtime int64 *indices input* lowers to ``hip.gather``
    # (wrap_gather), which IS correct on GPU (verified). The caller feeds
    # ``position_ids = [real_past, real_past+1, ..., real_past+S-1]`` where
    # ``real_past = total_tokens - S`` (0 at prefill, the running past count at
    # decode) — the SAME values the original Slice would have produced, now
    # computed host-side where it is trivial. This also decouples the
    # position offset from ``past_sequence_length`` (which still feeds the 32
    # self-attn MHAs as seqlens_k under the runtime's total = seqlens_k + 1
    # convention), removing the two-consumers-need-different-values entanglement.
    slice_node = None
    for node in m.graph.node:
        if (
            node.op_type == "Slice"
            and node.input
            and node.input[0].endswith("embed_positions.weight")
        ):
            slice_node = node
            break
    if slice_node is not None:
        if "position_ids" not in {i.name for i in m.graph.input}:
            m.graph.input.append(
                helper.make_tensor_value_info(
                    "position_ids", TensorProto.INT64, ["sequence_length"]
                )
            )
        embed_positions = slice_node.input[0]
        slice_out = slice_node.output[0]
        gather_node = helper.make_node(
            "Gather",
            [embed_positions, "position_ids"],
            [slice_out],  # reuse the Slice's output name → consumers unchanged
            name="surgery/pos_embed_gather",
            axis=0,
        )
        # Remove the Slice. Its now-dead index-producer chain (the
        # Shape→Gather→Add→Unsqueeze that computed the old [offset, offset+S]
        # bounds) becomes unreachable from any graph output; the _prune_dead_nodes
        # sweep below erases it so the saved graph is clean. (ONNX/the EP would
        # ignore the dead nodes at load anyway — this is purely for a tidy graph.)
        m.graph.node.remove(slice_node)
        m.graph.node.insert(0, gather_node)
        print(
            "  Replaced position-embedding Slice with "
            "Gather(embed_positions, position_ids) (shared-buffer GPU fix)"
        )

    # ── Widen input_ids int32 -> int64 (token-embedding Gather indices) ────────
    #
    # The MorphiZen ``hip_gather`` custom kernel hardcodes the INDICES operand as
    # ``int64_t*`` (gather_kernel.hip ``gather_axis0_kernel``); it has no int32
    # indices path. Whisper's ``input_ids`` is INT32, so the token-embedding
    # ``Gather(embed_tokens.weight, input_ids)`` reads the int32 indices as int64
    # on GPU → garbage indices → the kernel's bounds check zeros every row → the
    # token embedding is ALL ZEROS (verified: int32-indices Gather cos -0.016 /
    # row0 = 0 vs int64-indices Gather cos 1.0). That zeroed token embedding (on
    # top of the now-correct position embedding) corrupts the hidden state into
    # layer 0 → broken logits.
    #
    # An in-graph ``Cast(input_ids int32->int64)`` does NOT work either: the
    # ``hip_cast`` kernel rejects int32->int64 ("unsupported conversion
    # input_dtype=3 -> output_dtype=2"), producing zero indices. So fix at the
    # graph BOUNDARY: re-type the ``input_ids`` graph input to INT64 and have the
    # caller feed int64 ids directly. The Gather then receives genuine int64
    # indices (the only dtype the kernel supports). ``input_ids`` is consumed
    # only by the token Gather (and a now-dead Shape from the replaced offset
    # chain), so widening it is safe. No runtime rebuild; GATE-1 (Llama) is
    # untouched. (A proper runtime fix would thread the indices dtype through
    # hip_gather / hip_cast; left as a follow-up — the int64-input boundary fully
    # resolves Whisper.)
    for gi in m.graph.input:
        if gi.name == "input_ids":
            gi.type.tensor_type.elem_type = TensorProto.INT64
            print("  Re-typed input_ids graph input int32 -> int64 (GPU Gather fix)")
            break

    # Drop nodes orphaned by the surgery above (the position-embedding Slice's
    # index-producer chain, plus any Shape(input_ids) left dangling once the
    # offset arithmetic was removed). Purely cosmetic — these are already
    # unreachable from a graph output, so behavior is unchanged; this just keeps
    # the saved graph readable in a viewer.
    n_pruned = _prune_dead_nodes(m.graph)
    if n_pruned:
        print(f"  Pruned {n_pruned} dead node(s) orphaned by surgery")

    onnx.save(m, str(dst), save_as_external_data=False)
    print(
        f"  Injected past_sequence_length into {self_count} self-attn MHAs -> {dst.name}"
    )


# ── Perf helpers ─────────────────────────────────────────────────────────────


def run_timed(session, inputs, warmup=NUM_WARMUP, runs=NUM_RUNS):
    for _ in range(warmup):
        session.run(None, inputs)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        session.run(None, inputs)
        times.append(time.perf_counter() - t0)
    return times


def report(label, times_sec):
    ms = [t * 1000 for t in times_sec]
    avg, std = np.mean(ms), np.std(ms)
    print(f"\n{'=' * 60}")
    print(f"{label}  ({len(ms)} runs)")
    print(f"  Avg: {avg:8.2f} ms  (std: {std:.2f})")
    print(f"  Min: {min(ms):8.2f} ms")
    print(f"  Max: {max(ms):8.2f} ms")
    print(f"  Tokens/sec: {1000 / avg:.1f}")
    print(f"{'=' * 60}")
    return avg


#: Cosine threshold for EP-vs-CPU accuracy checks. Set to 0.90 (not the
#: stricter 0.95 we'd ideally want) because matmul_nbits GEMV autotune picks
#: different `(BLOCK_SIZE, TILE_N)` configs across pytest invocations based on
#: GPU timing noise — same shape, same input, different reduction order. The
#: kernels themselves are deterministic: 3 sequential calls within one EP
#: session produce bit-identical output (verified for gpt-oss-20b fixed
#: prefill sq=128 kv=256). The cross-session autotune choice typically gives
#: EP-vs-CPU cosine in [0.96, 0.99] for the comparisons that hover near 0.95,
#: but rare unlucky picks dip to ~0.94 — intermittently failing a strict
#: 0.95-threshold test. 0.90 absorbs the noise without being so loose that
#: it would miss a real accuracy regression (which would drop cosine far
#: below 0.9, often into the 0.5–0.8 range).
COSINE_THRESHOLD = 0.90


def compare_outputs(ref_outputs, test_outputs, output_names, label, valid_seq=None):
    """Compare test EP outputs against CPU reference. Returns (passed, report_str).

    `valid_seq` (optional): when set, slice `present.*.{key,value}` outputs
    along their sequence axis to `[0, valid_seq)` before comparing. The KV
    cache buffer is pre-allocated to max_seq_len but only the first
    `past_len + sq` slots are semantically defined (ONNX `GroupQueryAttention`
    spec); ORT CPU returns a zero-initialized OrtValue so its unused tail is
    zero, while MorphiZenEP draws present buffers from a non-zeroed pool and
    leaks whatever the prior pool consumer wrote. Slicing limits the
    comparison to the defined region.
    """
    lines = []
    all_ok = True
    for i, name in enumerate(output_names):
        ref_arr = ref_outputs[i]
        test_arr = test_outputs[i]

        # Slice present_key/present_value to the semantically-defined region.
        # Layout is (batch, num_kv_heads, max_seq_len, head_dim) -- axis 2.
        if (
            valid_seq is not None
            and (name.startswith("present.") or name.startswith("past_key_values."))
            and ref_arr.ndim == 4
            and ref_arr.shape[2] > valid_seq
        ):
            ref_arr = ref_arr[:, :, :valid_seq, :]
            test_arr = test_arr[:, :, :valid_seq, :]

        ref = ref_arr.astype(np.float32).flatten()
        test = test_arr.astype(np.float32).flatten()

        if ref.shape != test.shape:
            lines.append(f"  {name}: SHAPE MISMATCH {ref.shape} vs {test.shape}")
            all_ok = False
            continue

        finite = np.isfinite(ref) & np.isfinite(test)
        ref_f, test_f = ref[finite], test[finite]
        n_skipped = int(ref.size - np.count_nonzero(finite))

        if ref_f.size == 0:
            lines.append(f"  {name}: no finite elements to compare  [SKIP]")
            continue

        max_abs = float(np.max(np.abs(ref_f - test_f)))

        ref_norm = np.linalg.norm(ref_f)
        diff_norm = np.linalg.norm(ref_f - test_f)
        rel_l2 = float(diff_norm / ref_norm) if ref_norm > 0 else 0.0

        dot = np.dot(ref_f, test_f)
        norms = np.linalg.norm(ref_f) * np.linalg.norm(test_f)
        cosine = float(dot / norms) if norms > 0 else 1.0

        ok = cosine > COSINE_THRESHOLD
        if not ok:
            all_ok = False
        status = "OK" if ok else "FAIL"

        skip_note = f"  skipped={n_skipped}" if n_skipped else ""
        lines.append(
            f"  {name}: cosine={cosine:.6f}  rel_L2={rel_l2:.4e}"
            f"  max_abs={max_abs:.4e}{skip_note}  [{status}]"
        )

    header = f"\n{'=' * 60}\n{label} accuracy vs CPU\n"
    body = "\n".join(lines)
    footer = f"\n{'=' * 60}"
    report_str = header + body + footer
    print(report_str)
    return all_ok, report_str


def get_amd_dml_providers():
    """Return DML provider list targeting the AMD GPU, falling back to default."""
    if "DmlExecutionProvider" not in ort.get_available_providers():
        return None
    try:
        import subprocess

        r = subprocess.run(
            [
                "powershell",
                "-Command",
                "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        for i, line in enumerate(r.stdout.strip().splitlines()):
            if "AMD" in line.upper() or "RADEON" in line.upper():
                print(f"  DirectML: using device {i} ({line.strip()})")
                return [
                    ("DmlExecutionProvider", {"device_id": str(i)}),
                    "CPUExecutionProvider",
                ]
    except Exception:
        pass
    return ["DmlExecutionProvider", "CPUExecutionProvider"]


def register_morphizen_ep(repo_root):
    """Register the AMDGPU umbrella EP library with ONNX Runtime (once per process).

    Registers ``amdgpu-ep.dll`` under ``AMDGPUExecutionProvider``; the umbrella
    loads hip-backend.dll → hipgpu.dll underneath. Returns the matching EP
    devices (callers pass ``EP_PROVIDER_OPTIONS`` to ``add_provider_for_devices``).
    """
    global _morphizen_registered

    dist_bin, therock_bin = _ep_runtime_dirs(repo_root)
    ep_dll = dist_bin / EP_DLL_NAME

    if not ep_dll.exists():
        return None

    for d in [dist_bin, therock_bin]:
        if d.exists() and str(d) not in os.environ.get("PATH", ""):
            os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")

    if not _morphizen_registered:
        ort.register_execution_provider_library(EP_REGISTRATION_NAME, str(ep_dll))
        _morphizen_registered = True

    from onnxruntime.capi._pybind_state import get_ep_devices

    devices = get_ep_devices()
    return [d for d in devices if d.ep_name == EP_REGISTRATION_NAME]


# ── IOBinding helpers ────────────────────────────────────────────────────────


@dataclass
class LlamaModelConfig:
    num_kv_layers: int
    num_kv_heads: int
    head_dim: int
    max_seq_len: int
    has_position_ids: bool = False


def make_llama_inputs(cfg: LlamaModelConfig, seq_len=1, seed=0) -> dict:
    """Create inputs for a Llama model.

    seq_len=1 (default) creates single-token decode inputs.
    seq_len>1 creates prefill inputs with that many tokens.
    Seeded RNG (default seed=0) so inputs are deterministic across runs —
    required for the GoldenStore CPU-reference cache to be valid.
    """
    rng = np.random.default_rng(seed)
    inputs = {
        "input_ids": rng.integers(1, 32000, size=(1, seq_len), dtype=np.int64),
        "attention_mask": np.ones((1, cfg.max_seq_len), dtype=np.int64),
    }
    if cfg.has_position_ids:
        inputs["position_ids"] = np.arange(
            cfg.max_seq_len - seq_len, cfg.max_seq_len, dtype=np.int64
        ).reshape(1, seq_len)
    for i in range(cfg.num_kv_layers):
        inputs[f"past_key_values.{i}.key"] = np.zeros(
            (1, cfg.num_kv_heads, cfg.max_seq_len, cfg.head_dim), dtype=np.float16
        )
        inputs[f"past_key_values.{i}.value"] = np.zeros(
            (1, cfg.num_kv_heads, cfg.max_seq_len, cfg.head_dim), dtype=np.float16
        )
    return inputs


# ── Whisper-large-v3 setup ───────────────────────────────────────────────────
#
# The raw fp32 + fp16 Whisper ONNX bundles are PRODUCED by
# ``scripts/build_whisper_models.py`` (the pinned OGA DirectML model builder,
# from ``openai/whisper-large-v3`` at a pinned commit revision) — they are NOT
# downloaded. The setup functions below are consume-only: they run the shared
# decoder surgery + fix_shapes on an already-built raw bundle and raise
# FileNotFoundError with a build hint if it is absent.


@dataclass
class WhisperModelConfig:
    """Shape parameters for Whisper-large-v3.

    Defaults match the official ``openai/whisper-large-v3`` config; override
    when running smaller variants. ``n_audio_ctx=1500`` is fixed by the
    log-mel preprocessor (30 s @ 100 Hz / 2 conv stride).
    """

    n_audio_state: int = 1280
    n_audio_layer: int = 32
    n_audio_head: int = 20
    n_audio_ctx: int = 1500
    n_text_state: int = 1280
    n_text_layer: int = 32
    n_text_head: int = 20
    n_text_ctx: int = 448
    n_mels: int = 128
    n_vocab: int = 51866


# Canonical AMD-hosted raw OGA bundles (encoder/decoder + tokenizer + config).
# These are the default model source: the setup helpers below download them on
# first use; a local build (scripts/build_whisper_models.py) is the backup that
# pre-populates the dir so the download is skipped. The fp32 LOCAL dir is
# `models/whisper-large-v3-onnx` (no -fp32 suffix) but its HF repo carries the
# -fp32 suffix — the download targets whatever model_dir the caller passes.
WHISPER_HF_REPO_FP32 = "amd/whisper-large-v3-onnx-fp32"
WHISPER_HF_REPO_FP16 = "amd/whisper-large-v3-onnx-fp16"


def _ensure_whisper_raw_downloaded(model_dir: pathlib.Path, hf_repo: str) -> None:
    """Fetch the raw OGA bundle from ``hf_repo`` into ``model_dir`` if absent.

    No-op when encoder.onnx + decoder.onnx are already present (a prior download,
    or a local ``scripts/build_whisper_models.py`` build — the backup path). On
    any failure (huggingface_hub missing, repo gated/unreachable) raises
    FileNotFoundError pointing at BOTH the HF-auth and the local-build remedies,
    so callers (pytest fixture / CLI) get one actionable message.
    """
    if (model_dir / "encoder.onnx").exists() and (model_dir / "decoder.onnx").exists():
        return

    hint = (
        f"Could not obtain the Whisper raw model for {model_dir}.\n"
        f"  - Primary: it auto-downloads from https://huggingface.co/{hf_repo} "
        "(run `hf auth login` if the repo needs auth / you are rate-limited).\n"
        "  - Backup:  build it locally with `python build.py --build-whisper-models`."
    )
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise FileNotFoundError(
            f"{hint}\n  (huggingface_hub not installed: {e})"
        ) from e

    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"[whisper] downloading raw model from {hf_repo} -> {model_dir}")
    # GatedRepo / RepositoryNotFound / network errors all map to the same hint.
    try:
        snapshot_download(repo_id=hf_repo, local_dir=str(model_dir))
    except Exception as e:  # noqa: BLE001
        raise FileNotFoundError(f"{hint}\n  (download failed: {e})") from e


def setup_whisper_model_dir(model_dir: pathlib.Path) -> None:
    """Idempotent fp32 Whisper preparation — downloads the raw model if needed.

    The raw fp32 OGA bundle is fetched from ``amd/whisper-large-v3-onnx-fp32`` on
    first use (the backup is a local ``scripts/build_whisper_models.py`` build via
    the pinned OGA DirectML builder, which pre-populates ``model_dir``). This
    function then runs the shared surgery + fix_shapes on the raw bundle.

    Raises FileNotFoundError (with HF + local-build hints) if the raw model can
    be neither downloaded nor found, so callers can skip/instruct cleanly.
    """
    _ensure_whisper_raw_downloaded(model_dir, WHISPER_HF_REPO_FP32)
    # Guard checks BOTH encoder + decoder: _apply_whisper_surgery_and_fix_shapes
    # reads encoder.onnx (fix_shapes) too, so a half-built dir with one but not
    # the other must fail here with the hint, not crash later raw.
    for fname in ("encoder.onnx", "decoder.onnx"):
        if not (model_dir / fname).exists():
            raise FileNotFoundError(
                f"Whisper raw model incomplete at {model_dir} (missing {fname}). "
                f"Download from https://huggingface.co/{WHISPER_HF_REPO_FP32} or "
                "build it: python build.py --build-whisper-models"
            )
    _apply_whisper_surgery_and_fix_shapes(model_dir)


def _apply_whisper_surgery_and_fix_shapes(model_dir: pathlib.Path) -> None:
    """Surgery + fix_shapes on a locally-built Whisper bundle (fp32 OR fp16).

    Consumes ``model_dir/{encoder,decoder}.onnx`` (+ ``.data`` sidecars) and emits
    ``decoder_surgery.onnx`` + the three canonical fixed-shape variants. This is
    the SAME pipeline for both precisions: the fp16 OGA build has the identical
    graph layout (node names, 8-input self-attn MHA, ``embed_positions`` Slice,
    int32 ``input_ids``) as the locally-built fp32 model, so ``inject_seqlens_k`` and
    ``fix_shapes`` apply unchanged. Idempotent (each step gates on its output).
    """
    # Stage 2: KV-cache surgery on the decoder (past_sequence_length input +
    # position_ids Gather + int64 input_ids — see inject_seqlens_k).
    decoder_surgery = model_dir / "decoder_surgery.onnx"
    inject_seqlens_k(model_dir / "decoder.onnx", decoder_surgery)

    # Stage 3: fix dynamic shapes — three variants with canonical names.
    encoder_fixed = model_dir / "encoder_fixed.onnx"
    if not encoder_fixed.exists():
        fix_shapes(model_dir / "encoder.onnx", encoder_fixed, {"batch_size": 1})

    # Pin past_seq_len == total_seq_len == n_text_ctx so the runtime can alias
    # past and present KV buffers (shared-buffer mode).
    prefill_fixed = model_dir / "decoder_fixed_prefill.onnx"
    if not prefill_fixed.exists():
        fix_shapes(
            decoder_surgery,
            prefill_fixed,
            {
                "batch_size": 1,
                "sequence_length": 4,
                "past_sequence_length": 448,
                "total_sequence_length": 448,
            },
        )

    decode_fixed = model_dir / "decoder_fixed_decode.onnx"
    if not decode_fixed.exists():
        fix_shapes(
            decoder_surgery,
            decode_fixed,
            {
                "batch_size": 1,
                "sequence_length": 1,
                "past_sequence_length": 448,
                "total_sequence_length": 448,
            },
        )


def setup_whisper_fp16_model_dir(model_dir: pathlib.Path) -> None:
    """Idempotent fp16 Whisper preparation — downloads the raw model if needed.

    Identical to ``setup_whisper_model_dir`` but for the fp16 bundle (HF repo
    ``amd/whisper-large-v3-onnx-fp16``; backup is a local
    ``scripts/build_whisper_models.py`` build). The fp16 OGA graph has the same
    layout as fp32, so ``_apply_whisper_surgery_and_fix_shapes`` applies
    unchanged. The fp16 lm_head stays fp32 (OGA convention) -> argmax-lossless.
    """
    _ensure_whisper_raw_downloaded(model_dir, WHISPER_HF_REPO_FP16)
    # Guard checks BOTH encoder + decoder (see setup_whisper_model_dir).
    for fname in ("encoder.onnx", "decoder.onnx"):
        if not (model_dir / fname).exists():
            raise FileNotFoundError(
                f"Whisper fp16 raw model incomplete at {model_dir} (missing {fname}). "
                f"Download from https://huggingface.co/{WHISPER_HF_REPO_FP16} or "
                "build it: python build.py --build-whisper-models"
            )
    _apply_whisper_surgery_and_fix_shapes(model_dir)


def make_whisper_inputs(audio_path: pathlib.Path, cfg: WhisperModelConfig) -> dict:
    """Build encoder input from a 16 kHz mono WAV using the HF feature extractor.

    Returns ``{"audio_features": np.ndarray[1, n_mels, 3000] fp32}``.

    NOTE: Whisper is fp32-only here (``encoder_fixed.onnx`` and the CPU
    ``encoder.onnx`` reference are both fp32), so ``audio_features`` is fed as
    fp32 to every backend with no caller cast. ``WhisperFeatureExtractor``
    always returns fp32, which is exactly what the fp32 graphs expect.

    Audio loading: ``soundfile`` for the file, ``librosa.resample`` if the WAV
    isn't already 16 kHz. ``librosa`` is an optional dep (no hard requirement
    in environment.yml today) — the bundled ``jfk.wav`` is 16 kHz so the
    smoke path does NOT need librosa. If you supply a non-16k WAV and
    librosa is missing, this raises ``ImportError`` with a clear message.
    """
    import soundfile as sf
    from transformers import WhisperFeatureExtractor

    # The processor config (mel filterbanks, target sample rate, n_mels)
    # is the same across whisper-large-v3 variants — use the openai original
    # so we don't depend on the built ONNX bundle shipping a processor.
    fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3")

    audio, sr = sf.read(str(audio_path))
    # soundfile returns shape (N,) for mono, (N, C) for multi-channel.
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != 16000:
        try:
            import librosa
        except ImportError as e:
            raise ImportError(
                f"Audio at {audio_path} is {sr} Hz but librosa is not installed; "
                "install librosa to enable resampling, or supply a 16 kHz WAV."
            ) from e
        audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=16000)

    out = fe(audio, sampling_rate=16000, return_tensors="np")
    return {"audio_features": out["input_features"].astype(np.float32)}


# ── LibriSpeech test-clip provisioning (no `datasets` / `jiwer` deps) ─────────
#
# The HF datasets-server exposes the LibriSpeech "dummy" split (5-15 s read-speech
# clips from public-domain LibriVox audiobooks, CC BY 4.0) plus the ground-truth
# reference text and signed FLAC URLs over plain HTTP JSON. We fetch a handful of
# clips ONCE, transcode FLAC -> 16 kHz mono WAV via soundfile (no librosa needed —
# LibriSpeech is already 16 kHz), and write the WAVs + a references.json locally
# (gitignored, NOT committed) so the tests are deterministic and offline after the
# first run. The signed URLs expire, so the fetch is best-effort: if it fails AND
# no cached WAVs exist, the caller pytest.skips.

LIBRISPEECH_DATASET = "hf-internal-testing/librispeech_asr_dummy"
LIBRISPEECH_N_SAMPLES = 5  # sample_0.wav .. sample_4.wav
# The ~30 s "long" clip is built by concatenating the first few dummy clips with
# short silences between them; the reference is their texts joined in order. This
# gives a known ground truth for free and stays just under Whisper's 30 s window
# (so it exercises deep decode without triggering whisper's chunking).
LIBRISPEECH_LONG_TARGET_S = 29.0
LIBRISPEECH_LONG_SILENCE_S = 0.3


def _fetch_librispeech_rows(n: int) -> list:
    """Fetch the first ``n`` dummy clips' (flac_bytes, reference_text) over HTTP.

    Returns a list of ``(audio_bytes, text)`` tuples, or raises on any network /
    JSON / signature error so the caller can decide to skip vs use cache.
    """
    import json

    url = (
        "https://datasets-server.huggingface.co/rows"
        f"?dataset={LIBRISPEECH_DATASET}&config=clean&split=validation"
        f"&offset=0&length={max(n, 10)}"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    data = json.load(urllib.request.urlopen(req, timeout=120))
    rows = data["rows"]
    out = []
    for r in rows[:n]:
        row = r["row"]
        src = row["audio"][0]["src"]  # signed flac URL
        areq = urllib.request.Request(src, headers={"User-Agent": "Mozilla/5.0"})
        audio_bytes = urllib.request.urlopen(areq, timeout=120).read()
        out.append((audio_bytes, row["text"]))
    return out


def setup_librispeech_samples(data_dir: pathlib.Path) -> bool:
    """Idempotently provision the LibriSpeech test clips under ``data_dir``.

    Produces (all gated on existence — zero cost once cached):

      * ``sample_0.wav`` .. ``sample_{N-1}.wav``  — 16 kHz mono WAV transcodes of
        the first N dummy clips.
      * ``long_30s.wav``  — several clips concatenated (~29 s) with 0.3 s silence
        between, staying under Whisper's 30 s window.
      * ``references.json`` — ``{filename: UPPERCASE_reference_text}`` for every
        WAV above (the long clip's reference is the concatenation, in order, of
        the clips that compose it).

    Returns ``True`` if the clips are available (cached or freshly fetched),
    ``False`` if the network fetch failed and nothing is cached (caller should
    ``pytest.skip``). Never raises on a fetch failure.
    """
    import json

    import soundfile as sf

    data_dir.mkdir(parents=True, exist_ok=True)
    refs_path = data_dir / "references.json"
    sample_paths = [data_dir / f"sample_{i}.wav" for i in range(LIBRISPEECH_N_SAMPLES)]
    long_path = data_dir / "long_30s.wav"

    # Fast path: everything already on disk.
    if (
        refs_path.exists()
        and all(p.exists() for p in sample_paths)
        and long_path.exists()
    ):
        return True

    try:
        rows = _fetch_librispeech_rows(LIBRISPEECH_N_SAMPLES)
    except Exception as e:  # network / expired signature / JSON shape change
        print(f"  [librispeech] fetch failed ({e!r})")
        # If a previous run already cached the clips, use them anyway.
        if (
            refs_path.exists()
            and all(p.exists() for p in sample_paths)
            and long_path.exists()
        ):
            return True
        return False

    import io

    refs: dict = {}
    decoded = []  # (audio_16k_mono_fp32, text) for the long-clip build
    for i, (audio_bytes, text) in enumerate(rows):
        audio, sr = sf.read(io.BytesIO(audio_bytes))
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)
        # LibriSpeech is natively 16 kHz; assert rather than silently resample so
        # a future dataset change can't sneak a wrong sample rate past us.
        assert sr == 16000, f"expected 16 kHz LibriSpeech, got {sr} Hz for clip {i}"
        sf.write(str(sample_paths[i]), audio, 16000, subtype="PCM_16")
        refs[sample_paths[i].name] = text
        decoded.append((audio, text))

    # Build the ~30 s long clip by concatenating clips (with 0.3 s silence between)
    # until we approach LIBRISPEECH_LONG_TARGET_S. The reference is the joined text.
    silence = np.zeros(int(0.0 + LIBRISPEECH_LONG_SILENCE_S * 16000), dtype=np.float32)
    chunks: list = []
    long_texts: list = []
    total_s = 0.0
    # Greedy pack: take clips (in order) that still fit under the budget, skipping
    # ones too long rather than stopping — gets us closest to ~29 s.
    for audio, text in decoded:
        clip_s = len(audio) / 16000.0
        # +silence only between clips, not after the last; check the running budget.
        added = clip_s + (LIBRISPEECH_LONG_SILENCE_S if chunks else 0.0)
        if chunks and total_s + added > LIBRISPEECH_LONG_TARGET_S:
            continue  # this clip overflows; try the next (shorter) one
        if chunks:
            chunks.append(silence)
            total_s += LIBRISPEECH_LONG_SILENCE_S
        chunks.append(audio)
        total_s += clip_s
        long_texts.append(text)
    long_audio = np.concatenate(chunks).astype(np.float32)
    sf.write(str(long_path), long_audio, 16000, subtype="PCM_16")
    refs[long_path.name] = " ".join(long_texts)
    print(
        f"  [librispeech] long_30s.wav = {len(long_audio) / 16000.0:.1f} s "
        f"from {len(long_texts)} clips"
    )

    refs_path.write_text(json.dumps(refs, indent=2) + "\n")
    return True


AMD_VENDOR_ID = 0x1002


def run_timed_iobinding(
    sess,
    inputs,
    cfg: LlamaModelConfig,
    warmup=NUM_WARMUP,
    runs=NUM_RUNS,
    use_device_memory=False,
    dim_map=None,
):
    """Run inference using IOBinding with shared past/present KV cache buffers.

    Binds the same OrtValue to both past_key_values.N.{key,value} (input) and
    present.N.{key,value} (output) so the runtime sees past_ptr == present_ptr.

    When use_device_memory=True, KV cache OrtValues are allocated via the EP's
    GPU allocator (hipHostMalloc on AMD iGPU) so the runtime sees
    memory_type==TENSOR_MEMORY_GPU and aliases them directly — zero H2D/D2H.
    """
    io = sess.io_binding()

    kv_shape = (1, cfg.num_kv_heads, cfg.max_seq_len, cfg.head_dim)
    kv_cache = {}
    for i in range(cfg.num_kv_layers):
        for kind in ("key", "value"):
            if use_device_memory:
                kv_cache[(i, kind)] = ort.OrtValue.ortvalue_from_shape_and_type(
                    list(kv_shape),
                    np.float16,
                    device_type="gpu",
                    vendor_id=AMD_VENDOR_ID,
                )
            else:
                kv_cache[(i, kind)] = ort.OrtValue.ortvalue_from_numpy(
                    np.zeros(kv_shape, dtype=np.float16)
                )

    _ORT_TYPE_MAP = {
        "tensor(float16)": np.float16,
        "tensor(float)": np.float32,
        "tensor(int64)": np.int64,
        "tensor(int32)": np.int32,
    }

    output_vals = {}
    for o in sess.get_outputs():
        if o.name.startswith("present."):
            continue
        dtype = _ORT_TYPE_MAP.get(o.type, np.float32)
        shape = o.shape
        if dim_map:
            shape = [dim_map.get(d, d) if isinstance(d, str) else d for d in shape]
        output_vals[o.name] = ort.OrtValue.ortvalue_from_shape_and_type(shape, dtype)

    def bind_all():
        io.clear_binding_inputs()
        io.clear_binding_outputs()
        for name, arr in inputs.items():
            if name.startswith("past_key_values"):
                continue
            io.bind_ortvalue_input(name, ort.OrtValue.ortvalue_from_numpy(arr))
        for i in range(cfg.num_kv_layers):
            for kind in ("key", "value"):
                val = kv_cache[(i, kind)]
                io.bind_ortvalue_input(f"past_key_values.{i}.{kind}", val)
                io.bind_ortvalue_output(f"present.{i}.{kind}", val)
        for name, val in output_vals.items():
            io.bind_ortvalue_output(name, val)

    for _ in range(warmup):
        bind_all()
        sess.run_with_iobinding(io)

    times = []
    for _ in range(runs):
        bind_all()
        t0 = time.perf_counter()
        sess.run_with_iobinding(io)
        times.append(time.perf_counter() - t0)
    return times


def run_iobinding_once(
    sess,
    inputs,
    cfg: LlamaModelConfig,
    use_device_memory=False,
    dim_map=None,
):
    """Run single inference with IOBinding, return outputs as list.

    Returns outputs in sess.get_outputs() order, compatible with
    compare_outputs(). Same IOBinding setup as run_timed_iobinding (shared
    past/present KV cache buffers).

    use_device_memory=False (default) zero-initializes KV cache from numpy,
    matching CPU sess.run() for fair accuracy comparison.
    """
    io = sess.io_binding()

    kv_shape = (1, cfg.num_kv_heads, cfg.max_seq_len, cfg.head_dim)
    kv_cache = {}
    for i in range(cfg.num_kv_layers):
        for kind in ("key", "value"):
            if use_device_memory:
                kv_cache[(i, kind)] = ort.OrtValue.ortvalue_from_shape_and_type(
                    list(kv_shape),
                    np.float16,
                    device_type="gpu",
                    vendor_id=AMD_VENDOR_ID,
                )
            else:
                past_name = f"past_key_values.{i}.{kind}"
                arr = inputs.get(past_name, np.zeros(kv_shape, dtype=np.float16))
                kv_cache[(i, kind)] = ort.OrtValue.ortvalue_from_numpy(arr)

    _ORT_TYPE_MAP = {
        "tensor(float16)": np.float16,
        "tensor(float)": np.float32,
        "tensor(int64)": np.int64,
        "tensor(int32)": np.int32,
    }

    output_vals = {}
    for o in sess.get_outputs():
        if o.name.startswith("present."):
            continue
        dtype = _ORT_TYPE_MAP.get(o.type, np.float32)
        shape = o.shape
        if dim_map:
            shape = [dim_map.get(d, d) if isinstance(d, str) else d for d in shape]
        output_vals[o.name] = ort.OrtValue.ortvalue_from_shape_and_type(shape, dtype)

    for name, arr in inputs.items():
        if name.startswith("past_key_values"):
            continue
        io.bind_ortvalue_input(name, ort.OrtValue.ortvalue_from_numpy(arr))
    for i in range(cfg.num_kv_layers):
        for kind in ("key", "value"):
            val = kv_cache[(i, kind)]
            io.bind_ortvalue_input(f"past_key_values.{i}.{kind}", val)
            io.bind_ortvalue_output(f"present.{i}.{kind}", val)
    for name, val in output_vals.items():
        io.bind_ortvalue_output(name, val)

    sess.run_with_iobinding(io)

    results = []
    for o in sess.get_outputs():
        if o.name.startswith("present."):
            parts = o.name.split(".")
            idx, kind = int(parts[1]), parts[2]
            results.append(kv_cache[(idx, kind)].numpy())
        else:
            results.append(output_vals[o.name].numpy())
    return results


# ── Session helpers ──────────────────────────────────────────────────────────


def create_cpu_session(model_path):
    return ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])


def create_ep_session(model_path, repo_root, provider_options=None):
    """Create an AMDGPU-umbrella-EP InferenceSession.

    provider_options: optional dict forwarded to the EP. Defaults to
    EP_PROVIDER_OPTIONS ({"profile": "llm"}) — the umbrella uses "profile" to
    select the hipgpu backend. The backend compiles every model in
    output-allocator mode (the 2-arg in-graph hip.alloc_output ABI) -- there is
    no provider option to select a mode.
    """
    devices = register_morphizen_ep(repo_root)
    if not devices:
        pytest.skip("AMDGPU EP not found — run build.py first")
    so = ort.SessionOptions()
    apply_artifact_format(
        so
    )  # bitcode by default; HIPEP_ARTIFACT_FORMAT=NATIVE opts in
    so.add_provider_for_devices(devices, provider_options or dict(EP_PROVIDER_OPTIONS))
    return ort.InferenceSession(model_path, sess_options=so)


def cleanup(*args):
    for obj in args:
        del obj
    gc.collect()


# ── Dim map ─────────────────────────────────────────────────────────────────


def make_dim_map(seq_len, kv_len):
    return {
        "batch_size": 1,
        "sequence_length": seq_len,
        "past_sequence_length": kv_len,
        "total_sequence_length": kv_len,
    }


# ── Model download ─────────────────────────────────────────────────────────


def ensure_model(model_dir, onnx_file, data_file, hf_base):
    """Download ONNX + external data files into `model_dir` if missing.

    `data_file` may be a single filename (str) for models with one external
    data blob (Llama 1B/8B), or an iterable of filenames for models split
    across multiple blobs (e.g. gpt-oss-20b — 7 `*.onnx_data*` files).
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = model_dir / onnx_file
    if not onnx_path.exists():
        download(f"{hf_base}/{onnx_file}", onnx_path)
    data_files = [data_file] if isinstance(data_file, str) else list(data_file)
    for d in data_files:
        data_path = model_dir / d
        if not data_path.exists():
            download(f"{hf_base}/{d}", data_path)
    return str(onnx_path)


def ensure_fixed_model(model_dir, onnx_file, data_file, seq_len, kv_len):
    """Materialize a fixed-shape ONNX next to (not inside) the dynamic model dir.

    External weights are hardlinked (or copied as fallback) into the sibling
    `<model_dir>-fixed/` so the rewritten ONNX's external_data refs resolve.
    Keeping the dynamic model dir free of test-derived `model_fixed_kv*_sq*.onnx`
    leftovers — same hygiene as `ensure_pipeline_dir`. `data_file` accepts the
    same str-or-iterable forms as `ensure_model`.
    """
    fixed_dir = model_dir.parent / f"{model_dir.name}-fixed"
    fixed_dir.mkdir(parents=True, exist_ok=True)
    data_files = [data_file] if isinstance(data_file, str) else list(data_file)
    for d in data_files:
        weights_dst = fixed_dir / d
        if not weights_dst.exists():
            try:
                os.link(model_dir / d, weights_dst)
            except OSError:
                shutil.copy2(model_dir / d, weights_dst)
    name = f"{onnx_file.rsplit('.', 1)[0]}_fixed_kv{kv_len}_sq{seq_len}.onnx"
    dst = fixed_dir / name
    if not dst.exists():
        dim_map = make_dim_map(seq_len, kv_len)
        fix_shapes(model_dir / onnx_file, dst, dim_map)
    return str(dst)


# ── Generation loop helpers ─────────────────────────────────────────────────


def make_prefill_inputs(cfg, prompt_tokens, max_seq_len):
    seq_len = len(prompt_tokens)
    inputs = {
        "input_ids": np.array([prompt_tokens], dtype=np.int64),
        "attention_mask": np.zeros((1, max_seq_len), dtype=np.int64),
    }
    inputs["attention_mask"][0, :seq_len] = 1
    if cfg.has_position_ids:
        inputs["position_ids"] = np.arange(seq_len, dtype=np.int64).reshape(1, -1)
    for i in range(cfg.num_kv_layers):
        inputs[f"past_key_values.{i}.key"] = np.zeros(
            (1, cfg.num_kv_heads, max_seq_len, cfg.head_dim), dtype=np.float16
        )
        inputs[f"past_key_values.{i}.value"] = np.zeros(
            (1, cfg.num_kv_heads, max_seq_len, cfg.head_dim), dtype=np.float16
        )
    return inputs


def make_decode_inputs(cfg, token_id, position, kv_cache, max_seq_len):
    inputs = {
        "input_ids": np.array([[token_id]], dtype=np.int64),
        "attention_mask": np.zeros((1, max_seq_len), dtype=np.int64),
    }
    inputs["attention_mask"][0, : position + 1] = 1
    if cfg.has_position_ids:
        inputs["position_ids"] = np.array([[position]], dtype=np.int64)
    for i in range(cfg.num_kv_layers):
        inputs[f"past_key_values.{i}.key"] = kv_cache[(i, "key")]
        inputs[f"past_key_values.{i}.value"] = kv_cache[(i, "value")]
    return inputs


def extract_kv_cache(outputs, output_names):
    kv = {}
    for idx, name in enumerate(output_names):
        if name.startswith("present."):
            parts = name.split(".")
            kv[(int(parts[1]), parts[2])] = outputs[idx]
    return kv


def make_zero_kv_cache(cfg, max_seq_len):
    """Empty past-KV dict in the layout `make_decode_inputs` expects."""
    shape = (1, cfg.num_kv_heads, max_seq_len, cfg.head_dim)
    kv = {}
    for i in range(cfg.num_kv_layers):
        kv[(i, "key")] = np.zeros(shape, dtype=np.float16)
        kv[(i, "value")] = np.zeros(shape, dtype=np.float16)
    return kv


def get_next_token(logits):
    return int(np.argmax(logits[0, -1, :].astype(np.float32)))


def compare_logits(ref_logits, test_logits, step_name):
    ref = ref_logits.astype(np.float32).flatten()
    test = test_logits.astype(np.float32).flatten()
    finite = np.isfinite(ref) & np.isfinite(test)
    ref_f, test_f = ref[finite], test[finite]
    if ref_f.size == 0:
        print(f"  {step_name}: no finite elements [SKIP]")
        return True
    dot = np.dot(ref_f, test_f)
    norms = np.linalg.norm(ref_f) * np.linalg.norm(test_f)
    cosine = float(dot / norms) if norms > 0 else 1.0
    ref_norm = np.linalg.norm(ref_f)
    diff_norm = np.linalg.norm(ref_f - test_f)
    rel_l2 = float(diff_norm / ref_norm) if ref_norm > 0 else 0.0
    top1_match = np.argmax(ref_f) == np.argmax(test_f)
    ok = cosine > COSINE_THRESHOLD
    status = "OK" if ok else "FAIL"
    print(
        f"  {step_name}: cosine={cosine:.6f}  rel_L2={rel_l2:.4e}"
        f"  top1_match={top1_match}  [{status}]"
    )
    return ok


# ── OGA helpers ─────────────────────────────────────────────────────────────


def make_prompt_tokens(bos_token, filler_tokens, length):
    return [bos_token] + list(
        itertools.islice(itertools.cycle(filler_tokens), length - 1)
    )


_oga_ep_registered = False


def setup_oga_ep(repo_root):
    try:
        import onnxruntime_genai as og
    except ImportError:
        pytest.skip("onnxruntime-genai not installed")

    if not hasattr(og, "register_execution_provider_library"):
        pytest.skip("OGA version does not support custom EP registration")

    dist_bin, therock_bin = _ep_runtime_dirs(repo_root)
    ep_dll = dist_bin / EP_DLL_NAME
    if not ep_dll.exists():
        pytest.skip("AMDGPU EP DLL not found — run build.py first")

    for d in [dist_bin, therock_bin]:
        if d.exists() and str(d) not in os.environ.get("PATH", ""):
            os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")

    global _oga_ep_registered
    if not _oga_ep_registered:
        og.register_execution_provider_library(EP_OGA_NAME, str(ep_dll))
        _oga_ep_registered = True

    return og, ep_dll


def patch_genai_config_for_morphizen(model_dir, ep_dll):
    config_path = model_dir / "genai_config.json"
    backup_path = model_dir / "genai_config.json.bak"
    if not config_path.exists():
        return
    shutil.copy2(config_path, backup_path)
    with open(config_path) as f:
        config = json.load(f)
    morphizen_options = [{EP_OGA_NAME: dict(EP_PROVIDER_OPTIONS)}]
    config["model"]["decoder"]["session_options"]["provider_options"] = (
        morphizen_options
    )
    # Multimodal configs (e.g. gemma3 VLM type) have separate embedding /
    # vision / speech ONNX sessions in addition to the text decoder. OGA pre-
    # allocates the cross-session tensors (`inputs_embeds`, `image_features`)
    # on the decoder's primary device (MorphiZenEP GPU) via
    # `model_.p_device_->GetAllocator()`. When a CPU-only embedding session
    # tries to bind those GPU tensors at IoBinding setup, ORT requires an
    # `IDataTransfer` between CPU and the MorphiZenEP device — but
    # `HipDataTransferImpl` is only registered on sessions that include
    # MorphiZenEP in their provider_options. Register MorphiZenEP on all sub-
    # sessions so the transfer is available; MorphiZenEP does whole-graph
    # fusion only and will not claim the embedding / vision / speech graphs
    # (NonZero / ScatterND / Expand / Equal etc. have no HIP converters), so
    # ORT falls back to CPU EP for the actual compute. The graph-claim
    # attempt fails by design — `HIPDNN_EP_STRICT` is opt-in so the default
    # soft-fail path lets the fallback happen cleanly. **Do NOT set
    # `HIPDNN_EP_STRICT=1` when running these tests** — the env var aborts on
    # graph-claim-then-fail, and the failure on the sub-models is intentional.
    for sub in ("embedding", "vision", "speech"):
        sub_cfg = config["model"].get(sub)
        if isinstance(sub_cfg, dict):
            sub_session_opts = sub_cfg.setdefault("session_options", {})
            sub_session_opts["provider_options"] = morphizen_options
    # Pareto-optimal chunk_size for dynamic-shape Llama on MorphiZenEP.
    # 8B sweep on gfx1151 (2026-05-06): chunk=1024 beats chunk=512 on long
    # prompts (-1.4% TTFT at L=2048, -3.6% at L=4096) at +0.16 GB peak WS at
    # the worst L. At L<=1024 it's a no-op (single chunk). Decode is invariant.
    # Skip for fixed-shape configs — decoder-pipeline / sliding_window /
    # fixed_prompt_length already chunk via their own mechanisms, and injecting
    # search.chunk_size on top would either be ignored or fight them.
    decoder = config["model"].get("decoder", {})
    is_fixed_shape = (
        config["model"].get("type") == "decoder-pipeline"
        or "sliding_window" in decoder
        or "fixed_prompt_length" in decoder
    )
    if not is_fixed_shape:
        config.setdefault("search", {})["chunk_size"] = 1024
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)


def restore_genai_config(model_dir):
    config_path = model_dir / "genai_config.json"
    backup_path = model_dir / "genai_config.json.bak"
    if backup_path.exists():
        shutil.move(backup_path, config_path)


def oga_generate(og, model, tokenizer, prompt_tokens, max_new=30):
    """Generate tokens with OGA, return (tokens, ttft_ms, tps)."""
    params = og.GeneratorParams(model)
    params.set_search_options(max_length=len(prompt_tokens) + max_new, do_sample=False)
    generator = og.Generator(model, params)

    t0 = time.perf_counter()
    generator.append_tokens(np.array(prompt_tokens, dtype=np.int32))
    generator.generate_next_token()
    ttft_ms = (time.perf_counter() - t0) * 1000
    generated = [int(generator.get_next_tokens()[0])]

    if generator.is_done() or len(generated) >= max_new:
        return generated, ttft_ms, 0.0

    t_decode = time.perf_counter()
    while not generator.is_done() and len(generated) < max_new:
        generator.generate_next_token()
        generated.append(int(generator.get_next_tokens()[0]))
    decode_ms = (time.perf_counter() - t_decode) * 1000

    n_decode = len(generated) - 1
    tps = n_decode / (decode_ms / 1000) if decode_ms > 0 and n_decode > 0 else 0.0
    return generated, ttft_ms, tps


def oga_generate_timed(og, model, tokenizer, prompt_tokens, max_new=128):
    """Benchmark OGA generation, methodology aligned with model_benchmark.exe.

    Uses min_length to force exact token count and a clean decode loop
    (no get_next_tokens / list building inside timed section).
    """
    total_len = len(prompt_tokens) + max_new
    prompt_np = np.array(prompt_tokens, dtype=np.int32)

    params = og.GeneratorParams(model)
    params.set_search_options(
        max_length=total_len,
        min_length=total_len,
        do_sample=False,
    )
    generator = og.Generator(model, params)

    # Warmup: full prefill + all decode tokens (same generator object)
    generator.append_tokens(prompt_np)
    while not generator.is_done():
        generator.generate_next_token()
    generator.rewind_to(0)

    # Timed prefill (TTFT)
    t0 = time.perf_counter()
    generator.append_tokens(prompt_np)
    generator.generate_next_token()
    ttft_ms = (time.perf_counter() - t0) * 1000

    # Timed decode — only generate_next_token() + is_done() in hot loop
    n_decode = 0
    t_decode = time.perf_counter()
    while not generator.is_done():
        generator.generate_next_token()
        n_decode += 1
    decode_ms = (time.perf_counter() - t_decode) * 1000
    tps = n_decode / (decode_ms / 1000) if decode_ms > 0 and n_decode > 0 else 0.0

    generated = n_decode + 1  # +1 for the prefill token

    del generator
    return generated, ttft_ms, tps


def run_cpu_reference_generation(
    model_path, cfg, prompt_tokens, max_new, max_seq_len=None
):
    """CPU prefill + (max_new-1) decode steps via sess.run(). Returns token list.

    Used as accuracy reference for OGA-based generation tests. The session is
    created and torn down inside this helper to keep CPU memory transient.
    """
    if max_seq_len is None:
        max_seq_len = len(prompt_tokens) + max_new
    cpu_sess = create_cpu_session(model_path)
    output_names = [o.name for o in cpu_sess.get_outputs()]
    logits_idx = output_names.index("logits")

    cpu_inputs = make_prefill_inputs(cfg, prompt_tokens, max_seq_len)
    cpu_out = cpu_sess.run(None, cpu_inputs)
    cpu_token = get_next_token(cpu_out[logits_idx])
    cpu_kv = extract_kv_cache(cpu_out, output_names)
    generated = [cpu_token]

    for step in range(max_new - 1):
        position = len(prompt_tokens) + step
        cpu_inputs = make_decode_inputs(cfg, cpu_token, position, cpu_kv, max_seq_len)
        cpu_out = cpu_sess.run(None, cpu_inputs)
        cpu_token = get_next_token(cpu_out[logits_idx])
        cpu_kv = extract_kv_cache(cpu_out, output_names)
        generated.append(cpu_token)

    cleanup(cpu_sess)
    return generated


# ── Shared fixtures ──────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def repo_root():
    return REPO_ROOT


# ─────────────────────────────────────────────────────────────────────────────
# ModelSpec + base test classes + golden cache
# ─────────────────────────────────────────────────────────────────────────────
#
# Generalized per-model test framework. Each test_<model>.py declares one
# ModelSpec literal and binds it to two thin subclasses of BaseORTTests /
# BaseOGATests. The base classes own the 9 canonical test methods (5 ORT +
# 4 OGA) that were previously duplicated byte-for-byte across every per-
# model file. Per-model variations (markers, CPU skip predicates, custom
# input builders) live on the spec, not in overridden test methods.
#
# A GoldenStore caches the CPU reference outputs under install/golden/
# <spec.name>/ so the slow CPU pass runs only once per (model, test) tuple.
# On cache hit the test loads the saved arrays and runs EP only — typically
# 30%+ wall-clock cut on 8B and bigger. Auto-invalidated when the ONNX file
# mtime/size or the inputs hash changes. Force full regen with `rm -rf
# install/golden`; force one model with `rm -rf install/golden/<name>`. No
# CLI flag, no env var.


GOLDEN_ROOT = REPO_ROOT / "install" / "golden"


@dataclass
class ModelSpec:
    """Per-model test configuration consumed by BaseORTTests / BaseOGATests.

    `hf_base` (urllib, public repos) and `hf_repo` (huggingface_hub, gated
    repos) are mutually exclusive — set exactly one. `oga_files` are the
    tokenizer / genai_config files OGA needs alongside the ONNX. For models
    whose HF repo ships no genai_config (e.g. Llama-1B), set
    `genai_config_template` to a dict that will be written to disk on first
    OGA test; OGA-only tokenizer downloads then use `hf_root_for_tokenizer`
    (one level above `hf_base` for the 1B "onnx" subdir case).
    """

    # Identity
    name: str  # cache key prefix; also used in install/golden/<name>/
    model_dir: pathlib.Path
    onnx_file: str
    data_files: list  # list[str] — multi-blob support (gpt-oss-20b has 7)

    # Architecture
    num_layers: int = 0
    num_kv_heads: int = 0
    head_dim: int = 0
    has_position_ids: bool = False
    max_seq_len: int = 256

    # Tokens
    bos_token: int = 1
    filler_tokens: list = field(default_factory=list)
    num_generate_tokens: int = 10
    prompt_tokens: Optional[list] = None  # default: bos + filler[:6]

    # Fetch (exactly one of these)
    hf_base: Optional[str] = None  # urllib
    hf_repo: Optional[str] = None  # huggingface_hub
    # When set, hf_hub downloads come from `<hf_subdir>/<filename>` in the repo
    # and we move the result up to model_dir root (flatten). Used by AMD-
    # converted Qwen / Gemma3 repos that ship under "Upload * ONNX files/".
    hf_subdir: Optional[str] = None

    # Auxiliary ONNX + data files alongside the main one (gemma3 ships
    # embedding.onnx, vision.onnx and their .data blobs in addition to
    # text.onnx). Downloaded by fetch_model_files; ignored by ensure_fixed_model.
    extra_data_files: list = field(default_factory=list)

    # OGA companions
    oga_files: list = field(default_factory=list)
    genai_config_template: Optional[dict] = None
    hf_root_for_tokenizer: Optional[str] = None
    chunk_size: Optional[int] = 1024

    # Per-model hooks (rare)
    normalize_onnx_hook: Optional[Callable] = None
    # Optional callable overrides for input construction. Default behaviour
    # is input_ids-based (Llama / gpt-oss / Mistral / Phi / Qwen / DeepSeek).
    # Gemma3 sets these to its inputs_embeds-based builders so no per-model
    # test class is needed.
    #   prefill_input_fn(cfg, prefill_len: int, max_seq_len: int) -> dict
    #   decode_input_fn(cfg, position: int, kv_cache: dict, max_seq_len) -> dict
    prefill_input_fn: Optional[Callable] = None
    decode_input_fn: Optional[Callable] = None
    cpu_skip_predicate: Optional[Callable] = None
    cpu_skip_reason: Optional[str] = None
    markers: dict = field(default_factory=dict)

    # Position at which decode tests place the new token (i.e. attention_mask
    # covers [0..decode_position], the new K writes to slot `decode_position`).
    # 7 keeps attention nontrivial without being expensive.
    decode_position: int = 7

    # Session-reuse opt-in. When True (default for small/medium models), the
    # per-module EP session and OGA default-config Model are loaded once and
    # shared across tests in the file — saves ~3-15 s/test of session init.
    # Set False for very large models (DeepSeek-R1-70B, gpt-oss-120b) where
    # holding multiple sessions simultaneously would OOM the 96 GB host.
    reuse_ep_session: bool = True
    reuse_oga_default_model: bool = True

    # HIPDNN_EP_STRICT toggle for OGA tests. Default True (matches the
    # conftest-wide default). Set False for multimodal-OGA scenarios where
    # sub-sessions (embedding/vision/speech) intentionally have graph-claim-
    # then-fail behaviour — currently just gemma3, whose OGA tests load the
    # gemma3 VLM pipeline. ORT-direct tests for the same models still run
    # with STRICT=1 because they only exercise the text decoder.
    oga_strict: bool = True

    # Opt-in for the output-allocator (2-arg ABI) end-to-end accuracy test
    # (BaseORTTests.test_ort_output_allocator_dynamic). Default False so the
    # heavier model files skip it; enabled on the fast Llama-1B spec, which is
    # representative because allocator mode is model-agnostic EP plumbing.
    output_allocator_e2e: bool = False

    def __post_init__(self):
        if (self.hf_base is None) == (self.hf_repo is None):
            raise ValueError(
                f"ModelSpec {self.name!r}: must set exactly one of hf_base / hf_repo"
            )
        if not self.filler_tokens:
            raise ValueError(f"ModelSpec {self.name!r}: filler_tokens required")
        if self.prompt_tokens is None:
            # 7-token default: BOS + 6 fillers (matches the current per-file convention)
            self.prompt_tokens = [self.bos_token, *self.filler_tokens[:6]]

    @property
    def onnx_path(self):
        return self.model_dir / self.onnx_file

    def make_cfg(self, max_seq_len=None):
        return LlamaModelConfig(
            num_kv_layers=self.num_layers,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            max_seq_len=max_seq_len if max_seq_len is not None else self.max_seq_len,
            has_position_ids=self.has_position_ids,
        )

    def make_prompt(self, length):
        return make_prompt_tokens(self.bos_token, self.filler_tokens, length)

    def build_prefill_inputs(self, cfg, prefill_len, max_seq_len):
        """Inputs for a prefill of `prefill_len` "tokens".

        Default uses input_ids derived from spec.bos_token + filler_tokens.
        Multi-modal specs (gemma3) override via `prefill_input_fn` to emit
        `inputs_embeds` instead. Output dict always includes attention_mask,
        position_ids (when has_position_ids), and zero past_key_values for
        all layers.
        """
        if self.prefill_input_fn is not None:
            return self.prefill_input_fn(cfg, prefill_len, max_seq_len)
        return make_prefill_inputs(cfg, self.make_prompt(prefill_len), max_seq_len)

    def build_decode_inputs(self, cfg, position, kv_cache, max_seq_len):
        """Inputs for a single-token decode at the given absolute position,
        with `kv_cache` (extract_kv_cache-shaped dict) as the past.

        Default uses a deterministic synthetic token id. Multi-modal specs
        (gemma3) override via `decode_input_fn` to emit `inputs_embeds`.
        """
        if self.decode_input_fn is not None:
            return self.decode_input_fn(cfg, position, kv_cache, max_seq_len)
        # Deterministic mid-vocab token derived from filler_tokens.
        token = int(self.filler_tokens[position % len(self.filler_tokens)])
        return make_decode_inputs(cfg, token, position, kv_cache, max_seq_len)


# ── Fetch helpers ───────────────────────────────────────────────────────────


def _hf_hub_download_one(hf_repo, filename, model_dir, subdir=None):
    """Download one file from a (potentially gated) HF repo into model_dir.

    When `subdir` is set, the file lives at `<subdir>/<filename>` in the repo
    and we flatten it into `model_dir/<filename>` after download (AMD's
    Qwen/Gemma3 repos use the "Upload * ONNX files/" subdir pattern). Skips
    cleanly when the repo is gated and `hf auth login` hasn't run.
    """
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
    except ImportError:
        pytest.skip(
            "huggingface_hub not installed — add to environment.yml "
            "or `pip install huggingface_hub`"
        )

    model_dir.mkdir(parents=True, exist_ok=True)
    repo_filename = f"{subdir}/{filename}" if subdir else filename
    try:
        src = hf_hub_download(
            repo_id=hf_repo, filename=repo_filename, local_dir=str(model_dir)
        )
    except (GatedRepoError, RepositoryNotFoundError) as e:
        pytest.skip(
            f"Cannot access {hf_repo}/{repo_filename}: {e}. "
            "Authenticate via `hf auth login` with an HF token that has access."
        )

    src_path = pathlib.Path(src)
    dst = model_dir / filename
    if src_path != dst:
        shutil.move(str(src_path), str(dst))
        if subdir:
            # Best-effort cleanup of the now-empty subfolder.
            try:
                (model_dir / subdir).rmdir()
            except OSError:
                pass
    mb = dst.stat().st_size / (1024 * 1024)
    print(f"  Fetched {filename} ({mb:.1f} MB)")
    return dst


def normalize_drop_inputs_embeds(spec):
    """Drop any dangling `inputs_embeds` graph input from spec.onnx_file.

    Several AMD-converted ONNX exports (Qwen2.5, DeepSeek-R1, gpt-oss-120b)
    declare `inputs_embeds` as a graph input but no node consumes it — token
    embedding happens internally via Gather(embed_weights, input_ids). ORT
    and OGA both treat declared inputs as required and refuse to run when
    an unconsumed input isn't fed. Idempotent — no-op if the input isn't
    declared or has already been removed.
    """
    import onnx

    onnx_path = spec.onnx_path
    m = onnx.load(str(onnx_path), load_external_data=False)
    before = len(m.graph.input)
    new_inputs = [i for i in m.graph.input if i.name != "inputs_embeds"]
    if len(new_inputs) == before:
        return
    del m.graph.input[:]
    m.graph.input.extend(new_inputs)
    onnx.save(m, str(onnx_path), save_as_external_data=False)
    print(
        f"  Normalized {onnx_path.name}: dropped graph input 'inputs_embeds' "
        "(declared but unused)"
    )


def fetch_model_files(spec):
    """Download all of spec's required files (ONNX + data blobs + any
    extra_data_files) into spec.model_dir, run any normalize hook, return
    absolute ONNX path.
    """
    spec.model_dir.mkdir(parents=True, exist_ok=True)
    required = [spec.onnx_file, *spec.data_files, *spec.extra_data_files]
    if spec.hf_base is not None:
        for f in required:
            dst = spec.model_dir / f
            if not dst.exists():
                download(f"{spec.hf_base}/{f}", dst)
    else:
        for f in required:
            if not (spec.model_dir / f).exists():
                _hf_hub_download_one(spec.hf_repo, f, spec.model_dir, spec.hf_subdir)

    if spec.normalize_onnx_hook is not None:
        spec.normalize_onnx_hook(spec)

    return str(spec.onnx_path)


def fetch_oga_files(spec):
    """Materialize OGA companion files (tokenizer, genai_config) for spec.

    Then migrate the on-disk genai_config.json to set search.chunk_size to
    spec.chunk_size (Pareto-optimal default for dynamic-shape models on
    MorphiZenEP — see CLAUDE.md "OGA chunked prefill" entry).
    """
    spec.model_dir.mkdir(parents=True, exist_ok=True)

    if spec.genai_config_template is not None:
        # Models without an HF-shipped genai_config (1B): write our template,
        # download tokenizer files from hf_root_for_tokenizer.
        if spec.hf_root_for_tokenizer is None:
            raise ValueError(
                f"{spec.name}: genai_config_template requires hf_root_for_tokenizer"
            )
        for fname in spec.oga_files:
            dst = spec.model_dir / fname
            if not dst.exists():
                download(f"{spec.hf_root_for_tokenizer}/{fname}", dst)
        cfg_path = spec.model_dir / "genai_config.json"
        if not cfg_path.exists():
            with open(cfg_path, "w") as f:
                json.dump(spec.genai_config_template, f, indent=4)
            print(f"  Generated {cfg_path.name}")
    elif spec.hf_base is not None:
        for fname in spec.oga_files:
            dst = spec.model_dir / fname
            if not dst.exists():
                download(f"{spec.hf_base}/{fname}", dst)
    else:
        # hf_repo: download each oga file that isn't already present.
        for fname in spec.oga_files:
            if not (spec.model_dir / fname).exists():
                _hf_hub_download_one(
                    spec.hf_repo, fname, spec.model_dir, spec.hf_subdir
                )

    if spec.chunk_size is not None:
        cfg_path = spec.model_dir / "genai_config.json"
        if cfg_path.exists():
            with open(cfg_path) as f:
                existing = json.load(f)
            if existing.get("search", {}).get("chunk_size") != spec.chunk_size:
                existing.setdefault("search", {})["chunk_size"] = spec.chunk_size
                with open(cfg_path, "w") as f:
                    json.dump(existing, f, indent=4)
                print(
                    f"  Migrated {cfg_path.name}: search.chunk_size={spec.chunk_size}"
                )


# ── Golden cache ────────────────────────────────────────────────────────────


def _hash_inputs(inputs):
    """Deterministic SHA256 over a {name: ndarray} dict."""
    h = hashlib.sha256()
    for name in sorted(inputs.keys()):
        arr = inputs[name]
        h.update(name.encode("utf-8"))
        h.update(str(arr.shape).encode("utf-8"))
        h.update(str(arr.dtype).encode("utf-8"))
        h.update(arr.tobytes())
    return h.hexdigest()


def _onnx_meta(spec):
    st = spec.onnx_path.stat()
    return {"onnx_size": st.st_size, "onnx_mtime": int(st.st_mtime)}


class GoldenStore:
    """CPU reference output cache under install/golden/<spec.name>/.

    Each entry: <key>.npz (arrays) + <key>.meta.json (onnx mtime/size +
    inputs hash + output names). Cache hit ⇔ all metadata matches; miss →
    silently regenerate via compute_fn and persist.

    Force full regen: `rm -rf install/golden`.
    Force one model: `rm -rf install/golden/<spec.name>`.
    """

    def __init__(self, root=GOLDEN_ROOT):
        self.root = pathlib.Path(root)

    def _entry_paths(self, spec, key):
        d = self.root / spec.name
        return d / f"{key}.npz", d / f"{key}.meta.json"

    def _load_if_fresh(self, npz_path, meta_path, spec, inputs_hash):
        if not (npz_path.exists() and meta_path.exists()):
            return None
        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except (OSError, json.JSONDecodeError):
            return None
        current = _onnx_meta(spec)
        if (
            meta.get("onnx_size") != current["onnx_size"]
            or meta.get("onnx_mtime") != current["onnx_mtime"]
            or meta.get("inputs_sha256") != inputs_hash
        ):
            return None
        names = meta.get("output_names")
        if not names:
            return None
        try:
            data = np.load(npz_path)
            outs = [data[f"out_{i}"] for i in range(len(names))]
            return outs, names
        except (OSError, KeyError):
            return None

    def _save(self, npz_path, meta_path, spec, inputs_hash, outputs, output_names):
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {f"out_{i}": np.asarray(arr) for i, arr in enumerate(outputs)}
        np.savez_compressed(npz_path, **arrays)
        meta = {
            **_onnx_meta(spec),
            "inputs_sha256": inputs_hash,
            "output_names": list(output_names),
            "spec_name": spec.name,
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    def get_or_compute(self, spec, key, inputs, compute_fn):
        """Returns (outputs_list, output_names).

        compute_fn() must return (outputs_list, output_names). Pytest.skips
        when compute_fn raises and spec.cpu_skip_predicate matches the
        exception — handles 120B's QMoE / bad_alloc reference-unavailability.
        """
        inputs_hash = _hash_inputs(inputs)
        npz_path, meta_path = self._entry_paths(spec, key)
        cached = self._load_if_fresh(npz_path, meta_path, spec, inputs_hash)
        if cached is not None:
            print(f"  [golden] cache hit: {spec.name}/{key}")
            return cached
        print(f"  [golden] cache MISS: {spec.name}/{key} — running CPU reference")
        try:
            outputs, output_names = compute_fn()
        except Exception as e:
            if spec.cpu_skip_predicate and spec.cpu_skip_predicate(e):
                pytest.skip(spec.cpu_skip_reason or f"CPU baseline unavailable: {e}")
            raise
        self._save(npz_path, meta_path, spec, inputs_hash, outputs, output_names)
        return outputs, output_names


@pytest.fixture(scope="session")
def golden_store():
    return GoldenStore()


# ── Base test classes ───────────────────────────────────────────────────────


def _cpu_ref_compute(model_path, inputs):
    """Run a CPU inference and return (outputs, output_names). Closes session."""
    cpu_sess = create_cpu_session(model_path)
    try:
        outs = cpu_sess.run(None, inputs)
        names = [o.name for o in cpu_sess.get_outputs()]
        return outs, names
    finally:
        cleanup(cpu_sess)


class BaseORTTests:
    """Canonical MorphiZen-EP-vs-CPU accuracy + latency suite.

    Subclasses MUST set `spec = SOME_MODEL_SPEC` at class level. The per-
    module fixtures (`dynamic_model_path`, `fixed_decode_path`,
    `fixed_prefill_128_path`, `ep_dynamic_session`,
    `ep_fixed_decode_session`, `ep_fixed_prefill_128_session`) are produced
    by `register_model_fixtures(spec)` in the per-model file.

    Test layout:
      - 6 accuracy tests run by default (test_ort_*).
      - 4 latency tests marked `@pytest.mark.latency` — skipped by default,
        opt in via `pytest -m latency` or `pytest --latency`. They reuse the
        same shared session as the accuracy tests so the marginal cost over
        accuracy alone is just `warmup + 3 timed iterations`.
    """

    spec: ClassVar[ModelSpec]

    def _golden(self, golden_store, model_path, key, inputs):
        return golden_store.get_or_compute(
            self.spec, key, inputs, lambda: _cpu_ref_compute(model_path, inputs)
        )

    # ── Accuracy tests (default) ────────────────────────────────────────────

    def test_ort_fixed_decode(
        self,
        fixed_decode_path,
        ep_fixed_decode_session,
        golden_store,
    ):
        """Fixed-shape decode (seq=1): accuracy vs CPU golden."""
        spec = self.spec
        cfg = spec.make_cfg(spec.max_seq_len)
        position = spec.decode_position
        inputs = spec.build_decode_inputs(
            cfg, position, make_zero_kv_cache(cfg, spec.max_seq_len), spec.max_seq_len
        )
        key = f"fixed_decode_sq1_kv{spec.max_seq_len}"
        ref, output_names = self._golden(golden_store, fixed_decode_path, key, inputs)
        test = run_iobinding_once(
            ep_fixed_decode_session, inputs, cfg, use_device_memory=False
        )
        ok, _ = compare_outputs(
            ref,
            test,
            output_names,
            "EP fixed decode sq=1",
            valid_seq=position + 1,
        )
        assert ok, "Fixed decode accuracy check failed"

    def test_ort_fixed_prefill_128(
        self,
        fixed_prefill_128_path,
        ep_fixed_prefill_128_session,
        golden_store,
    ):
        """Fixed-shape prefill (seq=128): accuracy vs CPU golden."""
        spec = self.spec
        cfg = spec.make_cfg(spec.max_seq_len)
        inputs = spec.build_prefill_inputs(cfg, 128, spec.max_seq_len)
        key = f"fixed_prefill_sq128_kv{spec.max_seq_len}"
        ref, output_names = self._golden(
            golden_store, fixed_prefill_128_path, key, inputs
        )
        test = run_iobinding_once(
            ep_fixed_prefill_128_session, inputs, cfg, use_device_memory=False
        )
        ok, _ = compare_outputs(
            ref,
            test,
            output_names,
            "EP fixed prefill sq=128",
            valid_seq=128,
        )
        assert ok, "Fixed prefill sq=128 accuracy check failed"

    def test_ort_dynamic_prefill_128(
        self, dynamic_model_path, ep_dynamic_session, golden_store
    ):
        """Dynamic-shape prefill (seq=128): accuracy vs CPU golden."""
        spec = self.spec
        seq_len = 128
        cfg = spec.make_cfg(seq_len)
        inputs = spec.build_prefill_inputs(cfg, seq_len, seq_len)
        dim_map = make_dim_map(seq_len, seq_len)
        key = f"dynamic_prefill_sq{seq_len}"
        ref, output_names = self._golden(golden_store, dynamic_model_path, key, inputs)
        test = run_iobinding_once(
            ep_dynamic_session, inputs, cfg, use_device_memory=False, dim_map=dim_map
        )
        ok, _ = compare_outputs(
            ref,
            test,
            output_names,
            f"EP dynamic prefill sq={seq_len}",
            valid_seq=seq_len,
        )
        assert ok, f"Dynamic prefill sq={seq_len} accuracy check failed"

    def test_ort_dynamic_decode(
        self,
        dynamic_model_path,
        ep_dynamic_session,
        golden_store,
    ):
        """Dynamic-shape decode (seq=1): accuracy vs CPU golden."""
        spec = self.spec
        cfg = spec.make_cfg(spec.max_seq_len)
        position = spec.decode_position
        inputs = spec.build_decode_inputs(
            cfg, position, make_zero_kv_cache(cfg, spec.max_seq_len), spec.max_seq_len
        )
        dim_map = make_dim_map(1, spec.max_seq_len)
        key = f"dynamic_decode_sq1_kv{spec.max_seq_len}"
        ref, output_names = self._golden(golden_store, dynamic_model_path, key, inputs)
        test = run_iobinding_once(
            ep_dynamic_session, inputs, cfg, use_device_memory=False, dim_map=dim_map
        )
        ok, _ = compare_outputs(
            ref,
            test,
            output_names,
            "EP dynamic decode sq=1",
            valid_seq=position + 1,
        )
        assert ok, "Dynamic decode accuracy check failed"

    def test_ort_output_allocator_dynamic(
        self, dynamic_model_path, repo_root, golden_store
    ):
        """Output-allocator mode (2-arg ABI) on dynamic shapes, one session.

        Opt-in per spec (output_allocator_e2e; enabled on Llama-1B). Drives
        SEVERAL distinct shapes through a SINGLE EP session (no recompilation
        between shapes) and checks each against the CPU golden, exercising both
        the host-output D2H path (use_device_memory=False, all outputs land in
        CPU memory) and the GPU zero-copy path (use_device_memory=True). Reuses
        the existing prefill/decode goldens so the CPU references are shared with
        the other accuracy tests.
        """
        spec = self.spec
        if not spec.output_allocator_e2e:
            pytest.skip("output_allocator_e2e not enabled for this spec")

        # One EP session reused for every shape below — proves a single compiled
        # DLL handles dynamic shapes (the output allocator is the only ABI).
        sess = create_ep_session(dynamic_model_path, repo_root)
        try:
            # 1) Dynamic-shape stress: two prefill lengths, host D2H path.
            #    sq=128 reuses the classic golden key; sq=64 is a distinct shape
            #    in the SAME session (no recompile).
            for seq_len, key in (
                (128, "dynamic_prefill_sq128"),
                (64, "dynamic_prefill_sq64"),
            ):
                cfg = spec.make_cfg(seq_len)
                inputs = spec.build_prefill_inputs(cfg, seq_len, seq_len)
                dim_map = make_dim_map(seq_len, seq_len)
                ref, output_names = self._golden(
                    golden_store, dynamic_model_path, key, inputs
                )
                test = run_iobinding_once(
                    sess, inputs, cfg, use_device_memory=False, dim_map=dim_map
                )
                ok, _ = compare_outputs(
                    ref,
                    test,
                    output_names,
                    f"EP allocator dynamic prefill sq={seq_len}",
                    valid_seq=seq_len,
                )
                assert ok, f"Allocator-mode dynamic prefill sq={seq_len} failed"

            # 2) Decode (seq=1) in the same session — host D2H path.
            cfg = spec.make_cfg(spec.max_seq_len)
            position = spec.decode_position
            inputs = spec.build_decode_inputs(
                cfg,
                position,
                make_zero_kv_cache(cfg, spec.max_seq_len),
                spec.max_seq_len,
            )
            dim_map = make_dim_map(1, spec.max_seq_len)
            key = f"dynamic_decode_sq1_kv{spec.max_seq_len}"
            ref, output_names = self._golden(
                golden_store, dynamic_model_path, key, inputs
            )
            test = run_iobinding_once(
                sess, inputs, cfg, use_device_memory=False, dim_map=dim_map
            )
            ok, _ = compare_outputs(
                ref,
                test,
                output_names,
                "EP allocator dynamic decode sq=1",
                valid_seq=position + 1,
            )
            assert ok, "Allocator-mode dynamic decode failed"

            # 3) GPU zero-copy path: device-memory IOBinding. Use PREFILL (past
            #    KV is not attended at past_len=0, so the uninitialized device
            #    KV buffer is never read) so the result is still comparable to
            #    the zero-past CPU golden. This validates the callback returning
            #    ORT's GPU pointer directly (no host staging).
            #    Best-effort: allocating an AMD-device OrtValue from Python needs
            #    an ORT build with that capability; the stock CPU/DML wheel raises
            #    "Can't allocate memory on the AMD device". Skip just this
            #    sub-check there (host-D2H + dynamic stress above still ran). GPU
            #    zero-copy is additionally covered by the OGA suite (EP GPU
            #    allocator) and the opt-in latency tests.
            cfg = spec.make_cfg(128)
            inputs = spec.build_prefill_inputs(cfg, 128, 128)
            dim_map = make_dim_map(128, 128)
            ref, output_names = self._golden(
                golden_store, dynamic_model_path, "dynamic_prefill_sq128", inputs
            )
            try:
                test_gpu = run_iobinding_once(
                    sess, inputs, cfg, use_device_memory=True, dim_map=dim_map
                )
            except RuntimeError as e:
                if "AMD device" not in str(e):
                    raise
                print(f"  [skip] GPU zero-copy sub-check (no device memory): {e}")
            else:
                ok, _ = compare_outputs(
                    ref,
                    test_gpu,
                    output_names,
                    "EP allocator dynamic prefill sq=128 (GPU zero-copy)",
                    valid_seq=128,
                )
                assert ok, "Allocator-mode GPU zero-copy prefill failed"
        finally:
            cleanup(sess)

    # ── Latency tests (opt-in via `pytest -m latency` or `--latency`) ──────

    @pytest.mark.latency
    def test_ort_fixed_decode_latency(self, ep_fixed_decode_session):
        spec = self.spec
        cfg = spec.make_cfg(spec.max_seq_len)
        position = spec.decode_position
        inputs = spec.build_decode_inputs(
            cfg, position, make_zero_kv_cache(cfg, spec.max_seq_len), spec.max_seq_len
        )
        times = run_timed_iobinding(
            ep_fixed_decode_session, inputs, cfg, use_device_memory=True
        )
        report(f"EP fixed decode (sq=1 kv={spec.max_seq_len})", times)

    @pytest.mark.latency
    def test_ort_fixed_prefill_128_latency(self, ep_fixed_prefill_128_session):
        spec = self.spec
        cfg = spec.make_cfg(spec.max_seq_len)
        inputs = spec.build_prefill_inputs(cfg, 128, spec.max_seq_len)
        times = run_timed_iobinding(
            ep_fixed_prefill_128_session, inputs, cfg, use_device_memory=True
        )
        report(f"EP fixed prefill (sq=128 kv={spec.max_seq_len})", times)

    @pytest.mark.latency
    def test_ort_dynamic_prefill_128_latency(self, ep_dynamic_session):
        spec = self.spec
        seq_len = 128
        cfg = spec.make_cfg(seq_len)
        inputs = spec.build_prefill_inputs(cfg, seq_len, seq_len)
        dim_map = make_dim_map(seq_len, seq_len)
        times = run_timed_iobinding(
            ep_dynamic_session, inputs, cfg, use_device_memory=True, dim_map=dim_map
        )
        report(f"EP dynamic prefill (sq={seq_len})", times)

    @pytest.mark.latency
    def test_ort_dynamic_decode_latency(self, ep_dynamic_session):
        spec = self.spec
        cfg = spec.make_cfg(spec.max_seq_len)
        position = spec.decode_position
        inputs = spec.build_decode_inputs(
            cfg, position, make_zero_kv_cache(cfg, spec.max_seq_len), spec.max_seq_len
        )
        dim_map = make_dim_map(1, spec.max_seq_len)
        times = run_timed_iobinding(
            ep_dynamic_session, inputs, cfg, use_device_memory=True, dim_map=dim_map
        )
        report(f"EP dynamic decode (sq=1 kv={spec.max_seq_len})", times)

    def test_ort_dynamic_vs_fixed(
        self,
        dynamic_model_path,
        fixed_decode_path,
        repo_root,
    ):
        """Dynamic and fixed shape models produce same EP output (decode).

        Creates its own EP sessions inline (does NOT take the shared
        `ep_*_session` fixtures) and tears down the first before opening the
        second — keeps the peak memory at one session even when
        reuse_ep_session=True, so large-model specs (70B+) can still run
        this test without OOM. Pure EP-vs-EP, no CPU baseline, no golden.
        """
        spec = self.spec
        cfg = spec.make_cfg(spec.max_seq_len)
        position = spec.decode_position
        inputs = spec.build_decode_inputs(
            cfg, position, make_zero_kv_cache(cfg, spec.max_seq_len), spec.max_seq_len
        )
        dim_map = make_dim_map(1, spec.max_seq_len)

        fixed_sess = create_ep_session(fixed_decode_path, repo_root)
        fixed_out = run_iobinding_once(fixed_sess, inputs, cfg, use_device_memory=False)
        output_names = [o.name for o in fixed_sess.get_outputs()]
        cleanup(fixed_sess)

        dyn_sess = create_ep_session(dynamic_model_path, repo_root)
        dyn_out = run_iobinding_once(
            dyn_sess, inputs, cfg, use_device_memory=False, dim_map=dim_map
        )
        cleanup(dyn_sess)

        ok, _ = compare_outputs(
            fixed_out,
            dyn_out,
            output_names,
            "dynamic vs fixed decode",
            valid_seq=position + 1,
        )
        assert ok, "Dynamic vs fixed shape comparison failed"

    def test_ort_per_step_logits(
        self, dynamic_model_path, ep_dynamic_session, golden_store
    ):
        """Prefill + N-1 decode steps; compare per-step logits vs CPU golden.

        Caches just the per-step logits (not KV), so the npz stays small even
        for 70B+ models. EP runs its own chain (independent KV); compare
        each step's logits against the cached CPU step's logits. After
        token-divergence, both chains stay internally consistent — the
        comparison still measures kernel accuracy at each step.

        Uses spec.build_prefill_inputs / spec.build_decode_inputs so
        multimodal specs (gemma3) plug in their inputs_embeds builders
        without needing a custom test method.
        """
        spec = self.spec
        prefill_len = len(spec.prompt_tokens)
        n_steps = spec.num_generate_tokens
        cfg = spec.make_cfg(spec.max_seq_len)

        prefill_inputs = spec.build_prefill_inputs(cfg, prefill_len, spec.max_seq_len)

        key = f"per_step_logits_prompt{prefill_len}_gen{n_steps}"

        def compute():
            cpu_sess = create_cpu_session(dynamic_model_path)
            try:
                names = [o.name for o in cpu_sess.get_outputs()]
                logits_idx = names.index("logits")
                outs = cpu_sess.run(None, prefill_inputs)
                step_logits = [outs[logits_idx]]
                kv = extract_kv_cache(outs, names)
                for step in range(n_steps - 1):
                    pos = prefill_len + step
                    decode_inputs = spec.build_decode_inputs(
                        cfg, pos, kv, spec.max_seq_len
                    )
                    outs = cpu_sess.run(None, decode_inputs)
                    step_logits.append(outs[logits_idx])
                    kv = extract_kv_cache(outs, names)
                step_names = [f"step_{i}" for i in range(len(step_logits))]
                return step_logits, step_names
            finally:
                cleanup(cpu_sess)

        cpu_step_logits, _ = golden_store.get_or_compute(
            spec, key, prefill_inputs, compute
        )

        # ── EP side: run its own prefill + decode chain on the shared
        # `ep_dynamic_session` (no per-test session re-init). ──
        ep_sess = ep_dynamic_session
        ep_names = [o.name for o in ep_sess.get_outputs()]
        logits_idx = ep_names.index("logits")

        print(f"\n{'=' * 60}")
        print(
            f"EP vs CPU(golden) per-step logits: prompt={prefill_len}, "
            f"generate={n_steps}"
        )
        print(f"{'=' * 60}")

        all_ok = True
        ep_out = ep_sess.run(None, prefill_inputs)
        print(f"\n  Prefill (seq_len={prefill_len}):")
        ok = compare_logits(cpu_step_logits[0], ep_out[logits_idx], "prefill")
        all_ok = all_ok and ok

        ep_kv = extract_kv_cache(ep_out, ep_names)

        for step in range(n_steps - 1):
            pos = prefill_len + step
            decode_inputs = spec.build_decode_inputs(cfg, pos, ep_kv, spec.max_seq_len)
            ep_out = ep_sess.run(None, decode_inputs)
            ok = compare_logits(
                cpu_step_logits[step + 1],
                ep_out[logits_idx],
                f"decode[{step + 1}]",
            )
            all_ok = all_ok and ok
            ep_kv = extract_kv_cache(ep_out, ep_names)

        print(f"{'=' * 60}")
        assert all_ok, "Per-step logits accuracy check failed"


# ── Base OGA tests ──────────────────────────────────────────────────────────


def _oga_setup_and_patch(spec, repo_root):
    """Common preamble for every OGA test: setup, ensure files, patch config."""
    og, ep_dll = setup_oga_ep(repo_root)
    fetch_model_files(spec)
    fetch_oga_files(spec)
    patch_genai_config_for_morphizen(spec.model_dir, ep_dll)
    return og, ep_dll


def _oga_load_model_or_skip(og, source):
    """`og.Model(source)` wrapped to skip cleanly when OGA doesn't know our EP."""
    try:
        return og.Model(source)
    except RuntimeError as e:
        if "Unknown provider name" in str(e):
            pytest.skip("OGA does not recognize MorphiZen EP")
        raise


@contextlib.contextmanager
def hipdnn_strict_for_oga(spec):
    """Toggle HIPDNN_EP_STRICT around an OGA `og.Model(...)` load per
    `spec.oga_strict`. Restores the previous value on exit.

    Default conftest behaviour is STRICT=1 (catches silent CPU fallbacks).
    Gemma3 OGA sets `oga_strict=False` because its multimodal sub-sessions
    (embedding/vision/speech) intentionally fail graph claim — STRICT=1
    would abort. See the CLAUDE.md gemma3 OGA gotcha.
    """
    prev = os.environ.get("HIPDNN_EP_STRICT")
    if spec.oga_strict:
        os.environ["HIPDNN_EP_STRICT"] = "1"
    else:
        os.environ.pop("HIPDNN_EP_STRICT", None)
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("HIPDNN_EP_STRICT", None)
        else:
            os.environ["HIPDNN_EP_STRICT"] = prev


class BaseOGATests:
    """Canonical 4-test OGA-on-MorphiZen-EP suite.

    Subclasses MUST set `spec`. The two default-config tests
    (`test_oga_ep_generation`, `test_oga_ep_shape_switching`) share a
    module-scope `oga_default_model` fixture when
    `spec.reuse_oga_default_model=True` (default) — saves one Model load.
    The other two tests use config overlays (past_present_share_buffer=false,
    custom chunk_size) so they create their own ephemeral Models.
    """

    spec: ClassVar[ModelSpec]

    def test_oga_ep_generation(self, oga_default_model):
        """OGA + MorphiZen EP latency at prompt_len=128."""
        spec = self.spec
        og, model = oga_default_model
        tokenizer = og.Tokenizer(model)
        tokens = spec.make_prompt(128)
        generated, ttft_ms, tps = oga_generate_timed(
            og, model, tokenizer, tokens, max_new=128
        )
        print(
            f"\n  OGA EP prompt=128: prefill={ttft_ms:7.1f}ms  "
            f"tps={tps:5.1f}  generated={generated}"
        )
        assert generated > 0

    def test_oga_ep_no_share_buffer(self, repo_root):
        """Verify generation works with past_present_share_buffer=false."""
        spec = self.spec
        og, _ = _oga_setup_and_patch(spec, repo_root)
        try:
            config = og.Config(str(spec.model_dir))
            config.overlay(json.dumps({"search": {"past_present_share_buffer": False}}))
            with hipdnn_strict_for_oga(spec):
                model = _oga_load_model_or_skip(og, config)
            tokenizer = og.Tokenizer(model)
            tokens = spec.make_prompt(128)
            generated, ttft_ms, tps = oga_generate(
                og, model, tokenizer, tokens, max_new=10
            )
            print(
                f"\n  OGA EP no_share_buffer: generated={len(generated)}"
                f" ttft={ttft_ms:.1f}ms tps={tps:.1f}"
            )
            assert len(generated) > 0, (
                "No tokens generated with past_present_share_buffer=false"
            )
            del model
            gc.collect()
        finally:
            restore_genai_config(spec.model_dir)

    def test_oga_ep_shape_switching(self, oga_default_model):
        """Different prompt lengths via rewind_to on a single Generator.

        Reuses one Generator (rewind_to(0) between prompts) so the EP's
        constants and GEMM algorithm cache stay warm.
        """
        spec = self.spec
        og, model = oga_default_model
        max_prompt = 128
        max_new = 5
        params = og.GeneratorParams(model)
        params.set_search_options(max_length=max_prompt + max_new, do_sample=False)
        generator = og.Generator(model, params)
        try:
            print(f"\n{'=' * 60}")
            print("OGA EP shape switching test")
            print(f"{'=' * 60}")
            for prompt_len in [7, 128, 64, 128, 7]:
                generator.rewind_to(0)
                tokens = spec.make_prompt(prompt_len)
                t0 = time.perf_counter()
                generator.append_tokens(np.array(tokens, dtype=np.int32))
                generator.generate_next_token()
                ttft_ms = (time.perf_counter() - t0) * 1000
                generated = [int(generator.get_next_tokens()[0])]
                token_times_ms = []
                while not generator.is_done() and len(generated) < max_new:
                    t_tok = time.perf_counter()
                    generator.generate_next_token()
                    token_times_ms.append((time.perf_counter() - t_tok) * 1000)
                    generated.append(int(generator.get_next_tokens()[0]))
                tok_str = "  ".join(f"{t:.1f}" for t in token_times_ms)
                print(
                    f"  prompt={prompt_len:>3d}: "
                    f"prefill={ttft_ms:7.1f}ms  "
                    f"decode=[{tok_str}] ms"
                )
                assert len(generated) > 0
            print(f"{'=' * 60}")
        finally:
            del generator
            gc.collect()

    def test_oga_ep_chunked_prefill(self, dynamic_model_path, repo_root, golden_store):
        """Chunked prefill accuracy: OGA+EP with chunk_size=128 vs CPU golden.

        The CPU reference (`run_cpu_reference_generation`) is the slow side
        and gets cached. EP+OGA always re-runs.
        """
        spec = self.spec
        og, _ = _oga_setup_and_patch(spec, repo_root)

        chunk_size = 128
        prompt_len = 200
        max_new = 10
        prompt_tokens = spec.make_prompt(prompt_len)
        max_seq_len = prompt_len + max_new
        cfg = spec.make_cfg(max_seq_len)

        # Build a small "inputs" placeholder for hashing — the CPU chain is
        # fully determined by the prompt + cfg + dynamic_model_path. Include
        # the prompt array so a prompt change invalidates the cache.
        key = f"oga_chunked_prefill_prompt{prompt_len}_gen{max_new}"
        prompt_for_hash = {
            "prompt": np.array(prompt_tokens, dtype=np.int64),
        }

        def compute():
            tokens = run_cpu_reference_generation(
                dynamic_model_path, cfg, prompt_tokens, max_new, max_seq_len
            )
            return [np.array(tokens, dtype=np.int64)], ["tokens"]

        cpu_arrays, _ = golden_store.get_or_compute(spec, key, prompt_for_hash, compute)
        cpu_generated = [int(t) for t in cpu_arrays[0]]

        # ── OGA+EP with chunked prefill ──
        try:
            config = og.Config(str(spec.model_dir))
            config.overlay(json.dumps({"search": {"chunk_size": chunk_size}}))
            with hipdnn_strict_for_oga(spec):
                model = _oga_load_model_or_skip(og, config)
            tokenizer = og.Tokenizer(model)
            oga_generated, _, _ = oga_generate(
                og, model, tokenizer, prompt_tokens, max_new=max_new
            )
            del model
            gc.collect()
        finally:
            restore_genai_config(spec.model_dir)

        n = min(len(cpu_generated), len(oga_generated))
        cpu_trimmed = cpu_generated[:n]
        oga_trimmed = oga_generated[:n]
        matches = sum(1 for a, b in zip(cpu_trimmed, oga_trimmed) if a == b)
        match_rate = matches / n if n > 0 else 0.0

        print(f"\n{'=' * 60}")
        print(
            f"Chunked prefill accuracy (chunk_size={chunk_size}, "
            f"prompt={prompt_len}, generate={max_new})"
        )
        print(f"  CPU  tokens: {cpu_trimmed}")
        print(f"  OGA  tokens: {oga_trimmed}")
        print(f"  Match rate:  {match_rate:.0%} ({matches}/{n})")
        print(f"{'=' * 60}")

        assert len(oga_generated) > 0, "OGA generated no tokens"
        assert match_rate >= 0.5, (
            f"Token match rate {match_rate:.0%} too low — chunked prefill "
            f"may be producing incorrect results"
        )


# ── Per-spec fixture registration ───────────────────────────────────────────


def register_model_fixtures(spec):
    """Return the per-module fixtures a per-model file needs, bound to spec.

    The caller MUST assign the returned tuple to module-globals with the
    exact fixture names so pytest discovers them:

        (dynamic_model_path, fixed_decode_path, fixed_prefill_128_path,
         ep_dynamic_session, ep_fixed_decode_session,
         ep_fixed_prefill_128_session, oga_default_model) = (
            register_model_fixtures(SPEC)
        )

    Session-reuse fixtures (`ep_*_session`, `oga_default_model`) use module
    scope when spec.reuse_ep_session / reuse_oga_default_model is True
    (default), so subsequent tests in the file skip the ~3-15 s session-init
    cost. For very large models (set False), they fall back to function
    scope — identical to the un-shared, function-scope-per-test baseline.

    `test_ort_dynamic_vs_fixed` intentionally does NOT take the shared
    session fixtures; it creates and tears down its own sessions inline so
    that the two-session memory peak stays bounded even on 70B-scale models.
    """

    @pytest.fixture(scope="session")
    def dynamic_model_path():
        return fetch_model_files(spec)

    @pytest.fixture(scope="session")
    def fixed_decode_path(dynamic_model_path):
        return ensure_fixed_model(
            spec.model_dir,
            spec.onnx_file,
            spec.data_files,
            seq_len=1,
            kv_len=spec.max_seq_len,
        )

    @pytest.fixture(scope="session")
    def fixed_prefill_128_path(dynamic_model_path):
        return ensure_fixed_model(
            spec.model_dir,
            spec.onnx_file,
            spec.data_files,
            seq_len=128,
            kv_len=spec.max_seq_len,
        )

    ep_scope = "module" if spec.reuse_ep_session else "function"

    @pytest.fixture(scope=ep_scope)
    def ep_dynamic_session(dynamic_model_path, repo_root):
        sess = create_ep_session(dynamic_model_path, repo_root)
        yield sess
        cleanup(sess)

    @pytest.fixture(scope=ep_scope)
    def ep_fixed_decode_session(fixed_decode_path, repo_root):
        sess = create_ep_session(fixed_decode_path, repo_root)
        yield sess
        cleanup(sess)

    @pytest.fixture(scope=ep_scope)
    def ep_fixed_prefill_128_session(fixed_prefill_128_path, repo_root):
        sess = create_ep_session(fixed_prefill_128_path, repo_root)
        yield sess
        cleanup(sess)

    oga_scope = "module" if spec.reuse_oga_default_model else "function"

    @pytest.fixture(scope=oga_scope)
    def oga_default_model(repo_root):
        """Default-config OGA Model + og module, patched + ready for inference.

        Shared across `test_oga_ep_generation` and `test_oga_ep_shape_switching`
        (both use the unpatched-overlay config). The other two OGA tests need
        per-test config overlays (past_present_share_buffer=false, custom
        chunk_size) so they create their own ephemeral Model objects.

        HIPDNN_EP_STRICT is toggled around the og.Model load via
        `hipdnn_strict_for_oga(spec)` — gemma3-style multimodal specs
        opt out of strict to allow sub-session graph-claim-then-fail.
        """
        og, ep_dll = setup_oga_ep(repo_root)
        fetch_model_files(spec)
        fetch_oga_files(spec)
        patch_genai_config_for_morphizen(spec.model_dir, ep_dll)
        try:
            with hipdnn_strict_for_oga(spec):
                model = _oga_load_model_or_skip(og, str(spec.model_dir))
            yield og, model
        finally:
            try:
                del model
            except UnboundLocalError:
                pass
            gc.collect()
            restore_genai_config(spec.model_dir)

    return (
        dynamic_model_path,
        fixed_decode_path,
        fixed_prefill_128_path,
        ep_dynamic_session,
        ep_fixed_decode_session,
        ep_fixed_prefill_128_session,
        oga_default_model,
    )


# ── Pytest hooks: --latency flag, latency marker, per-spec markers ─────────


def pytest_addoption(parser):
    """`--latency` opts into the latency-marked tests (skipped by default).

    Equivalent: `pytest -m latency ...` selects only latency tests AND
    implicitly opts in. Without either, latency tests are skipped with a
    pointer to `--latency` in the skip reason.
    """
    parser.addoption(
        "--latency",
        action="store_true",
        default=False,
        help="Run latency-marked tests (skipped by default).",
    )


def pytest_configure(config):
    """Register the `latency` marker so `-m latency` works without warnings."""
    config.addinivalue_line(
        "markers",
        "latency: opt-in latency / TPS measurement test (skipped by default; "
        "run with --latency or `-m latency`).",
    )


def _latency_explicitly_selected(config):
    """True when the user is asking for latency tests (any of these):

    - `--latency` flag set
    - `-m latency` (or marker expression that includes latency without
      a "not" qualifier — covers `-m "latency or X"` too)
    """
    if config.getoption("--latency"):
        return True
    marker_expr = config.getoption("-m") or ""
    return "latency" in marker_expr and "not latency" not in marker_expr


def pytest_collection_modifyitems(config, items):
    """Two responsibilities:

    1. Skip latency-marked tests by default. Opt in with `--latency` or
       `-m latency`. (See `pytest_addoption` / `pytest_configure`.)
    2. Apply per-spec markers (skip / xfail) declared on `cls.spec.markers`,
       keeping per-model files declarative.
    """
    latency_enabled = _latency_explicitly_selected(config)
    skip_latency = pytest.mark.skip(
        reason="latency test — run with `pytest --latency` or `-m latency`"
    )

    for item in items:
        # 1. Latency gate.
        if "latency" in item.keywords and not latency_enabled:
            item.add_marker(skip_latency)

        # 2. Per-spec markers.
        cls = getattr(item, "cls", None)
        if cls is None:
            continue
        spec = getattr(cls, "spec", None)
        if spec is None or not getattr(spec, "markers", None):
            continue
        method_name = item.originalname or item.name.split("[")[0]
        marker = spec.markers.get(method_name)
        if marker is not None:
            item.add_marker(marker)
