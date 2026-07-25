#!/usr/bin/env python3
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""
Export fixed-shape Llama prefill/decode ONNX variants.

- batch_size = 1.
- **max_length** (default 16384, CLI ``--max-length``): this matches the graph-level
  meaning of past_sequence_length, total_sequence_length, the GQA total scalar, the
  second dimension of attention_mask, and related shapes; it is not just another name
  for "cache_len".
- Except fixed-prompt variants, the ONNX / genai filename tag is ``p{chunk}m{max_length}``,
  e.g. ``prefill_p128m16384.onnx`` and ``genai_config_p128m16384.json``.
- **Fixed-prompt variants** (``decoder.fixed_prompt_length`` with no ``sliding_window``):
  additionally export fixed ONNX / genai variants for ``S∈{128,2048,12200}``; the stem is
  ``128``, ``2k`` (for ``S=2048``, while JSON ``fixed_prompt_length`` stays ``2048``), and
  ``12200``.
  ``128`` and ``2048`` still keep their sliding-window variants ``prefill_pSm...``;
  ``12200`` is fixed-only (no ``p12200m...`` variant).
  **Control fixed-prompt** (chunk=0 only, when ``128`` / ``2048`` appear in the full export
  matrix ``DEFAULT_SEQ_LENS`` used with ``--all``):
  ``prefill_128_m256.onnx`` / ``genai_config_128_m256.json`` (``context_length``/KV = 256,
  ``fixed_prompt_length`` 128) and ``prefill_2k_m3072.onnx`` /
  ``genai_config_2k_m3072.json`` (3072 / 2048; stem ``2k`` like other fixed 2048 files).
- **Default (no ``--all``):** export a single sliding-window variant only:
  ``prefill_p512m{max_length}.onnx``, ``decode_p512m{max_length}.onnx``, and
  ``genai_config_p512m{max_length}.json`` (plus shared external weights as today).
- **With ``--all``:** export the full fixed matrix (same tuple as ``DEFAULT_SEQ_LENS`` in
  code: ``128,256,512,1024,2048,3072,4096,12200`` and their expanded variants); decode
  maps one-to-one with prefill.
- input_ids: [1, S] (prefill) or [1, 1] (decode); attention_mask: [1, max_length];
  position_ids: [1, S] (prefill) or [1, 1] (decode), matching the current-step sequence.
- past_key_values.*.key/value: the 3rd dimension is forced to max_length, with layout
  [batch, num_kv_heads, past_sequence_length, head_size].
- The 3rd dimension of ``present.*.key/value`` graph outputs and matching ``value_info``
  is also forced to max_length (source ONNX often has concrete values like 256, and
  symbolic substitution alone does not update those).
- If GroupQueryAttention input 5 is ``attn_mask_reformat/.../Cast/output_0``, keep the
  original edge and do not replace it with a seqlens constant.
- If the original ``Constant fold/total_seq_len`` → ``total_seq_len_constant`` becomes
  unreferenced after rewiring GQA total input, remove it to avoid duplicating the new
  injected total constant; a backup copy is restored between variants for continued export.
- Before saving each variant, run **dead-code elimination**: mark tensors reachable
  backwards from ``graph.output``, remove nodes that cannot affect any output, trim
  ``value_info``, drop unused ``graph.input``, and remove unreferenced ``initializer``s
  (e.g. isolated Gather/Cast chains in ``attn_mask_subgraph``).
- Prefill and Decode ONNX files are written **directly** under ``--output`` / ``-o`` (no
  ``full_model/`` subfolder); external weight filenames and shard layout match the source
  ONNX exactly (e.g. Llama
  ``model.onnx.data``, GPT-OSS ``model_q4f16.onnx_data``, ``model_q4f16.onnx_data_1``, etc.).
  If source has no external data, default remains ``model.onnx.data``. All variants share
  the same external weight files: only the first variant writes blobs, later variants write
  only their ``.onnx`` graph (no weight re-read, no ``*.data`` rewrite).
  ONNX basenames are always ``prefill_{tag}.onnx`` or ``decode_{tag}.onnx`` (including when
  ``extract_full_model`` takes the layered-graph path that used to emit ``full_model_``).
  For multi-shard files (e.g. ``model_q4f16.onnx_data``, ``_1``, ...), repacking preserves
  source ``location``/``offset`` and writes all mapped initializers (including pieces <1024B);
  script-added unmapped initializers stay embedded in ``.onnx`` to avoid polluting shard 0.
- If ``--model`` is provided and ``--no-genai-config`` is not set, write one
  ``genai_config_{tag}.json`` per variant under ``--output`` (same folder as the ONNX).
  The template path
  must be provided via ``-T`` / ``--config-template`` (choose per model family; e.g. Llama
  can use ``splite_onnx/genai_config.json`` in this repo). The script fills fields including
  ``context_length``, prefill/decode filenames, and ``search.max_length``.
  If the source ONNX does not contain ``QMoE`` nodes, also write
  ``genai_config_{tag}_dml.json`` for the three fixed-prompt variants
  (``genai_config_128.json``, ``genai_config_2k.json``, ``genai_config_12200.json``):
  only ``decoder.session_options`` is switched to DirectML
  (``provider_options: [{ "dml": { "device_id": "0" } }]``), all other fields match the
  non-DML config.
  For chunk=0 (fixed-prompt 128 / 2048 / 12200), ensure ``decoder.fixed_prompt_length``
  exists (the template may omit it) and remove ``sliding_window``.
  For chunk!=0, remove ``fixed_prompt_length`` and set ``sliding_window.window_size``.

Reuses ``SingleOpExtractor.extract_full_model`` via subclass; shares small helpers
with ``extract_submodels.py`` (total-seq initializer / fold Constant).
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import re
import sys
from typing import Any

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import onnx
from onnx import numpy_helper
from onnx import TensorProto
from onnx.external_data_helper import set_external_data

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_extract_submodels():
    """Load sibling ``extract_submodels.py`` without modifying ``sys.path``."""
    path = os.path.join(_SCRIPT_DIR, "extract_submodels.py")
    spec = importlib.util.spec_from_file_location("extract_submodels", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load extract_submodels from {path!r}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["extract_submodels"] = mod
    spec.loader.exec_module(mod)
    return mod


_es = _load_extract_submodels()
BATCH_SIZE = _es.BATCH_SIZE
FOLD_TOTAL_SEQ_NODE_NAME = _es.FOLD_TOTAL_SEQ_NODE_NAME
TOTAL_SEQ_LEN_CONSTANT_OUT = _es.TOTAL_SEQ_LEN_CONSTANT_OUT
SingleOpExtractor = _es.SingleOpExtractor
_build_dim_subs = _es._build_dim_subs
patch_fold_total_seq_len_constant_value = _es.patch_fold_total_seq_len_constant_value
total_seq_len_scalar_from_dim_map = _es.total_seq_len_scalar_from_dim_map
upsert_total_seq_len_const_initializer = _es.upsert_total_seq_len_const_initializer


# Full export matrix when CLI ``--all`` is set (not configurable).
DEFAULT_SEQ_LENS = (128, 256, 512, 1024, 2048, 3072, 4096, 12200)
# Default quick export: one sliding variant ``p512m{max_length}`` only.
DEFAULT_QUICK_EXPORT_SEQ_LENS = (512,)
GQA_INIT_PREFIX = "gqa_ctxfix_"

# Fallback when the source ONNX has no external-data initializers
DEFAULT_EXTERNAL_WEIGHTS_FILENAME = "model.onnx.data"

DEFAULT_GENAI_CONFIG_TEMPLATE = os.path.join(_SCRIPT_DIR, "genai_config.json")


def _tensor_external_kv(tensor: onnx.TensorProto) -> dict[str, str]:
    return {e.key: e.value for e in tensor.external_data}


def _initializer_external_loc_off(
    tensor: onnx.TensorProto,
) -> tuple[str | None, int]:
    """Return (location, offset) from TensorProto.external_data; offset 0 if absent."""
    kv = _tensor_external_kv(tensor)
    loc = kv.get("location")
    if not loc:
        return None, 0
    off = int(kv["offset"]) if "offset" in kv else 0
    return loc, off


@dataclass(frozen=True)
class ExternalDataLayout:
    """How large initializers were split across files in the source ONNX."""

    ordered_locations: tuple[str, ...]
    init_to_location: dict[str, str]
    init_pack_order: dict[str, tuple[int, int]]  # name -> (file_index, offset)


def scan_external_data_layout(model_path: str) -> ExternalDataLayout | None:
    """Scan ``model_path`` (load_external_data=False) for initializer external_data.

    Returns ``None`` if no initializer uses external storage.
    """
    m = onnx.load(model_path, load_external_data=False)
    rows: list[tuple[str, str, int]] = []
    for init in m.graph.initializer:
        loc, off = _initializer_external_loc_off(init)
        if not loc:
            continue
        rows.append((init.name, loc, off))
    if not rows:
        return None

    ordered: list[str] = []
    for init in m.graph.initializer:
        loc, _ = _initializer_external_loc_off(init)
        if loc and loc not in ordered:
            ordered.append(loc)
    file_index = {loc: i for i, loc in enumerate(ordered)}
    init_to_location = {name: loc for name, loc, _ in rows}
    init_pack_order = {name: (file_index[loc], off) for name, loc, off in rows}
    return ExternalDataLayout(
        ordered_locations=tuple(ordered),
        init_to_location=init_to_location,
        init_pack_order=init_pack_order,
    )


def _stale_external_filenames(layout: ExternalDataLayout | None) -> list[str]:
    """External weight basenames to remove in the output directory before a fresh export."""
    names = {DEFAULT_EXTERNAL_WEIGHTS_FILENAME}
    if layout:
        names.update(layout.ordered_locations)
    return sorted(names)


def _tensor_raw_byte_len(t: onnx.TensorProto) -> int:
    if not t.HasField("raw_data"):
        return 0
    return len(t.raw_data)


def _needs_external_blob_rebuild(
    model: onnx.ModelProto,
    size_threshold: int,
    layout: ExternalDataLayout | None,
) -> bool:
    """True if external blob(s) must be rewritten from ``raw_data``.

    For **multi-shard** models (several ``*.onnx_data`` files), any initializer that
    the source maps to a shard must be packed into blobs **regardless of size**;
    skipping tensors under ``size_threshold`` would truncate shard 0 and desync
    MD5 vs the original exporter. Single-shard / default path still uses
    ``size_threshold`` like ``onnx.save``.
    """
    multi = layout is not None and len(layout.ordered_locations) > 1
    for init in model.graph.initializer:
        if not init.HasField("raw_data"):
            continue
        if multi and init.name in layout.init_to_location:
            return True
        if not multi and len(init.raw_data) >= size_threshold:
            return True
    return False


def _save_onnx_external_aligned(
    model: onnx.ModelProto,
    onnx_path: str,
    layout: ExternalDataLayout | None,
    *,
    size_threshold: int,
) -> bool:
    """Save ONNX; external weight filenames / multi-file split match ``layout`` (source).

    Returns:
        True if only the ``.onnx`` proto was written and existing external data files
        were left unchanged (subsequent shape variants; fast path).
    """
    out_dir = os.path.dirname(onnx_path)
    os.makedirs(out_dir, exist_ok=True)

    if not _needs_external_blob_rebuild(model, size_threshold, layout):
        onnx.save(model, onnx_path, save_as_external_data=False)
        return True

    if layout is None or len(layout.ordered_locations) == 1:
        loc = (
            layout.ordered_locations[0] if layout else DEFAULT_EXTERNAL_WEIGHTS_FILENAME
        )
        ext_path = os.path.join(out_dir, loc)
        if os.path.isfile(ext_path):
            os.remove(ext_path)
        onnx.save(
            model,
            onnx_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=loc,
            size_threshold=size_threshold,
        )
        return False

    def _sort_key(init: onnx.TensorProto) -> tuple:
        fi, off = layout.init_pack_order[init.name]
        return (fi, off, init.name)

    # Pack exactly the tensors the source ONNX placed in each file (all sizes).
    # Do not use size_threshold here: shard 0 often mixes sub-1024B tensors with
    # large weights; omitting small tensors corrupts the first blob vs upstream.
    # Initializers not listed in init_to_location (e.g. script-added GQA scalars)
    # stay embedded in the .onnx — never append them into a shard.
    by_loc: dict[str, list[onnx.TensorProto]] = defaultdict(list)
    for init in model.graph.initializer:
        if init.name not in layout.init_to_location:
            continue
        if not init.HasField("raw_data"):
            continue
        loc = layout.init_to_location[init.name]
        by_loc[loc].append(init)

    for loc in layout.ordered_locations:
        lst = sorted(by_loc.get(loc, []), key=_sort_key)
        if not lst:
            continue
        blob = b"".join(init.raw_data for init in lst)
        fpath = os.path.join(out_dir, loc)
        with open(fpath, "wb") as f:
            f.write(blob)
        off = 0
        for init in lst:
            chunk = init.raw_data
            ln = len(chunk)
            set_external_data(init, loc, off, ln)
            off += ln
            init.ClearField("raw_data")
            init.data_location = TensorProto.EXTERNAL

    onnx.save(model, onnx_path, save_as_external_data=False)
    return False


def variant_filename_tag(chunk: int, max_length: int) -> str:
    """Stem segment ``p{chunk}m{max_length}`` (``m`` = max length label, no underscore)."""
    return f"p{chunk}m{max_length}"


# Besides sliding-window variants, also export numeric-stem fixed-prompt variants;
# 12200 exists only in this fixed set (no p12200m sliding variant).
FIXED_PROMPT_SEQUENCE_LENS: frozenset[int] = frozenset((128, 2048, 12200))


def fixed_prompt_filename_stem(sequence_len: int) -> str:
    """Stem for chunk=0 ONNX/genai filenames. ``2048`` → ``2k`` (shorter path segment)."""
    s = int(sequence_len)
    if s == 2048:
        return "2k"
    return str(s)


def expand_export_variant_specs(
    seq_lens: tuple[int, ...],
    max_length: int,
    *,
    decode_mode: bool,
) -> list[tuple[str, dict[str, int], int, int | None, int | None]]:
    """Expand a sequence-length tuple into the ONNX/genai variant list.

    Each item is
    ``(file_stem, dim_map, genai_window_chunk, fixed_prompt_len, genai_context_length)``:

    - ``genai_window_chunk==0``: fixed-prompt JSON variant (writes ``fixed_prompt_length``);
      ``fixed_prompt_len`` is the corresponding sequence length.
    - ``genai_window_chunk>0``: sliding-window variant, ``window_size`` uses this value;
      ``fixed_prompt_len`` is ``None``.
    - ``genai_context_length``: if not ``None``, JSON ``context_length`` /
      ``search.max_length`` use this value (ONNX KV / mask dims follow ``dim_map``); if
      ``None``, use CLI ``--max-length``.
    - ``12200`` only generates a fixed variant, no ``p12200m...`` sliding variant.
    - For fixed variants where ``S==2048``, stem is ``2k`` (e.g. ``prefill_2k.onnx``),
      while dimensions still use 2048.
    """
    ml = int(max_length)
    out: list[tuple[str, dict[str, int], int, int | None, int | None]] = []
    for s_raw in seq_lens:
        s = int(s_raw)
        if s != 12200:
            tag_sl = variant_filename_tag(s, ml)
            if decode_mode:
                dm_sl = {
                    "sequence_length": 1,
                    "past_sequence_length": ml,
                    "total_sequence_length": ml,
                }
            else:
                dm_sl = {
                    "sequence_length": s,
                    "past_sequence_length": ml,
                    "total_sequence_length": ml,
                }
            out.append((tag_sl, dm_sl, s, None, None))
        if s in FIXED_PROMPT_SEQUENCE_LENS:
            tag_fx = fixed_prompt_filename_stem(s)
            if decode_mode:
                dm_fx = {
                    "sequence_length": 1,
                    "past_sequence_length": ml,
                    "total_sequence_length": ml,
                }
            else:
                dm_fx = {
                    "sequence_length": s,
                    "past_sequence_length": ml,
                    "total_sequence_length": ml,
                }
            out.append((tag_fx, dm_fx, 0, s, None))
            if s == 128:
                ctrl_ml = 256
                tag_ctrl = "128_m256"
                if decode_mode:
                    dm_ctrl = {
                        "sequence_length": 1,
                        "past_sequence_length": ctrl_ml,
                        "total_sequence_length": ctrl_ml,
                    }
                else:
                    dm_ctrl = {
                        "sequence_length": 128,
                        "past_sequence_length": ctrl_ml,
                        "total_sequence_length": ctrl_ml,
                    }
                out.append((tag_ctrl, dm_ctrl, 0, 128, ctrl_ml))
            if s == 2048:
                ctrl_ml = 3072
                tag_ctrl = "2k_m3072"
                if decode_mode:
                    dm_ctrl = {
                        "sequence_length": 1,
                        "past_sequence_length": ctrl_ml,
                        "total_sequence_length": ctrl_ml,
                    }
                else:
                    dm_ctrl = {
                        "sequence_length": 2048,
                        "past_sequence_length": ctrl_ml,
                        "total_sequence_length": ctrl_ml,
                    }
                out.append((tag_ctrl, dm_ctrl, 0, 2048, ctrl_ml))
    return out


# MoE quantized op type; if present, do not generate parallel DML profiles
# (not meaningful to combine with MorphiZen-like EP configurations).
QMoe_OP_TYPES: frozenset[str] = frozenset({"QMoE"})


def _graph_contains_any_op_type(
    graph: onnx.GraphProto, op_types: frozenset[str]
) -> bool:
    return any(node.op_type in op_types for node in graph.node)


def onnx_model_contains_qmoe(model_path: str) -> bool:
    """Load ONNX without external weights; True if top-level graph contains ``QMoE``."""
    if not model_path or not os.path.isfile(model_path):
        return False
    m = onnx.load(model_path, load_external_data=False)
    return _graph_contains_any_op_type(m.graph, QMoe_OP_TYPES)


def _emit_genai_dml_sidecars(
    *,
    prefill_onnx_path: str | None,
    decode_onnx_path: str | None,
) -> bool:
    """True iff we should write ``*_dml.json`` for fixed-prompt variants (no QMoE in graphs)."""
    if not prefill_onnx_path or not os.path.isfile(prefill_onnx_path):
        return False
    if onnx_model_contains_qmoe(prefill_onnx_path):
        return False
    if decode_onnx_path and os.path.isfile(decode_onnx_path):
        if onnx_model_contains_qmoe(decode_onnx_path):
            return False
    return True


def _decoder_session_options_dml() -> dict[str, Any]:
    """Match ``genai_config_*_dml.json`` next to ONNX (DirectML only)."""
    return {
        "provider_options": [
            {"dml": {"device_id": "0"}},
        ],
    }


def write_genai_config_jsons(
    output_dir: str,
    max_length: int,
    seq_lens: tuple[int, ...],
    template_path: str,
    *,
    prefill_onnx_path: str | None = None,
    decode_onnx_path: str | None = None,
) -> list[str]:
    """Write one ``genai_config_{tag}.json`` per variant into ``output_dir`` (``-o``)."""
    if not os.path.isfile(template_path):
        raise FileNotFoundError(f"genai_config template not found: {template_path}")

    with open(template_path, encoding="utf-8") as f:
        template = json.load(f)

    written: list[str] = []
    emit_dml = _emit_genai_dml_sidecars(
        prefill_onnx_path=prefill_onnx_path,
        decode_onnx_path=decode_onnx_path,
    )
    specs = expand_export_variant_specs(seq_lens, max_length, decode_mode=False)
    for tag, _dm, genai_chunk, fixed_prompt_len, genai_ctx in specs:
        ctx_len = genai_ctx if genai_ctx is not None else max_length
        cfg = copy.deepcopy(template)
        model = cfg.setdefault("model", {})
        model["context_length"] = ctx_len
        dec = model.setdefault("decoder", {})

        if genai_chunk == 0:
            dec.pop("sliding_window", None)
        else:
            dec.pop("fixed_prompt_length", None)
            sw = dec.setdefault("sliding_window", {})
            sw["window_size"] = int(genai_chunk)

        dec["pipeline"]["prefill"]["filename"] = f"prefill_{tag}.onnx"
        dec["pipeline"]["decode"]["filename"] = f"decode_{tag}.onnx"
        cfg["search"]["max_length"] = ctx_len

        # Fixed-prompt variant: ORT GenAI requires ``decoder.fixed_prompt_length``.
        # The template may not contain this key, so write it last to avoid losing it
        # during deep-copy or subsequent edits.
        if genai_chunk == 0 and fixed_prompt_len is not None:
            dec["fixed_prompt_length"] = int(fixed_prompt_len)

        os.makedirs(output_dir, exist_ok=True)
        out_name = f"genai_config_{tag}.json"
        out_path = os.path.join(output_dir, out_name)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=4, ensure_ascii=False)
            f.write("\n")
        written.append(out_path)

        if (
            emit_dml
            and genai_chunk == 0
            and fixed_prompt_len is not None
            and int(fixed_prompt_len) in FIXED_PROMPT_SEQUENCE_LENS
        ):
            cfg_dml = copy.deepcopy(cfg)
            dec_dml = cfg_dml.setdefault("model", {}).setdefault("decoder", {})
            dec_dml["session_options"] = _decoder_session_options_dml()
            dml_name = f"genai_config_{tag}_dml.json"
            dml_path = os.path.join(output_dir, dml_name)
            with open(dml_path, "w", encoding="utf-8") as f:
                json.dump(cfg_dml, f, indent=4, ensure_ascii=False)
                f.write("\n")
            written.append(dml_path)

    return written


INPUT_IDS_NAME = "input_ids"
ATTENTION_MASK_NAME = "attention_mask"
POSITION_IDS_NAME = "position_ids"

# ORT LLM: past_key_values.{i}.key / .value — axis 2 is past cache length
PAST_KV_INPUT_RE = re.compile(r"^past_key_values\.\d+\.(?:key|value)$")
# KV cache outputs: axis 2 must match max_length (concrete dims in source ONNX are not
# updated by _set_vi_shape_ctx, which only resolves dim_param).
PRESENT_IO_RE = re.compile(r"^present\.\d+\.(?:key|value)$")


def _force_rank2_input_shape(vi, d0: int, d1: int):
    """Set graph input ValueInfo to fixed rank-2 shape [d0, d1], keep elem_type."""
    if not vi.type.HasField("tensor_type"):
        return
    tt = vi.type.tensor_type
    del tt.shape.dim[:]
    for v in (d0, d1):
        dim = tt.shape.dim.add()
        dim.dim_value = v


def _apply_input_ids_mask_shapes(
    graph: onnx.GraphProto,
    input_ids_seq: int,
    attention_second: int,
):
    """input_ids [1, seq], attention_mask [1, max_length], position_ids [1, seq]."""
    for vi in graph.input:
        if vi.name == INPUT_IDS_NAME:
            _force_rank2_input_shape(vi, 1, input_ids_seq)
        elif vi.name == ATTENTION_MASK_NAME:
            _force_rank2_input_shape(vi, 1, attention_second)
        elif vi.name == POSITION_IDS_NAME:
            _force_rank2_input_shape(vi, 1, input_ids_seq)


def _force_past_key_values_past_seq_dim(
    graph: onnx.GraphProto,
    past_sequence_length: int,
) -> int:
    """Set past_sequence_length (dim index 2) for all past_key_values inputs."""
    n = 0
    for vi in graph.input:
        if not PAST_KV_INPUT_RE.match(vi.name):
            continue
        if not vi.type.HasField("tensor_type"):
            continue
        dims = vi.type.tensor_type.shape.dim
        if len(dims) != 4:
            continue
        dims[0].ClearField("dim_param")
        dims[0].dim_value = 1
        dims[2].ClearField("dim_param")
        dims[2].dim_value = past_sequence_length
        n += 1
    return n


def _force_present_kv_seq_dim(graph: onnx.GraphProto, seq_dim: int) -> int:
    """Set dim index 2 to ``seq_dim`` for all ``present.*.key/value`` graph outputs
    and matching ``value_info`` (layout [batch, kv_heads, seq, head_size])."""
    n = 0

    def _fix_vi(vi):
        nonlocal n
        if not PRESENT_IO_RE.match(vi.name):
            return
        if not vi.type.HasField("tensor_type"):
            return
        dims = vi.type.tensor_type.shape.dim
        if len(dims) != 4:
            return
        dims[2].ClearField("dim_param")
        dims[2].dim_value = seq_dim
        n += 1

    for vi in graph.output:
        _fix_vi(vi)
    for vi in graph.value_info:
        _fix_vi(vi)
    return n


def resolve_dim_param_ctx(param: str, dim_map) -> int | str:
    """Like extract_submodels.resolve_dim_param, without int-dim_map fallback
    that would wrongly substitute unknown symbols when dim_map is dict."""
    subs = _build_dim_subs(dim_map)

    has_op = any(c in param for c in ("*", "+", "-", "/", "(", ")"))
    if has_op:
        expr = param
        for sym, val in sorted(subs.items(), key=lambda x: -len(x[0])):
            expr = expr.replace(sym, str(val))
        try:
            return int(eval(expr))  # noqa: S307
        except Exception:
            return param

    if param in subs:
        return subs[param]

    if "batch" in param and "seq" not in param:
        return BATCH_SIZE

    if isinstance(dim_map, int):
        return dim_map

    return param


def _set_vi_shape_ctx(vi, dim_map):
    if not vi.type.tensor_type.shape:
        return
    for dim in vi.type.tensor_type.shape.dim:
        if dim.dim_param:
            val = resolve_dim_param_ctx(dim.dim_param, dim_map)
            if isinstance(val, int):
                dim.ClearField("dim_param")
                dim.dim_value = val


def _sanitize(s: str, max_len: int = 64) -> str:
    t = re.sub(r"[^a-zA-Z0-9_]", "_", s)
    return t[:max_len] if t else "gqa"


def _backup_gqa_nodes(model: onnx.ModelProto):
    """(node_name, full input list copy) for each GroupQueryAttention.

    Must use **names**, not node list indices: after ``_remove_orphan_fold_total_seq_len``
    deletes ``fold/total_seq_len``, all following indices shift; restoring by stale
    index would overwrite the wrong node (e.g. o_proj MatMul gets GQA's 9 inputs)."""
    return [
        (n.name, list(n.input))
        for n in model.graph.node
        if n.op_type == "GroupQueryAttention"
    ]


def _restore_gqa_nodes(model: onnx.ModelProto, backup):
    by_name = dict(backup)
    for node in model.graph.node:
        if node.name in by_name:
            del node.input[:]
            node.input.extend(by_name[node.name])


def _remove_initializer_prefix(graph: onnx.GraphProto, prefix: str):
    keep = [init for init in graph.initializer if not init.name.startswith(prefix)]
    del graph.initializer[:]
    graph.initializer.extend(keep)


def _tensor_name_producer_index(graph: onnx.GraphProto) -> dict[str, int]:
    """Map tensor name -> index of the node that defines it (last writer wins)."""
    out: dict[str, int] = {}
    for i, node in enumerate(graph.node):
        for o in node.output:
            if o:
                out[o] = i
    return out


def _backward_needed_tensor_names(graph: onnx.GraphProto) -> set[str]:
    """Tensor names that lie on some path from graph outputs backward through producers."""
    producers = _tensor_name_producer_index(graph)
    needed = {o.name for o in graph.output if o.name}
    changed = True
    while changed:
        changed = False
        for t in list(needed):
            if t not in producers:
                continue
            node = graph.node[producers[t]]
            for inp in node.input:
                if inp and inp not in needed:
                    needed.add(inp)
                    changed = True
    return needed


def _remove_dead_nodes_one_pass(graph: onnx.GraphProto) -> int:
    """Remove nodes none of whose outputs are in ``backward_needed``."""
    needed = _backward_needed_tensor_names(graph)
    kept: list[onnx.NodeProto] = []
    removed = 0
    for node in graph.node:
        outs = [o for o in node.output if o]
        if not outs:
            kept.append(node)
            continue
        if any(o in needed for o in outs):
            kept.append(node)
        else:
            removed += 1
    if removed:
        del graph.node[:]
        graph.node.extend(kept)
    return removed


def _dce_remove_unreachable_nodes(graph: onnx.GraphProto) -> int:
    """Iterative dead-code elimination until fixpoint."""
    total = 0
    while True:
        n = _remove_dead_nodes_one_pass(graph)
        total += n
        if n == 0:
            break
    return total


def _graph_live_tensor_names(graph: onnx.GraphProto) -> set[str]:
    names = {i.name for i in graph.input if i.name}
    names |= {o.name for o in graph.output if o.name}
    names |= {t.name for t in graph.initializer}
    for n in graph.node:
        for t in n.input:
            if t:
                names.add(t)
        for t in n.output:
            if t:
                names.add(t)
    return names


def _trim_value_info_to_live_tensors(graph: onnx.GraphProto) -> int:
    """Drop value_info for tensors that no longer appear in the graph."""
    live = _graph_live_tensor_names(graph)
    keep = [vi for vi in graph.value_info if vi.name in live]
    before = len(graph.value_info)
    if len(keep) != before:
        del graph.value_info[:]
        graph.value_info.extend(keep)
    return before - len(keep)


def _remove_unused_graph_inputs(graph: onnx.GraphProto, needed: set[str]) -> int:
    """Remove graph inputs not consumed on any path to a graph output (keep I/O aliases)."""
    out_names = {o.name for o in graph.output if o.name}
    drop_idx = []
    for idx, inp in enumerate(graph.input):
        name = inp.name
        if not name or name in out_names:
            continue
        if name not in needed:
            drop_idx.append(idx)
    for idx in reversed(drop_idx):
        del graph.input[idx]
    return len(drop_idx)


def _remove_unused_initializers(graph: onnx.GraphProto, needed: set[str]) -> int:
    """Remove initializers whose names never appear in ``needed``."""
    keep = [init for init in graph.initializer if init.name in needed]
    removed = len(graph.initializer) - len(keep)
    if removed:
        del graph.initializer[:]
        graph.initializer.extend(keep)
    return removed


def _post_export_prune_unreachable(model: onnx.ModelProto) -> tuple[int, int, int, int]:
    """DCE + trim value_info + unused inputs + unused initializers.

    Returns ``(nodes_removed, value_info_removed, inputs_removed, inits_removed)``.
    """
    graph = model.graph
    n_nodes = _dce_remove_unreachable_nodes(graph)
    n_vi = _trim_value_info_to_live_tensors(graph)
    needed = _backward_needed_tensor_names(graph)
    n_in = _remove_unused_graph_inputs(graph, needed)
    n_init = _remove_unused_initializers(graph, needed)
    return n_nodes, n_vi, n_in, n_init


def _is_gqa_attn_mask_reformat_input(inp: str) -> bool:
    """True if this GQA input is the shared attn_mask_reformat Cast output — must not replace."""
    if not inp:
        return False
    if "attn_mask_reformat" in inp:
        return True
    if "attn_mask_subgraph" in inp and "Cast" in inp:
        return True
    return False


def _patch_gqa_scalars(
    model: onnx.ModelProto,
    dim_map: dict,
    name_prefix: str,
) -> int:
    """Optionally wire GroupQueryAttention total_sequence_length (input 6).

    Llama-3.1 exports use input 5 for ``attn_mask_reformat/.../Cast/output_0``;
    that edge must stay unchanged. Input 6 (e.g. ``total_seq_len_constant``)
    is replaced with a scalar initializer = total_sequence_length from dim_map.

    On graphs where input 5 is not the mask path (legacy seqlens_k), input 5
    may still be patched with [sequence_length-1]."""
    subs = _build_dim_subs(dim_map)
    seq_len = int(subs["sequence_length"])
    total_len = int(subs["total_sequence_length"])

    patched = 0
    for ni, node in enumerate(model.graph.node):
        if node.op_type != "GroupQueryAttention":
            continue
        if len(node.input) <= 6:
            continue
        suf = _sanitize(node.name or f"nd{ni}")
        sl = f"{name_prefix}n{ni}_{suf}_seqlens_k"
        tl = f"{name_prefix}n{ni}_{suf}_total_seq"

        new_inits = []

        inp5 = node.input[5] if len(node.input) > 5 else ""
        if inp5 and not _is_gqa_attn_mask_reformat_input(inp5):
            node.input[5] = sl
            new_inits.append(
                numpy_helper.from_array(
                    np.array([seq_len - 1], dtype=np.int32),
                    name=sl,
                )
            )
        if len(node.input) > 6 and node.input[6]:
            node.input[6] = tl
            new_inits.append(
                numpy_helper.from_array(
                    np.array(total_len, dtype=np.int32),
                    name=tl,
                )
            )

        for it in new_inits:
            model.graph.initializer.append(it)
            patched += 1
    return patched


def _snapshot_fold_total_seq_len_node(model: onnx.ModelProto):
    """Copy of the original Constant that produces total_seq_len_constant (if present)."""
    for n in model.graph.node:
        if n.name == FOLD_TOTAL_SEQ_NODE_NAME and n.op_type == "Constant":
            t = onnx.NodeProto()
            t.CopyFrom(n)
            return t
    return None


def _ensure_fold_total_seq_len_node(model: onnx.ModelProto, template):
    """Re-insert Constant before each variant if it was removed in the previous save."""
    if template is None:
        return
    if any(n.name == FOLD_TOTAL_SEQ_NODE_NAME for n in model.graph.node):
        return
    nc = onnx.NodeProto()
    nc.CopyFrom(template)
    model.graph.node.append(nc)


def _tensor_refcount(graph: onnx.GraphProto, tensor_name: str) -> int:
    n = 0
    for node in graph.node:
        for inp in node.input:
            if inp == tensor_name:
                n += 1
    for out in graph.output:
        if out.name == tensor_name:
            n += 1
    return n


def _remove_orphan_fold_total_seq_len(model: onnx.ModelProto) -> bool:
    """After GQA input 6 is rewired off total_seq_len_constant, remove dead Constant.

    Original graphs use ``Constant fold/total_seq_len`` → ``total_seq_len_constant``.
    Our GQA patch replaces input 6 with ``gqa_ctxfix_*_total_seq``, so the old
    Constant becomes redundant — remove it from exported models to avoid confusion."""
    if _tensor_refcount(model.graph, TOTAL_SEQ_LEN_CONSTANT_OUT) > 0:
        return False
    kept = [n for n in model.graph.node if n.name != FOLD_TOTAL_SEQ_NODE_NAME]
    if len(kept) == len(model.graph.node):
        return False
    del model.graph.node[:]
    model.graph.node.extend(kept)
    # Drop stale value_info for the removed edge
    vi_keep = [
        v for v in model.graph.value_info if v.name != TOTAL_SEQ_LEN_CONSTANT_OUT
    ]
    if len(vi_keep) != len(model.graph.value_info):
        del model.graph.value_info[:]
        model.graph.value_info.extend(vi_keep)
    return True


class FixedCtxExtractor(SingleOpExtractor):
    """``prefill_{stem}.onnx`` / ``decode_{stem}.onnx`` — stem is ``pCHUNKmMAX`` or ``S`` (chunk=0)."""

    def __init__(
        self,
        model_path: str,
        output_dir: str | None = None,
        *,
        max_length: int = 16384,
        seq_lens: tuple[int, ...] | None = None,
        decode_mode: bool = False,
        export_prefix: str = "prefill",
    ):
        self._max_length = max_length
        self._seq_lens = (
            seq_lens if seq_lens is not None else DEFAULT_QUICK_EXPORT_SEQ_LENS
        )
        self._decode_mode = decode_mode
        self._export_prefix = export_prefix
        super().__init__(model_path, output_dir=output_dir)
        # Write ONNX (and shared external data) directly under ``-o``, not ``.../full_model/``.
        self.full_model_dir = self.base_dir
        self._ext_layout = scan_external_data_layout(model_path)

    def _full_model_variant_onnx_basename(self, vname: str) -> str:
        """Layered ``extract_full_model`` path must match flat exports: only prefill_/decode_."""
        return f"{self._export_prefix}_{vname}.onnx"

    def _get_variants(self):
        ml = self._max_length
        specs = expand_export_variant_specs(
            self._seq_lens, ml, decode_mode=self._decode_mode
        )
        return [(tag, dm) for tag, dm, _, _, _ in specs]

    def _save_variants(self, model, out_dir, _prefix):
        """``_prefix`` from parent (``full_model``) ignored; files use ``_export_prefix``."""
        os.makedirs(out_dir, exist_ok=True)
        for fn in _stale_external_filenames(self._ext_layout):
            p = os.path.join(out_dir, fn)
            if os.path.isfile(p):
                os.remove(p)

        if self._ext_layout:
            nloc = len(self._ext_layout.ordered_locations)
            ex0 = self._ext_layout.ordered_locations[0]
            print(f"  External data: {nloc} file(s) (from source), e.g. {ex0!r}")
        else:
            print(
                f"  External data: single file {DEFAULT_EXTERNAL_WEIGHTS_FILENAME!r} "
                f"(source has no external initializers)"
            )

        orig_in = {i.name: self._save_vi_shape(i) for i in model.graph.input}
        orig_out = {o.name: self._save_vi_shape(o) for o in model.graph.output}
        orig_vi = {v.name: self._save_vi_shape(v) for v in model.graph.value_info}

        gqa_backup = _backup_gqa_nodes(model)
        gqa_node_count = sum(
            1 for n in model.graph.node if n.op_type == "GroupQueryAttention"
        )
        if gqa_node_count:
            print(
                f"  GQA nodes: {gqa_node_count} (will patch seqlens/total per variant)"
            )

        fold_total_template = _snapshot_fold_total_seq_len_node(model)
        _printed_reuse_hint = False

        for idx, (vname, dim_map) in enumerate(self.variants):
            if isinstance(dim_map, dict):
                cache_len = int(dim_map["total_sequence_length"])
            else:
                cache_len = self._max_length
            eff_total_llm = (
                total_seq_len_scalar_from_dim_map(dim_map)
                if self.model_type == "llm"
                else None
            )
            _ensure_fold_total_seq_len_node(model, fold_total_template)
            if eff_total_llm is not None:
                patch_fold_total_seq_len_constant_value(model, eff_total_llm)
            _remove_initializer_prefix(model.graph, GQA_INIT_PREFIX)
            _restore_gqa_nodes(model, gqa_backup)

            if eff_total_llm is not None:
                upsert_total_seq_len_const_initializer(model.graph, eff_total_llm)

            n_gqa_inits = _patch_gqa_scalars(model, dim_map, GQA_INIT_PREFIX)
            if n_gqa_inits and idx == 0:
                print(f"  Patched GQA initializer tensor(s): {n_gqa_inits}")

            if _remove_orphan_fold_total_seq_len(model) and idx == 0:
                print(
                    f"  Removed orphan {FOLD_TOTAL_SEQ_NODE_NAME!r} "
                    f"(output {TOTAL_SEQ_LEN_CONSTANT_OUT!r} unused after GQA total patch)"
                )

            for vi in (
                list(model.graph.input)
                + list(model.graph.output)
                + list(model.graph.value_info)
            ):
                _set_vi_shape_ctx(vi, dim_map)

            ids_seq = 1 if self._decode_mode else int(dim_map["sequence_length"])
            _apply_input_ids_mask_shapes(
                model.graph,
                input_ids_seq=ids_seq,
                attention_second=cache_len,
            )

            n_pk = _force_past_key_values_past_seq_dim(model.graph, cache_len)
            if n_pk and idx == 0:
                print(
                    f"  past_key_values inputs with past_sequence_length="
                    f"{cache_len} (variant cache): {n_pk}"
                )

            n_pr = _force_present_kv_seq_dim(model.graph, cache_len)
            if n_pr and idx == 0:
                print(
                    f"  present.* outputs/value_info dim[2]="
                    f"{cache_len} (variant cache): {n_pr} tensor(s)"
                )

            n_nodes, n_vi, n_in, n_init = _post_export_prune_unreachable(model)
            if idx == 0 and (n_nodes or n_vi or n_in or n_init):
                print(
                    f"  Pruned unreachable: {n_nodes} node(s), {n_vi} value_info, "
                    f"{n_in} unused input(s), {n_init} unused initializer(s)"
                )

            fname = f"{self._export_prefix}_{vname}.onnx"
            path = os.path.join(out_dir, fname)
            # Always use onnx.save (not raw SerializeToString). For multi-variant
            # exports with external weights, SerializeToString() produced invalid
            # graphs on idx>=1 (e.g. MatMul with 9 inputs / ORT schema errors) while
            # idx==0 loaded fine.
            graph_only = _save_onnx_external_aligned(
                model, path, self._ext_layout, size_threshold=1024
            )
            if graph_only and not _printed_reuse_hint:
                print(
                    "  Reusing shared external weight file(s); later variants write "
                    ".onnx only (no re-read / no blob rewrite)."
                )
                _printed_reuse_hint = True
            print(f"  Saved: {path}")

            for vi in model.graph.input:
                self._restore_vi_shape(vi, orig_in.get(vi.name))
            for vi in model.graph.output:
                self._restore_vi_shape(vi, orig_out.get(vi.name))
            for vi in model.graph.value_info:
                self._restore_vi_shape(vi, orig_vi.get(vi.name))

        _remove_initializer_prefix(model.graph, GQA_INIT_PREFIX)
        _restore_gqa_nodes(model, gqa_backup)


def run_export(
    model_path: str,
    output_dir: str,
    max_length: int,
    seq_lens: tuple[int, ...],
    *,
    decode_mode: bool = False,
    export_prefix: str = "prefill",
):
    ext = FixedCtxExtractor(
        model_path,
        output_dir=output_dir,
        max_length=max_length,
        seq_lens=seq_lens,
        decode_mode=decode_mode,
        export_prefix=export_prefix,
    )
    ext.extract_full_model()


def main():
    p = argparse.ArgumentParser(
        description=(
            "Export Llama prefill/decode ONNX with fixed max_length "
            f"(default {16384}). By default exports only the "
            f"p512m{{max_length}} pair plus genai_config; pass --all for the full "
            f"variant matrix ({', '.join(map(str, DEFAULT_SEQ_LENS))} and expanded stems)."
        )
    )
    p.add_argument(
        "--model",
        required=True,
        help="Path to source ONNX used for both prefill and decode exports.",
    )
    p.add_argument(
        "-o",
        "--output",
        help="Output directory: ONNX and genai_config_*.json are written here (no subfolder).",
    )
    p.add_argument(
        "-T",
        "--config-template",
        "--genai-config-template",
        default=None,
        metavar="PATH",
        dest="config_template",
        help=(
            "Template used to generate genai_config_*.json (choose file per model family). "
            f"Required unless --no-genai-config is set. Example (Llama template in this repo): {DEFAULT_GENAI_CONFIG_TEMPLATE}"
        ),
    )
    p.add_argument(
        "--no-genai-config",
        action="store_true",
        help="Do not write genai_config_*.json files.",
    )
    p.add_argument(
        "--max-length",
        "--cache-len",
        type=int,
        default=16384,
        dest="max_length",
        metavar="N",
        help="max_length (e.g. 16384): past/total sequence dims, GQA total, attention_mask "
        "second dim. --cache-len is an alias.",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help=(
            "Export the full variant matrix (sequence lengths "
            f"{', '.join(map(str, DEFAULT_SEQ_LENS))} and all derived fixed-prompt / "
            "control stems). Default is a single sliding variant p512m{max_length} only."
        ),
    )

    args = p.parse_args()
    if not args.output:
        p.error("--output / -o is required")

    write_genai = not args.no_genai_config
    if write_genai:
        if not args.config_template:
            p.error(
                "A template path is required when generating genai_config: "
                "-T / --config-template / --genai-config-template PATH "
                f"(example: {DEFAULT_GENAI_CONFIG_TEMPLATE}), "
                "or pass --no-genai-config to skip generation."
            )
        cfg_tpl = os.path.abspath(args.config_template)
        if not os.path.isfile(cfg_tpl):
            p.error(f"GenAI config template file does not exist: {cfg_tpl}")
    else:
        cfg_tpl = (
            os.path.abspath(args.config_template) if args.config_template else None
        )

    seq_lens = DEFAULT_SEQ_LENS if args.all else DEFAULT_QUICK_EXPORT_SEQ_LENS
    out = os.path.abspath(args.output)

    model_abs = os.path.abspath(args.model)
    if not os.path.isfile(model_abs):
        p.error(f"Model file does not exist: {model_abs}")

    mode = "full matrix (--all)" if args.all else "quick (p512 only)"
    print(f"=== Prefill ({mode}, out={out}, max_length={args.max_length}) ===")
    run_export(
        model_abs,
        out,
        args.max_length,
        seq_lens,
        decode_mode=False,
        export_prefix="prefill",
    )
    print(f"=== Decode ({mode}, out={out}, max_length={args.max_length}) ===")
    run_export(
        model_abs,
        out,
        args.max_length,
        seq_lens,
        decode_mode=True,
        export_prefix="decode",
    )

    if write_genai:
        print(f"=== genai_config_*.json → {out}/ ===")
        print(f"  Template: {cfg_tpl}")
        paths = write_genai_config_jsons(
            out,
            args.max_length,
            seq_lens,
            cfg_tpl,
            prefill_onnx_path=model_abs,
            decode_onnx_path=model_abs,
        )
        for q in paths:
            print(f"  Wrote: {q}")

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
