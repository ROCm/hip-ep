#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""
Read structural metadata from `genai_config.json` (or another JSON) in a model folder,
then generate a `decoder-pipeline` style `genai_config.json` for ORT GenAI.

Typical usage (folder contains a single-model `model.onnx` config; prepare prefill/decode ONNX files separately)::

    python genai_config_pipeline_from_folder.py D:\\path\\Llama-3.1-8B-awq-g128-int4-onnx-directml \\
        --fixed-prompt-length 128 --max-length 4096

Default output: ``<model_dir>/genai_config_pipeline.json`` (override with ``-o``).

`decoder.session_options` follows the same shape as
`Llama-3.1-8B/genai_config_12200.json`:
`session.disable_cpu_ep_fallback` is forced to ``"1"``,
and `provider_options` is forced to ``[{"MorphiZenEP": {}}]``.
Optional `--session-options-json` only merges other keys and never overrides those two.

For ``decoder.inputs`` / ``decoder.outputs``: **keys absent in the source are not
back-filled** from Llama-style defaults (e.g. gptoss without ``position_ids`` stays
without it; partial ``outputs`` stays partial). ``pipeline.inputs`` order follows only
bindings present: ``input_ids`` or ``inputs_embeds``, then optional ``attention_mask`` /
``position_ids``, then ``past_*`` (``past_key_names`` / ``past_value_names`` are still
required when auto-building lists). If ``decoder.inputs`` or ``decoder.outputs`` is
missing or empty (e.g. bare HF ``config.json``), the full default maps are used for that
side so the tool still produces a usable template.
Fully custom main-input names with neither pattern above must provide an ONNX input name
string array in ``model.decoder.pipeline.prefill.inputs``.

When ``model_dir`` is passed to the builder (CLI default), if ``prefill_*.onnx`` and
``decode_*.onnx`` are found (search: ``model_dir``, then ``chunk/``, ``full_model/``,
``export_fixed_ctx/full_model/``, plus any ``--onnx-subdir`` prefixes), **I/O name lists
are read from the ONNX graphs** so sparse KV / extra state tensors / binding order match
the exported model (e.g. Qwen3.5). If ONNX is missing or ``onnx`` is not installed, the
script falls back to template lists derived from ``decoder.inputs``. Use ``--no-onnx-io``
to force templates only.

The output is a **template**: ``decoder`` always includes both
``fixed_prompt_length`` and ``sliding_window``
(``window_size`` equals ``fixed_prompt_length`` and contains ``alignment: "left"``),
so it can be adjusted manually afterward.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from typing import Any

# Keep chunk==0 naming consistent with export_llama_fixed_ctx.py variants.
CHUNK_ZERO_STEM = "12200"

# Relative to model_dir when resolving prefill/decode ONNX paths (first existing file wins).
_DEFAULT_ONNX_SEARCH_SUBDIRS: tuple[str, ...] = (
    ".",
    "chunk",
    "full_model",
    os.path.join("export_fixed_ctx", "full_model"),
)


def _find_onnx_path(
    model_dir: str,
    onnx_basename: str,
    search_subdirs: tuple[str, ...],
) -> str | None:
    """Return absolute path to ``onnx_basename`` under ``model_dir`` / subdirs, or None."""
    root = os.path.abspath(model_dir)
    raw = str(onnx_basename).strip()
    base = os.path.basename(raw)
    if not base:
        return None
    if os.path.isabs(raw) and os.path.isfile(raw):
        return raw
    if base != os.path.basename(os.path.normpath(raw)):
        return None
    for sub in search_subdirs:
        sub_path = os.path.normpath(os.path.join(root, sub))
        try:
            if os.path.commonpath([root, sub_path]) != root:
                continue
        except ValueError:
            continue
        cand = os.path.join(sub_path, base)
        if os.path.isfile(cand):
            return cand
    return None


def _onnx_graph_io_lists(onnx_path: str) -> tuple[list[str], list[str]]:
    """Graph input and output tensor names in ONNX order (``load_external_data=False``)."""
    try:
        import onnx
    except ImportError as e:
        raise ImportError(
            "The 'onnx' package is required to read I/O names from ONNX files "
            "(pip install onnx), or pass --no-onnx-io to use template lists only."
        ) from e

    model = onnx.load(onnx_path, load_external_data=False)
    g = model.graph
    return [vi.name for vi in g.input], [vi.name for vi in g.output]


def _default_stem(fixed_prompt: int, max_length: int) -> str:
    if fixed_prompt == 12200:
        return CHUNK_ZERO_STEM
    return f"p{fixed_prompt}m{max_length}"


def _canonical_morphizen_session_options() -> dict[str, Any]:
    """Match ``decoder.session_options`` from ``genai_config_12200.json``."""
    return {
        "session.disable_cpu_ep_fallback": "1",
        "provider_options": [{"MorphiZenEP": {}}],
    }


# ORT GenAI decoder.inputs / decoder.outputs naming templates
# (aligned with Llama genai_config files in this repo).
_DEFAULT_DECODER_INPUTS: dict[str, str] = {
    "input_ids": "input_ids",
    "attention_mask": "attention_mask",
    "position_ids": "position_ids",
    "past_key_names": "past_key_values.%d.key",
    "past_value_names": "past_key_values.%d.value",
}
_DEFAULT_DECODER_OUTPUTS: dict[str, str] = {
    "logits": "logits",
    "present_key_names": "present.%d.key",
    "present_value_names": "present.%d.value",
}

# decoder.inputs layout kinds (for validation and pipeline.inputs auto-ordering).
INPUT_LAYOUT_FULL_IDS = "full_ids"
INPUT_LAYOUT_EMBEDS = "embeds"
INPUT_LAYOUT_CUSTOM = "custom"


def _merge_decoder_io_templates(
    inputs: dict[str, Any] | None,
    outputs: dict[str, Any] | None,
) -> tuple[dict[str, str], dict[str, str], str]:
    """Merge ``decoder.inputs`` / ``outputs`` templates.

    Source ``inputs`` / ``outputs`` dicts: keys are **not** back-filled from defaults.
    If ``inputs`` (or ``outputs``) is missing, non-dict, or has no non-empty values after
    filtering, the full default map is used for that side only.

    Returns:
        ``(merged_inputs, merged_outputs, input_layout_kind)``, where
        ``input_layout_kind`` is ``full_ids`` | ``embeds`` | ``custom``.
    """
    merged_out: dict[str, str] = {}
    if isinstance(outputs, dict):
        for k, v in outputs.items():
            if v is not None and str(v).strip() != "":
                merged_out[str(k)] = str(v)
    if not merged_out:
        merged_out = dict(_DEFAULT_DECODER_OUTPUTS)

    if inputs is None:
        return dict(_DEFAULT_DECODER_INPUTS), merged_out, INPUT_LAYOUT_FULL_IDS

    if not isinstance(inputs, dict):
        return dict(_DEFAULT_DECODER_INPUTS), merged_out, INPUT_LAYOUT_FULL_IDS

    merged_in: dict[str, str] = {}
    for k, v in inputs.items():
        if v is not None and str(v).strip() != "":
            merged_in[str(k)] = str(v)

    if not merged_in:
        return dict(_DEFAULT_DECODER_INPUTS), merged_out, INPUT_LAYOUT_FULL_IDS

    if merged_in.get("inputs_embeds"):
        return merged_in, merged_out, INPUT_LAYOUT_EMBEDS

    if merged_in.get("input_ids"):
        return merged_in, merged_out, INPUT_LAYOUT_FULL_IDS

    return merged_in, merged_out, INPUT_LAYOUT_CUSTOM


def _extract_pipeline_io_lists_from_decoder(
    dec: dict[str, Any],
) -> tuple[list[str] | None, list[str] | None, list[str] | None, list[str] | None]:
    """Read prefill/decode input/output lists from ``decoder.pipeline`` when present."""
    pipe = dec.get("pipeline")
    if not isinstance(pipe, dict):
        return None, None, None, None
    pre = pipe.get("prefill")
    dblk = pipe.get("decode")
    pin = por = din = dor = None
    if isinstance(pre, dict):
        if isinstance(pre.get("inputs"), list) and pre["inputs"]:
            pin = [str(x) for x in pre["inputs"]]
        if isinstance(pre.get("outputs"), list) and pre["outputs"]:
            por = [str(x) for x in pre["outputs"]]
    if isinstance(dblk, dict):
        if isinstance(dblk.get("inputs"), list) and dblk["inputs"]:
            din = [str(x) for x in dblk["inputs"]]
        if isinstance(dblk.get("outputs"), list) and dblk["outputs"]:
            dor = [str(x) for x in dblk["outputs"]]
    return pin, por, din, dor


def _pipeline_io_lists(num_hidden_layers: int) -> tuple[list[str], list[str]]:
    """Prefill/decode I/O name lists matching existing Llama pipeline configs."""
    return (
        _pipeline_inputs_from_decoder_templates(
            dict(_DEFAULT_DECODER_INPUTS), num_hidden_layers
        ),
        _pipeline_outputs_from_layer_count(num_hidden_layers),
    )


def _pipeline_outputs_from_layer_count(num_hidden_layers: int) -> list[str]:
    outputs: list[str] = ["logits"]
    for i in range(num_hidden_layers):
        outputs.append(f"present.{i}.key")
        outputs.append(f"present.{i}.value")
    return outputs


def _pipeline_inputs_from_decoder_templates(
    inputs_tmpl: dict[str, str],
    num_hidden_layers: int,
) -> list[str]:
    """Generate ONNX input-name order from ``decoder.inputs`` bindings (embeds-first for Qwen/Gemma)."""
    pin: list[str] = []
    if inputs_tmpl.get("inputs_embeds"):
        pin.append(inputs_tmpl["inputs_embeds"])
    elif inputs_tmpl.get("input_ids"):
        pin.append(inputs_tmpl["input_ids"])
    for k in ("attention_mask", "position_ids"):
        v = inputs_tmpl.get(k)
        if v:
            pin.append(v)
    pk = inputs_tmpl.get("past_key_names") or _DEFAULT_DECODER_INPUTS["past_key_names"]
    pv = (
        inputs_tmpl.get("past_value_names")
        or _DEFAULT_DECODER_INPUTS["past_value_names"]
    )
    for i in range(num_hidden_layers):
        pin.append(pk % i)
        pin.append(pv % i)
    return pin


def _load_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return data


def _find_config_file(model_dir: str, config_name: str | None) -> str:
    if config_name:
        p = os.path.join(model_dir, config_name)
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Specified config file not found: {p}")
        return p
    candidates = ["genai_config.json", "config.json"]
    for name in candidates:
        p = os.path.join(model_dir, name)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        f"Neither genai_config.json nor config.json was found in directory: {model_dir}"
    )


def _extract_decoder_fields(src: dict[str, Any]) -> dict[str, Any]:
    """Extract decoder/architecture fields from GenAI or HuggingFace-style configs."""
    model = src.get("model")
    if isinstance(model, dict) and isinstance(model.get("decoder"), dict):
        dec = model["decoder"]
        pin0, por0, din0, dor0 = _extract_pipeline_io_lists_from_decoder(dec)
        raw_in = dec.get("inputs")
        raw_out = dec.get("outputs")
        merged_in, merged_out, layout_kind = _merge_decoder_io_templates(
            raw_in if isinstance(raw_in, dict) else None,
            raw_out if isinstance(raw_out, dict) else None,
        )
        out = {
            "head_size": dec.get("head_size"),
            "hidden_size": dec.get("hidden_size"),
            "num_attention_heads": dec.get("num_attention_heads"),
            "num_key_value_heads": dec.get("num_key_value_heads"),
            "num_hidden_layers": dec.get("num_hidden_layers"),
            "inputs": merged_in,
            "outputs": merged_out,
            "session_options": copy.deepcopy(dec.get("session_options")),
            "input_layout_kind": layout_kind,
            "prefill_pipeline_inputs": pin0,
            "prefill_pipeline_outputs": por0,
            "decode_pipeline_inputs": din0,
            "decode_pipeline_outputs": dor0,
        }
        tok = {
            "bos_token_id": model.get("bos_token_id"),
            "eos_token_id": model.get("eos_token_id"),
            "pad_token_id": model.get("pad_token_id"),
            "vocab_size": model.get("vocab_size"),
            "context_length": model.get("context_length"),
        }
        return {"model_partial": tok, "decoder_partial": out}

    # HuggingFace config.json (without nested decoder section)
    n_layers = src.get("num_hidden_layers")
    n_heads = src.get("num_attention_heads")
    n_kv = src.get("num_key_value_heads", n_heads)
    hidden = src.get("hidden_size")
    head_size = None
    if hidden is not None and n_heads:
        try:
            head_size = int(hidden) // int(n_heads)
        except (TypeError, ZeroDivisionError):
            head_size = None
    merged_in, merged_out, layout_hf = _merge_decoder_io_templates(None, None)
    dec = {
        "head_size": head_size,
        "hidden_size": hidden,
        "num_attention_heads": n_heads,
        "num_key_value_heads": n_kv,
        "num_hidden_layers": n_layers,
        "inputs": merged_in,
        "outputs": merged_out,
        "session_options": None,
        "input_layout_kind": layout_hf,
        "prefill_pipeline_inputs": None,
        "prefill_pipeline_outputs": None,
        "decode_pipeline_inputs": None,
        "decode_pipeline_outputs": None,
    }
    tok = {
        "bos_token_id": src.get("bos_token_id"),
        "eos_token_id": src.get("eos_token_id"),
        "pad_token_id": src.get("pad_token_id"),
        "vocab_size": src.get("vocab_size"),
        "context_length": src.get("max_position_embeddings"),
    }
    return {"model_partial": tok, "decoder_partial": dec}


def _validate_required(d: dict[str, Any], label: str) -> None:
    missing = [k for k, v in d.items() if v is None]
    if missing:
        raise ValueError(f"{label} missing fields: {', '.join(missing)}")


def build_pipeline_config(
    src_flat: dict[str, Any],
    *,
    fixed_prompt_length: int,
    max_length: int,
    prefill_filename: str,
    decode_filename: str,
    session_options_override: dict[str, Any] | None,
    model_dir: str | None = None,
    read_onnx_io: bool = True,
    onnx_io_search_subdirs: tuple[str, ...] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    extracted = _extract_decoder_fields(src_flat)
    tok = extracted["model_partial"]
    dec = extracted["decoder_partial"]

    n_layers = dec["num_hidden_layers"]
    _validate_required(
        {
            "num_hidden_layers": n_layers,
            "head_size": dec["head_size"],
            "hidden_size": dec["hidden_size"],
            "num_attention_heads": dec["num_attention_heads"],
            "num_key_value_heads": dec["num_key_value_heads"],
            "vocab_size": tok["vocab_size"],
        },
        "building pipeline config",
    )

    inputs_tmpl = dec["inputs"] or {}
    outputs_tmpl = dec["outputs"] or {}
    layout = str(dec.get("input_layout_kind") or INPUT_LAYOUT_FULL_IDS)
    pin_src = dec.get("prefill_pipeline_inputs")
    por_src = dec.get("prefill_pipeline_outputs")
    din_src = dec.get("decode_pipeline_inputs")
    dor_src = dec.get("decode_pipeline_outputs")

    for key in ("past_key_names", "past_value_names"):
        if not inputs_tmpl.get(key):
            raise ValueError(f"decoder.inputs incomplete, missing: {key}")
    if layout == INPUT_LAYOUT_FULL_IDS:
        if not inputs_tmpl.get("input_ids"):
            raise ValueError("decoder.inputs incomplete, missing: input_ids")
    elif layout == INPUT_LAYOUT_EMBEDS:
        if not inputs_tmpl.get("inputs_embeds"):
            raise ValueError("decoder.inputs incomplete, missing: inputs_embeds")

    session_opts = _canonical_morphizen_session_options()
    if session_options_override:
        for k, v in session_options_override.items():
            if k in ("session.disable_cpu_ep_fallback", "provider_options"):
                continue
            session_opts[k] = copy.deepcopy(v)

    meta: dict[str, Any] = {
        "pipeline_io_source": "template",
        "onnx_prefill_path": None,
        "onnx_decode_path": None,
    }
    pin: list[str] | None = None
    pout: list[str] | None = None
    pin_decode: list[str] | None = None
    pout_decode: list[str] | None = None

    if isinstance(pin_src, list) and pin_src:
        pin = list(pin_src)
        meta["pipeline_io_source"] = "source_config"
        if isinstance(por_src, list) and por_src:
            pout = list(por_src)
        else:
            pout = _pipeline_outputs_from_layer_count(int(n_layers))
        if isinstance(din_src, list) and din_src:
            pin_decode = list(din_src)
        else:
            pin_decode = list(pin)
        if isinstance(dor_src, list) and dor_src:
            pout_decode = list(dor_src)
        else:
            pout_decode = list(pout)
    else:
        subdirs = onnx_io_search_subdirs or _DEFAULT_ONNX_SEARCH_SUBDIRS
        if read_onnx_io and model_dir:
            pp = _find_onnx_path(model_dir, prefill_filename, subdirs)
            dp = _find_onnx_path(model_dir, decode_filename, subdirs)
            if pp and dp:
                try:
                    pin, pout = _onnx_graph_io_lists(pp)
                    pin_decode, pout_decode = _onnx_graph_io_lists(dp)
                    meta["pipeline_io_source"] = "onnx"
                    meta["onnx_prefill_path"] = pp
                    meta["onnx_decode_path"] = dp
                except Exception as exc:  # noqa: BLE001
                    meta["onnx_read_error"] = str(exc)
                    pin = pout = pin_decode = pout_decode = None

        if pin is None and layout in (INPUT_LAYOUT_FULL_IDS, INPUT_LAYOUT_EMBEDS):
            pin = _pipeline_inputs_from_decoder_templates(inputs_tmpl, int(n_layers))
            pout = _pipeline_outputs_from_layer_count(int(n_layers))
            pin_decode = list(pin)
            pout_decode = list(pout)
            if meta.get("pipeline_io_source") != "source_config":
                meta["pipeline_io_source"] = "template"
        elif pin is None:
            raise ValueError(
                "Cannot infer pipeline I/O: no decoder.pipeline.prefill.inputs in source; "
                "ONNX files for prefill/decode were not found or could not be read; and "
                "decoder.inputs has neither input_ids nor inputs_embeds for template lists."
            )

    # Keep key order aligned with Llama decoder-pipeline configs in this repo for cleaner diffs.
    decoder: dict[str, Any] = {
        "head_size": dec["head_size"],
        "hidden_size": dec["hidden_size"],
        "num_attention_heads": dec["num_attention_heads"],
        "num_key_value_heads": dec["num_key_value_heads"],
        "num_hidden_layers": dec["num_hidden_layers"],
    }
    fp = int(fixed_prompt_length)
    decoder["fixed_prompt_length"] = fp
    decoder["sliding_window"] = {
        "window_size": fp,
        "alignment": "left",
        "slide_inputs": True,
        "slide_key_value_cache": False,
    }
    decoder["session_options"] = session_opts
    decoder["inputs"] = inputs_tmpl
    decoder["outputs"] = outputs_tmpl
    decoder["pipeline"] = {
        "prefill": {
            "filename": prefill_filename,
            "run_on_prompt": True,
            "run_on_token_gen": False,
            "inputs": pin,
            "outputs": pout,
        },
        "decode": {
            "filename": decode_filename,
            "run_on_prompt": False,
            "run_on_token_gen": True,
            "inputs": pin_decode,
            "outputs": pout_decode,
        },
    }

    model_out: dict[str, Any] = {
        "bos_token_id": tok["bos_token_id"],
        "eos_token_id": tok["eos_token_id"],
        "pad_token_id": tok["pad_token_id"],
        "vocab_size": tok["vocab_size"],
        "context_length": int(max_length),
        "type": "decoder-pipeline",
        "decoder": decoder,
    }

    search = {
        "diversity_penalty": 0.0,
        "do_sample": True,
        "early_stopping": True,
        "length_penalty": 1.0,
        "max_length": int(max_length),
        "min_length": 0,
        "no_repeat_ngram_size": 0,
        "num_beams": 1,
        "num_return_sequences": 1,
        "past_present_share_buffer": True,
        "repetition_penalty": 1.0,
        "temperature": 0.6,
        "top_k": 1,
        "top_p": 0.9,
    }
    if isinstance(src_flat.get("search"), dict):
        for k, v in src_flat["search"].items():
            if k in search:
                search[k] = v
        search["max_length"] = int(max_length)

    return {"model": model_out, "search": search}, meta


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Read model-folder config and generate decoder-pipeline genai_config.json",
    )
    p.add_argument(
        "model_dir",
        help="Model directory (must contain genai_config.json, or use --input-config)",
    )
    p.add_argument(
        "-o",
        "--output",
        help="Output JSON path (default: <model_dir>/genai_config_pipeline.json)",
    )
    p.add_argument(
        "--input-config",
        metavar="NAME",
        help="Config filename under model_dir (default auto-detect: genai_config.json or config.json)",
    )
    p.add_argument(
        "--fixed-prompt-length",
        type=int,
        default=128,
        metavar="P",
        help="decoder.fixed_prompt_length; sliding_window.window_size uses the same value (default 128).",
    )
    p.add_argument(
        "--max-length",
        type=int,
        default=None,
        metavar="M",
        help="model.context_length and search.max_length (default from source context_length/max_position_embeddings)",
    )
    p.add_argument(
        "--stem",
        default=None,
        help="Variant filename stem used for default prefill/decode names: "
        "prefill_{stem}.onnx / decode_{stem}.onnx; if omitted, auto-generate "
        "p{P}m{M} (or 12200 when P=12200)",
    )
    p.add_argument(
        "--prefill",
        dest="prefill_filename",
        default=None,
        help="Override prefill ONNX filename (default prefill_{stem}.onnx)",
    )
    p.add_argument(
        "--decode",
        dest="decode_filename",
        default=None,
        help="Override decode ONNX filename (default decode_{stem}.onnx)",
    )
    p.add_argument(
        "--session-options-json",
        metavar="PATH",
        help="JSON object: merge additional keys after fixed session_options (same as genai_config_12200)",
    )
    p.add_argument(
        "--no-onnx-io",
        action="store_true",
        help="Do not read pipeline prefill/decode I/O from ONNX; use template lists only.",
    )
    p.add_argument(
        "--onnx-subdir",
        action="append",
        default=None,
        dest="onnx_subdirs_append",
        metavar="RELDIR",
        help="Prepend RELDIR to ONNX search dirs (relative to model_dir). Repeatable.",
    )
    args = p.parse_args(argv)

    model_dir = os.path.abspath(args.model_dir)
    if not os.path.isdir(model_dir):
        print(f"error: not a directory: {model_dir}", file=sys.stderr)
        return 2

    cfg_path = _find_config_file(model_dir, args.input_config)
    src_root = _load_json(cfg_path)

    max_len = args.max_length
    if max_len is None:
        model = src_root.get("model")
        if isinstance(model, dict) and model.get("context_length") is not None:
            max_len = int(model["context_length"])
        elif src_root.get("max_position_embeddings") is not None:
            max_len = int(src_root["max_position_embeddings"])
        else:
            print(
                "error: cannot infer max_length; pass --max-length",
                file=sys.stderr,
            )
            return 2

    fixed_p = int(args.fixed_prompt_length)
    stem = args.stem or _default_stem(fixed_p, max_len)
    prefill_fn = args.prefill_filename or f"prefill_{stem}.onnx"
    decode_fn = args.decode_filename or f"decode_{stem}.onnx"

    session_override = None
    if args.session_options_json:
        session_override = _load_json(args.session_options_json)
        if not isinstance(session_override, dict):
            raise ValueError("session-options-json root must be an object")

    onnx_sd: tuple[str, ...] | None = None
    if args.onnx_subdirs_append:
        onnx_sd = tuple(
            str(x).strip() for x in args.onnx_subdirs_append if str(x).strip()
        )
        onnx_sd = onnx_sd + _DEFAULT_ONNX_SEARCH_SUBDIRS

    out_cfg, io_meta = build_pipeline_config(
        src_root,
        fixed_prompt_length=fixed_p,
        max_length=max_len,
        prefill_filename=prefill_fn,
        decode_filename=decode_fn,
        session_options_override=session_override,
        model_dir=model_dir,
        read_onnx_io=not args.no_onnx_io,
        onnx_io_search_subdirs=onnx_sd,
    )

    out_path = args.output or os.path.join(model_dir, "genai_config_pipeline.json")
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_cfg, f, indent=4, ensure_ascii=False)
        f.write("\n")

    print(f"wrote: {out_path}")
    print(f"  stem={stem} prefill={prefill_fn} decode={decode_fn} max_length={max_len}")
    src = io_meta.get("pipeline_io_source", "?")
    print(f"  pipeline_io_source={src}")
    if src == "onnx":
        print(f"  onnx_prefill={io_meta.get('onnx_prefill_path')}")
        print(f"  onnx_decode={io_meta.get('onnx_decode_path')}")
    elif io_meta.get("onnx_read_error"):
        print(f"  onnx_read_error={io_meta.get('onnx_read_error')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
