#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import onnx

from .step1_qdq_fp16 import DEFAULT_MAX_SEQ_LEN

DEFAULT_OUTPUT_DIR_NAME = "merged"

_EMB_GLOBS = ("*_emb.onnx", "*embedding*.onnx", "*embeddings*.onnx")
_HEAD_GLOBS = ("*_lm_head.onnx", "*lm_head*.onnx")


class PipelineKind(str, Enum):
    QUANTIZED_LINEAR = "quantized_linear"
    INT8_KV = "int8_kv"
    LOW_BIT = "low_bit"


@dataclass(frozen=True)
class ModelBundle:
    input_dir: Path
    pipeline: PipelineKind
    decoder_stem: str
    dec_prefill: Path
    dec_decode: Path
    emb: Path
    head: Path
    prefill_seq_len: int
    merged_stem: str
    chunk_size: int
    fold_gemm_weights: bool = False


def _load_graph_summary(path: Path) -> dict[str, int | bool]:
    model = onnx.load(str(path), load_external_data=False)
    nodes = model.graph.node
    gemm = sum(1 for n in nodes if n.op_type == "Gemm")
    mnb = sum(1 for n in nodes if n.op_type == "MatMulNBits")
    two_bit_mnb = sum(
        1
        for n in nodes
        if n.op_type == "MatMulNBits"
        and any(a.name == "bits" and a.i == 2 for a in n.attribute)
    )
    int8_kv = any(
        n.op_type == "GroupQueryAttention"
        and any(a.name == "kv_cache_bit_width" and a.i == 8 for a in n.attribute)
        for n in nodes
    )
    return {
        "gemm": gemm,
        "mnb": mnb,
        "two_bit_mnb": two_bit_mnb,
        "int8_kv": int8_kv,
    }


def _infer_prefill_seq_len(prefill: Path) -> int:
    if prefill.stem.endswith("_128"):
        return 128
    model = onnx.load(str(prefill), load_external_data=False)
    for vi in model.graph.input:
        shape = vi.type.tensor_type.shape
        for dim in shape.dim:
            if dim.dim_value > 1:
                return int(dim.dim_value)
    for vi in model.graph.input:
        shape = vi.type.tensor_type.shape
        if len(shape.dim) >= 2 and shape.dim[1].dim_value > 0:
            return int(shape.dim[1].dim_value)
    return 128


def _find_component(input_dir: Path, globs: tuple[str, ...]) -> Path | None:
    seen: set[str] = set()
    for pattern in globs:
        for path in sorted(input_dir.glob(pattern)):
            if path.name in seen:
                continue
            seen.add(path.name)
            return path
    return None


def _find_decoder_pairs(input_dir: Path) -> list[tuple[Path, Path, str, int]]:
    pairs: list[tuple[Path, Path, str, int]] = []
    for prefill in sorted(input_dir.glob("*.onnx")):
        stem = prefill.stem
        if stem.endswith("_128"):
            decoder_stem = stem[: -len("_128")]
            decode = input_dir / f"{decoder_stem}_1.onnx"
            if decode.is_file():
                pairs.append((prefill, decode, decoder_stem, 128))
        elif stem.endswith("_0"):
            decoder_stem = stem[: -len("_0")]
            decode = input_dir / f"{decoder_stem}_1.onnx"
            if decode.is_file():
                pairs.append(
                    (prefill, decode, decoder_stem, _infer_prefill_seq_len(prefill))
                )
    return pairs


def _select_pipeline(prefill: Path) -> tuple[PipelineKind, bool]:
    summary = _load_graph_summary(prefill)
    if summary["int8_kv"]:
        return PipelineKind.INT8_KV, False
    if summary["gemm"] >= 50 and summary["mnb"] < 20:
        return PipelineKind.QUANTIZED_LINEAR, True
    if summary["two_bit_mnb"] >= 50:
        return PipelineKind.LOW_BIT, False
    return PipelineKind.QUANTIZED_LINEAR, False


def detect_bundle(input_dir: Path) -> ModelBundle:
    input_dir = input_dir.resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(input_dir)

    pairs = _find_decoder_pairs(input_dir)
    if not pairs:
        raise ValueError(
            f"No decoder pair found under {input_dir} "
            "(expected *_128.onnx + *_1.onnx or *_0.onnx + *_1.onnx)"
        )

    last_error: Exception | None = None
    for dec_prefill, dec_decode, decoder_stem, prefill_seq_len in pairs:
        emb = _find_component(input_dir, _EMB_GLOBS)
        head = _find_component(input_dir, _HEAD_GLOBS)
        if emb is None or head is None:
            last_error = FileNotFoundError(
                f"Missing embedding or lm_head ONNX next to decoder pair {dec_prefill.name}"
            )
            continue

        pipeline, fold_gemm_weights = _select_pipeline(dec_prefill)
        merged_stem = f"{decoder_stem}_merged"
        return ModelBundle(
            input_dir=input_dir,
            pipeline=pipeline,
            decoder_stem=decoder_stem,
            dec_prefill=dec_prefill,
            dec_decode=dec_decode,
            emb=emb,
            head=head,
            prefill_seq_len=prefill_seq_len,
            merged_stem=merged_stem,
            chunk_size=prefill_seq_len,
            fold_gemm_weights=fold_gemm_weights,
        )

    raise last_error or ValueError(f"Cannot classify model bundle: {input_dir}")


def _find_genai_config_source(input_dir: Path) -> Path | None:
    candidates = [
        input_dir / "genai_config.json",
        input_dir / "processed" / "genai_config.json",
    ]
    for pattern in (
        "*_fp16_no_rewrite",
        "*_fp16_pure_gemm",
        "*_fp16",
        "processed",
        "merged",
    ):
        candidates.extend(sorted(input_dir.glob(f"{pattern}/genai_config.json")))
    seen: set[Path] = set()
    for path in candidates:
        path = path.resolve()
        if path in seen:
            continue
        seen.add(path)
        if path.is_file():
            return path
    return None


def introspect_merged_io(merged_path: Path) -> dict[str, int | list[str]]:
    model = onnx.load(str(merged_path), load_external_data=False)
    graph = model.graph

    input_names = [vi.name for vi in graph.input]

    num_layers = sum(1 for name in input_names if name.startswith("past_keys_"))

    pipeline_inputs = ["input_ids", "past_seq_len", "total_seq_len"]
    for i in range(num_layers):
        pipeline_inputs.extend([f"past_keys_{i}", f"past_values_{i}"])

    pipeline_outputs = ["logits"]
    for i in range(num_layers):
        pipeline_outputs.extend([f"present_keys_{i}", f"present_values_{i}"])

    return {
        "num_layers": num_layers,
        "pipeline_inputs": pipeline_inputs,
        "pipeline_outputs": pipeline_outputs,
    }


def _pipeline_stage(filename: str, io: dict) -> dict:
    return {
        "filename": filename,
        "session_options": {
            "provider_options": [{"AMDGPU": {"profile": "hip"}}],
        },
        "inputs": io["pipeline_inputs"],
        "outputs": io["pipeline_outputs"],
    }


def build_genai_config(
    bundle: ModelBundle,
    merged_filename: str,
    merged_path: Path,
    *,
    source: Path | None,
) -> dict:
    io = introspect_merged_io(merged_path)
    prefill = _pipeline_stage(merged_filename, io)
    prefill["run_on_prompt"] = True
    prefill["run_on_token_gen"] = False
    decode = _pipeline_stage(merged_filename, io)
    decode["run_on_prompt"] = False
    decode["run_on_token_gen"] = True

    if source is not None and source.is_file():
        cfg = json.loads(source.read_text(encoding="utf-8"))
    else:
        cfg = {
            "model": {
                "context_length": DEFAULT_MAX_SEQ_LEN,
                "type": "decoder-pipeline",
                "decoder": {
                    "session_options": {
                        "log_id": "onnxruntime-genai",
                        "session.disable_cpu_ep_fallback": "1",
                        "provider_options": [{"AMDGPU": {"profile": "hip"}}],
                    },
                    "head_size": 128,
                    "hidden_size": 2048,
                    "num_attention_heads": 24,
                    "num_hidden_layers": int(io["num_layers"]),
                    "num_key_value_heads": 8,
                    "inputs": {
                        "input_ids": "input_ids",
                        "past_sequence_length": "past_seq_len",
                        "total_sequence_length": "total_seq_len",
                        "past_key_names": "past_keys_%d",
                        "past_value_names": "past_values_%d",
                    },
                    "outputs": {
                        "logits": "logits",
                        "present_key_names": "present_keys_%d",
                        "present_value_names": "present_values_%d",
                    },
                },
            },
            "search": {
                "diversity_penalty": 0.0,
                "do_sample": False,
                "early_stopping": True,
                "length_penalty": 1.0,
                "max_length": DEFAULT_MAX_SEQ_LEN,
                "min_length": 0,
                "no_repeat_ngram_size": 0,
                "num_beams": 1,
                "num_return_sequences": 1,
                "past_present_share_buffer": True,
                "repetition_penalty": 1.0,
                "temperature": 0.01,
                "top_k": 5,
                "top_p": 1.0,
                "chunk_size": bundle.chunk_size,
            },
        }

    model = cfg.setdefault("model", {})
    decoder = model.setdefault("decoder", {})
    decoder["num_hidden_layers"] = int(io["num_layers"])
    decoder["pipeline"] = [{"prefill": prefill, "decode": decode}]

    # Merged graph always takes token ids at the boundary, not precomputed embeddings.
    inputs = decoder.setdefault("inputs", {})
    inputs.pop("embeddings", None)
    inputs["input_ids"] = "input_ids"
    inputs.setdefault("past_sequence_length", "past_seq_len")
    inputs.setdefault("total_sequence_length", "total_seq_len")
    inputs.setdefault("past_key_names", "past_keys_%d")
    inputs.setdefault("past_value_names", "past_values_%d")

    session_opts = decoder.setdefault("session_options", {})
    for opt in session_opts.get("provider_options", []):
        if isinstance(opt, dict) and "AMDGPU" in opt:
            opt["AMDGPU"]["profile"] = "hip"

    if "vocab_size" not in model:
        for tok_cfg in (
            bundle.input_dir / "tokenizer" / "config.json",
            bundle.input_dir / "config.json",
        ):
            if tok_cfg.is_file():
                vocab = json.loads(tok_cfg.read_text(encoding="utf-8")).get(
                    "vocab_size"
                )
                if isinstance(vocab, int) and vocab > 0:
                    model["vocab_size"] = vocab
                    break

    search = cfg.setdefault("search", {})
    search["chunk_size"] = bundle.chunk_size
    if "max_length" not in search:
        search["max_length"] = model.get("context_length", DEFAULT_MAX_SEQ_LEN)

    return cfg
