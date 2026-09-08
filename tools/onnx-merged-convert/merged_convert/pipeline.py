#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from .bundle import (
    DEFAULT_OUTPUT_DIR_NAME,
    ModelBundle,
    PipelineKind,
    build_genai_config,
    _find_genai_config_source,
)
from .int8kv import convert_decoder_int8kv
from .lora_weight_pack import write_dequant_lora_artifacts, write_packed_lora_artifacts
from .pipeline_aliases import (
    CONVERT_PROFILE_LOW_BIT,
    convert_head_quantized,
    merge_split_pipeline,
    merge_split_pipeline_lowbit,
    merge_split_pipeline_quantized,
    patch_emb_quantized,
    process_one_onnx,
)
from .qdq_ext import CONVERT_PROFILE_LITE
from .step1_qdq_fp16 import (
    DECODER_WORK_EXTERNAL_DATA,
    ShapeFixConfig,
    convert_decoder_model,
    convert_lm_head_model,
    patch_emb_model,
)
from .step2_fp16_cleanup import patch_model_file
from .step4_unfix_seq_len import unfix_seq_len

DEFAULT_MAX_SEQ_LEN = 16384
INTERMEDIATES_DIR_NAME = "work"


def _apply_pure_fp16(*paths: Path) -> None:
    for path in paths:
        patch_model_file(path, path)


def _convert_quantized_linear(
    bundle: ModelBundle, work: Path, merged_path: Path
) -> dict[str, dict] | None:
    shape_prefill = ShapeFixConfig(
        enabled=True,
        batch=1,
        seq_len=bundle.prefill_seq_len,
        max_seq_len=DEFAULT_MAX_SEQ_LEN,
    )
    shape_decode = ShapeFixConfig(
        enabled=True, batch=1, seq_len=1, max_seq_len=DEFAULT_MAX_SEQ_LEN
    )
    shape_dyn = ShapeFixConfig(enabled=False)

    dec_prefill = work / "dec_prefill.onnx"
    dec_decode = work / "dec_decode.onnx"
    emb_out = work / "emb.onnx"
    head_out = work / "head.onnx"
    dynamic = work / "decoder_dynamic.onnx"
    lora_dequant_meta: dict[str, dict] = {}

    decoder_mode = (
        "folded Gemm weights + fp16 adapter inputs"
        if bundle.fold_gemm_weights
        else "quantized linear (MatMulNBits)"
    )
    print(f"  [1/5] QDQ removal + fp16 (decoder, {decoder_mode}) ...")
    convert_decoder_model(
        bundle.dec_prefill,
        dec_prefill,
        shape_fix=shape_prefill,
        bundle_root=bundle.input_dir,
        external_data_name=DECODER_WORK_EXTERNAL_DATA,
        gqa_seqlens_rewrite=False,
        pure_gemm=bundle.fold_gemm_weights,
        lora_dequant_meta=lora_dequant_meta if bundle.fold_gemm_weights else None,
    )
    convert_decoder_model(
        bundle.dec_decode,
        dec_decode,
        shape_fix=shape_decode,
        bundle_root=bundle.input_dir,
        external_data_name=DECODER_WORK_EXTERNAL_DATA,
        reuse_external_data=True,
        external_data_ref=dec_prefill,
        gqa_seqlens_rewrite=False,
        pure_gemm=bundle.fold_gemm_weights,
        lora_dequant_meta=lora_dequant_meta if bundle.fold_gemm_weights else None,
    )

    print("  [2/5] QDQ removal + fp16 (emb / lm_head) + pruned logits ...")
    patch_emb_model(
        bundle.emb,
        emb_out,
        shape_fix=shape_dyn,
        rewrite_head_data=True,
        head_data_dir=work,
    )
    convert_lm_head_model(
        bundle.head,
        head_out,
        shape_fix=shape_dyn,
        rewrite_head_data=False,
        head_data_dir=work,
        gather_unsqueeze=True,
    )

    print(f"  [3/5] Unfix seq_len ({bundle.prefill_seq_len}/1 -> dynamic) ...")
    unfix_seq_len(
        dec_prefill,
        dec_decode,
        dec_prefill,
        dynamic,
        static_seq_lens=(1, bundle.prefill_seq_len),
    )

    print("  [4/5] Merge emb + dynamic decoder + lm_head ...")
    data_name = f"{bundle.merged_stem}.data"
    merge_split_pipeline(
        emb_out,
        dynamic,
        head_out,
        merged_path,
        external_data_name=data_name,
    )
    print("  [5/5] Final fp16 activation cleanup on merged graph ...")
    _apply_pure_fp16(merged_path)
    return lora_dequant_meta if bundle.fold_gemm_weights and lora_dequant_meta else None


def _convert_int8_kv(bundle: ModelBundle, work: Path, merged_path: Path) -> None:
    shape_dyn = ShapeFixConfig(enabled=False)
    profile = CONVERT_PROFILE_LITE

    dec_prefill = work / "dec_prefill.onnx"
    dec_decode = work / "dec_decode.onnx"
    dynamic = work / "decoder_dynamic.onnx"
    emb_out = work / "emb.onnx"
    head_out = work / "head.onnx"

    print("  [1/5] QDQ removal + fp16 decoder (int8 KV preserved, prefill/decode) ...")
    convert_decoder_int8kv(
        bundle.dec_prefill,
        dec_prefill,
        bundle_root=bundle.input_dir,
        external_data_name=DECODER_WORK_EXTERNAL_DATA,
        gqa_seqlens_rewrite=False,
    )
    convert_decoder_int8kv(
        bundle.dec_decode,
        dec_decode,
        bundle_root=bundle.input_dir,
        external_data_name=DECODER_WORK_EXTERNAL_DATA,
        reuse_external_data=True,
        external_data_ref=dec_prefill,
        gqa_seqlens_rewrite=False,
    )

    print(f"  [2/5] Unfix seq_len ({bundle.prefill_seq_len}/1 -> dynamic) ...")
    unfix_seq_len(
        dec_prefill,
        dec_decode,
        dec_prefill,
        dynamic,
        static_seq_lens=(1, bundle.prefill_seq_len),
    )

    print("  [3/5] QDQ removal + fp16 (emb / lm_head) ...")
    patch_emb_quantized(
        bundle.emb,
        emb_out,
        shape_fix=shape_dyn,
        rewrite_head_data=False,
        head_data_dir=work,
        profile=profile,
    )
    convert_head_quantized(
        bundle.head,
        head_out,
        shape_fix=shape_dyn,
        rewrite_head_data=False,
        head_data_dir=work,
        gather_unsqueeze=False,
        profile=profile,
    )

    print("  [4/5] Merge emb + dynamic decoder + lm_head (pruned logits at merge) ...")
    data_name = f"{bundle.merged_stem}.data"
    merge_split_pipeline_quantized(
        emb_out,
        dynamic,
        head_out,
        merged_path,
        external_data_name=data_name,
    )
    print("  [5/5] Final fp16 activation cleanup on merged graph ...")
    _apply_pure_fp16(merged_path)


def _convert_low_bit(bundle: ModelBundle, work: Path, merged_path: Path) -> None:
    shape_prefill = ShapeFixConfig(
        enabled=True,
        batch=1,
        seq_len=bundle.prefill_seq_len,
        max_seq_len=DEFAULT_MAX_SEQ_LEN,
    )
    shape_decode = ShapeFixConfig(
        enabled=True, batch=1, seq_len=1, max_seq_len=DEFAULT_MAX_SEQ_LEN
    )
    shape_dyn = ShapeFixConfig(enabled=False)
    profile = CONVERT_PROFILE_LOW_BIT

    dec_prefill = work / "dec_prefill.onnx"
    dec_decode = work / "dec_decode.onnx"
    emb_out = work / "emb.onnx"
    head_out = work / "head.onnx"
    dynamic = work / "decoder_dynamic.onnx"

    print("  [1/5] QDQ removal + fp16 (decoder prefill/decode) ...")
    process_one_onnx(
        bundle.dec_prefill,
        dec_prefill,
        kind="decoder",
        shape_fix=shape_prefill,
        bundle_root=bundle.input_dir,
        head_data_dir=work,
        rewrite_head_data=False,
        external_data_name=DECODER_WORK_EXTERNAL_DATA,
        profile=profile,
    )
    process_one_onnx(
        bundle.dec_decode,
        dec_decode,
        kind="decoder",
        shape_fix=shape_decode,
        bundle_root=bundle.input_dir,
        head_data_dir=work,
        rewrite_head_data=False,
        external_data_name=DECODER_WORK_EXTERNAL_DATA,
        reuse_external_data=True,
        external_data_ref=dec_prefill,
        profile=profile,
    )

    print("  [2/5] QDQ removal + fp16 (emb / lm_head) ...")
    process_one_onnx(
        bundle.emb,
        emb_out,
        kind="emb",
        shape_fix=shape_dyn,
        bundle_root=bundle.input_dir,
        head_data_dir=work,
        rewrite_head_data=True,
        profile=profile,
    )
    process_one_onnx(
        bundle.head,
        head_out,
        kind="lm_head",
        shape_fix=shape_dyn,
        bundle_root=bundle.input_dir,
        head_data_dir=work,
        rewrite_head_data=False,
        gather_unsqueeze=False,
        profile=profile,
    )

    print(f"  [3/5] Unfix seq_len ({bundle.prefill_seq_len}/1 -> dynamic) ...")
    unfix_seq_len(
        dec_prefill,
        dec_decode,
        dec_prefill,
        dynamic,
        static_seq_lens=(1, bundle.prefill_seq_len),
    )

    print("  [4/5] Merge emb + dynamic decoder + lm_head (pruned logits) ...")
    data_name = f"{bundle.merged_stem}.data"
    merge_split_pipeline_lowbit(
        emb_out,
        dynamic,
        head_out,
        merged_path,
        external_data_name=data_name,
        gather_unsqueeze=True,
    )
    print("  [5/5] Final fp16 activation cleanup on merged graph ...")
    _apply_pure_fp16(merged_path)


def _reset_work_dir(work_dir: Path) -> Path:
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    return work_dir


def _run_pipeline(
    bundle: ModelBundle, work: Path, merged_path: Path
) -> dict[str, dict] | None:
    if bundle.pipeline is PipelineKind.QUANTIZED_LINEAR:
        return _convert_quantized_linear(bundle, work, merged_path)
    if bundle.pipeline is PipelineKind.INT8_KV:
        _convert_int8_kv(bundle, work, merged_path)
        return None
    if bundle.pipeline is PipelineKind.LOW_BIT:
        _convert_low_bit(bundle, work, merged_path)
        return None
    raise ValueError(f"Unsupported pipeline: {bundle.pipeline}")


def convert_bundle(
    bundle: ModelBundle,
    output_dir: Path,
    *,
    keep_intermediates: bool = False,
) -> Path:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    merged_path = output_dir / f"{bundle.merged_stem}.onnx"
    merged_data = output_dir / f"{bundle.merged_stem}.data"
    for path in (merged_path, merged_data):
        if path.exists():
            path.unlink()

    flags = []
    if bundle.fold_gemm_weights:
        flags.append("folded-gemm")
    flag_text = f", {', '.join(flags)}" if flags else ""
    print(
        f"\n=== {bundle.input_dir.name} ({bundle.pipeline.value}{flag_text}) "
        f"-> {output_dir.name}/{bundle.merged_stem}.onnx ==="
    )

    lora_dequant_meta: dict[str, dict] | None = None
    if keep_intermediates:
        work = _reset_work_dir(output_dir / INTERMEDIATES_DIR_NAME)
        lora_dequant_meta = _run_pipeline(bundle, work, merged_path)
        print(f"  Kept intermediate artifacts in {work}")
    else:
        with tempfile.TemporaryDirectory(
            prefix=f"merged_convert_{bundle.decoder_stem}_"
        ) as tmp:
            lora_dequant_meta = _run_pipeline(bundle, Path(tmp), merged_path)

    if not merged_path.exists():
        raise RuntimeError(
            f"Conversion finished but merged model missing: {merged_path}"
        )
    if not merged_data.exists():
        raise RuntimeError(
            f"Conversion finished but external weights missing: {merged_data}"
        )

    source_cfg = _find_genai_config_source(bundle.input_dir)
    cfg = build_genai_config(
        bundle,
        merged_path.name,
        merged_path,
        source=source_cfg,
    )
    cfg_path = output_dir / "genai_config.json"
    cfg_path.write_text(
        json.dumps(cfg, indent=4, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    extras: list[str] = []
    adapter_src = bundle.input_dir / "adapter.safetensors"
    if lora_dequant_meta:
        extras.extend(
            write_dequant_lora_artifacts(adapter_src, output_dir, lora_dequant_meta)
        )
    else:
        try:
            extras.extend(
                write_packed_lora_artifacts(merged_path, adapter_src, output_dir)
            )
        except FileNotFoundError as exc:
            if (
                bundle.pipeline is PipelineKind.QUANTIZED_LINEAR
                and not bundle.fold_gemm_weights
            ):
                raise RuntimeError(
                    "LoRA MatMulNBits export requires raw adapter.safetensors in input-dir"
                ) from exc

    extra_msg = f" + {', '.join(extras)}" if extras else ""
    print(
        f"  Wrote {merged_path.name} + {merged_data.name} + genai_config.json{extra_msg}"
    )
    return merged_path


def _default_output_dir(bundle: ModelBundle) -> Path:
    return bundle.input_dir / DEFAULT_OUTPUT_DIR_NAME
