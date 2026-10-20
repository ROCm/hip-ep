"""Compare original OGA, derived-target, and OGA-backed DFlash output."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

HIPEP_STUDY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HIPEP_STUDY_ROOT))

from oga_target_native import (
    DEFAULT_HIPEP_ROOT,
    DEFAULT_MODEL_DIR,
    DEFAULT_ROCM_ROOT,
    OgaDFlashTarget,
    dflash_generate_oga,
    ensure_runtime_environment,
)


DEFAULT_DRAFT = "z-lab/Qwen3.6-35B-A3B-DFlash"
DEFAULT_SOURCE_TARGET = Path(
    "/proj/xcohdstaff7/devledas/nobkup/models/"
    "Qwen3.6-35B-A3B_gpuep_1.0_gpu_128"
)
DEFAULT_HF_HOME = Path("/proj/xcohdstaff7/devledas/nobkup/huggingface")
REPOSITORY_ROOT = HIPEP_STUDY_ROOT.parent
PROMPTS = (
    "Explain speculative decoding in two concise sentences.",
    "Write a Python function that returns the square of an integer.",
    "What is the capital of France?",
)


def _load_original_baselines(
    source_target: Path,
    prompts: tuple[str, ...],
    max_new_tokens: int,
) -> tuple[list[dict[str, object]], str, list[int]]:
    """Run untouched target-only OGA, then release it before derived-target load."""
    import onnxruntime_genai as og

    config = og.Config(str(source_target))
    config.overlay(
        json.dumps(
            {
                "model": {
                    "decoder": {
                        "session_options": {
                            "disable_cpu_ep_fallback": "1",
                        }
                    }
                }
            }
        )
    )
    config.clear_providers()
    config.append_provider("AMDGPU")
    config.set_provider_option("AMDGPU", "profile", "hip")
    model = og.Model(config)
    device = str(model.device_type)
    if device.lower() == "cpu":
        raise RuntimeError("Original OGA target reported CPU despite the AMDGPU override")

    tokenizer = og.Tokenizer(model)
    eos_token_ids = [int(token) for token in tokenizer.eos_token_ids]
    records: list[dict[str, object]] = []
    for prompt in prompts:
        # OGA's template parser cannot evaluate Qwen's
        # ``enable_thinking is false`` expression. This is the exact
        # single-user, thinking-disabled rendering produced by the same Qwen
        # chat template through Transformers.
        prompt_text = (
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )
        prompt_ids = tokenizer.encode(prompt_text)
        params = og.GeneratorParams(model)
        params.set_search_options(
            max_length=int(prompt_ids.size) + max_new_tokens,
            do_sample=False,
        )
        generator = og.Generator(model, params)
        generator.append_tokens(prompt_ids)
        generated_ids: list[int] = []
        while not generator.is_done():
            generator.generate_next_token()
            generated_ids.append(int(generator.get_next_tokens()[0]))
        records.append(
            {
                "prompt": prompt,
                "prompt_ids": [int(token) for token in prompt_ids],
                "output_ids": [
                    *[int(token) for token in prompt_ids],
                    *generated_ids,
                ],
            }
        )
        del generator, params

    del tokenizer, model, config
    gc.collect()
    return records, device, eos_token_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-target",
        type=Path,
        default=DEFAULT_SOURCE_TARGET,
        help="Untouched OGA model used for the true target-only baseline.",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Derived DFlash target model.",
    )
    parser.add_argument("--draft-model", "--draft", dest="draft", default=DEFAULT_DRAFT)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optionally write the detailed validation report as JSON.",
    )
    parser.add_argument("--hipep-root", type=Path, default=DEFAULT_HIPEP_ROOT)
    parser.add_argument("--rocm-root", type=Path, default=DEFAULT_ROCM_ROOT)
    parser.add_argument("--hf-home", type=Path, default=DEFAULT_HF_HOME)
    parser.add_argument(
        "--draft-device",
        "--device",
        dest="draft_device",
        default="cpu",
        help="PyTorch device for the draft model; the OGA target remains on AMDGPU.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=4)
    args = parser.parse_args()
    if args.max_new_tokens < args.block_size:
        parser.error("--max-new-tokens must be >= --block-size for parity validation")

    ensure_runtime_environment(args.hipep_root, args.rocm_root)
    args.hf_home.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(args.hf_home))

    # Run and release the untouched model before loading the derived 35B target.
    original_records, original_device, stop_ids = _load_original_baselines(
        args.source_target.resolve(),
        PROMPTS,
        args.max_new_tokens,
    )

    # Load the derived target model through OGA/HIPEP.
    target = OgaDFlashTarget(args.target, args.hipep_root)

    import torch
    from transformers import AutoConfig, AutoTokenizer

    sys.path.insert(0, str(REPOSITORY_ROOT))
    from dflash.model import DFlashDraftModel

    # Load the DFlash draft model in CPU PyTorch.
    text_tokenizer = AutoTokenizer.from_pretrained(
        args.source_target,
        trust_remote_code=True,
    )
    draft_config = AutoConfig.from_pretrained(args.draft, trust_remote_code=True)
    if not hasattr(draft_config, "block_size"):
        draft_config.block_size = draft_config.dflash_config["block_size"]
    draft = DFlashDraftModel.from_pretrained(
        args.draft,
        config=draft_config,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(args.draft_device).eval()

    cases = []
    for index, original in enumerate(original_records):
        input_ids = torch.tensor(
            [original["prompt_ids"]],
            dtype=torch.long,
            device=args.draft_device,
        )
        derived_target_only = dflash_generate_oga(
            draft,
            target,
            input_ids,
            max_new_tokens=args.max_new_tokens,
            stop_token_ids=stop_ids,
            temperature=0.0,
            block_size=1,
            return_stats=True,
        )
        speculative = dflash_generate_oga(
            draft,
            target,
            input_ids,
            max_new_tokens=args.max_new_tokens,
            stop_token_ids=stop_ids,
            temperature=0.0,
            block_size=args.block_size,
            return_stats=True,
        )
        original_ids = original["output_ids"]
        derived_ids = derived_target_only.output_ids[0].detach().cpu().tolist()
        speculative_ids = speculative.output_ids[0].detach().cpu().tolist()
        prompt_length = len(original["prompt_ids"])
        original_text = text_tokenizer.decode(
            original_ids[prompt_length:],
            skip_special_tokens=True,
        )
        derived_text = text_tokenizer.decode(
            derived_ids[prompt_length:],
            skip_special_tokens=True,
        )
        speculative_text = text_tokenizer.decode(
            speculative_ids[prompt_length:],
            skip_special_tokens=True,
        )
        derived_matches_original = original_ids == derived_ids
        speculative_matches_derived = derived_ids == speculative_ids
        speculative_matches_original = original_ids == speculative_ids
        cases.append(
            {
                "index": index,
                "prompt": original["prompt"],
                "derived_matches_original": derived_matches_original,
                "speculative_matches_derived": speculative_matches_derived,
                "speculative_matches_original": speculative_matches_original,
                "original_target_ids": original_ids,
                "derived_target_ids": derived_ids,
                "speculative_ids": speculative_ids,
                "original_target_text": original_text,
                "derived_target_text": derived_text,
                "speculative_text": speculative_text,
                "acceptance_lengths": speculative.acceptance_lengths,
                "partial_rollback_count": speculative.partial_rollback_count,
            }
        )
        print(
            f"case {index}: derived_vs_original={derived_matches_original}, "
            f"speculative_vs_derived={speculative_matches_derived}, "
            f"speculative_vs_original={speculative_matches_original}, "
            f"acceptance={speculative.acceptance_lengths}, "
            f"partial_rollbacks={speculative.partial_rollback_count}",
            flush=True,
        )
        print(f"  original target: {original_text!r}", flush=True)
        print(f"  derived target:  {derived_text!r}", flush=True)
        print(f"  DFlash:          {speculative_text!r}", flush=True)

    report = {
        "original_oga_device": original_device,
        "derived_oga_device": str(target.model.device_type),
        "block_size": args.block_size,
        "max_new_tokens": args.max_new_tokens,
        "all_derived_match_original": all(
            case["derived_matches_original"] for case in cases
        ),
        "all_speculative_match_derived": all(
            case["speculative_matches_derived"] for case in cases
        ),
        "all_speculative_match_original": all(
            case["speculative_matches_original"] for case in cases
        ),
        "cases": cases,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {args.output}")
    if not report["all_derived_match_original"]:
        print("NOTE: derived block-size-1 output differs from untouched OGA")
    if not report["all_speculative_match_derived"]:
        print("NOTE: speculative output differs from derived block-size-1 decoding")
    if not report["all_speculative_match_original"]:
        print("NOTE: speculative output differs from untouched OGA")


if __name__ == "__main__":
    main()
