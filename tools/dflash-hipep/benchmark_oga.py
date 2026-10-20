"""Benchmark OGA-backed DFlash using the original repository's datasets and metrics."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from itertools import chain
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from oga_target_native import (
    DEFAULT_HIPEP_ROOT,
    DEFAULT_MODEL_DIR,
    DEFAULT_ROCM_ROOT,
    OgaDFlashTarget,
    dflash_generate_oga,
    ensure_runtime_environment,
)

random.seed(42)


DEFAULT_DRAFT = "z-lab/Qwen3.6-35B-A3B-DFlash"
DEFAULT_TOKENIZER = Path(
    "/proj/xcohdstaff7/devledas/nobkup/models/"
    "Qwen3.6-35B-A3B_gpuep_1.0_gpu_128"
)
DEFAULT_HF_HOME = Path("/proj/xcohdstaff7/devledas/nobkup/huggingface")
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = Path(__file__).parent.parent / "cache"

DATASETS = {
    "gsm8k": {
        "load_args": ("openai/gsm8k", "main"),
        "load_kwargs": {"split": "test"},
        "format": lambda x: "{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.".format(**x),
    },
    "math500": {
        "load_args": ("HuggingFaceH4/MATH-500",),
        "load_kwargs": {"split": "test"},
        "format": lambda x: "{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}.".format(**x),
    },
    "humaneval": {
        "load_args": ("openai/openai_humaneval",),
        "load_kwargs": {"split": "test"},
        "format": lambda x: "Write a solution to the following problem and make sure that it passes the tests:\n```python\n{prompt}\n```".format(**x),
    },
    "mbpp": {
        "load_args": ("google-research-datasets/mbpp", "sanitized"),
        "load_kwargs": {"split": "test"},
        "format": lambda x: x["prompt"],
    },
    "mt-bench": {
        "load_args": ("HuggingFaceH4/mt_bench_prompts",),
        "load_kwargs": {"split": "train"},
        "format": lambda x: x["prompt"],
        "multi_turn": True,
    },
}


def _prepare_dataset(name: str) -> Path:
    from datasets import load_dataset

    cfg = DATASETS[name]
    CACHE_DIR.mkdir(exist_ok=True)
    out_path = CACHE_DIR / f"{name}.jsonl"
    tmp_path = out_path.with_name(f"{out_path.name}.{os.getpid()}.tmp")

    print(f"[download] {name} ...")
    dataset = load_dataset(*cfg["load_args"], **cfg["load_kwargs"])

    with open(tmp_path, "w") as f:
        for row in dataset:
            if cfg.get("multi_turn"):
                turns = cfg["format"](row)
            else:
                turns = [cfg["format"](row)]
            f.write(json.dumps({"turns": turns}) + "\n")
    os.replace(tmp_path, out_path)

    with open(out_path) as f:
        num_samples = sum(1 for _ in f)
    print(f"[cached] {out_path}  ({num_samples} samples)")
    return out_path


def load_and_process_dataset(data_name: str) -> list[dict]:
    if data_name not in DATASETS:
        raise ValueError(f"Unknown dataset '{data_name}'. Available: {list(DATASETS.keys())}")

    path = CACHE_DIR / f"{data_name}.jsonl"
    if not path.exists():
        _prepare_dataset(data_name)

    with open(path) as f:
        return [json.loads(line) for line in f]


def _limit_dataset(dataset: list[dict], max_samples: int | None) -> list[dict]:
    if max_samples is None or len(dataset) <= max_samples:
        return dataset
    random.shuffle(dataset)
    return dataset[:max_samples]


def _apply_chat_template(tokenizer, messages: list[dict], enable_thinking: bool) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def _print_decode_summary(responses: list[dict[int, SimpleNamespace]], block_size: int) -> None:
    baseline_tpot = np.mean([r[1].time_per_output_token for r in responses])
    dflash_tpot = np.mean([r[block_size].time_per_output_token for r in responses])
    print(f"Baseline throughput: {1 / baseline_tpot:.2f} tok/s")
    print(f"DFlash throughput:  {1 / dflash_tpot:.2f} tok/s")
    print(f"Decoding speedup: {baseline_tpot / dflash_tpot:.2f}")

    mean_accept = np.mean([np.mean(r[block_size].acceptance_lengths) for r in responses])
    print(f"Average Acceptance length: {mean_accept:.2f}")

    acceptance_lengths = list(chain.from_iterable(r[block_size].acceptance_lengths for r in responses))
    histogram = [acceptance_lengths.count(b) / len(acceptance_lengths) for b in range(block_size + 1)]
    print(f"Acceptance length histogram: {[f'{x * 100:.1f}%' for x in histogram]}")


def main() -> None:
    sys.path.insert(0, str(REPOSITORY_ROOT))

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--draft-model", "--draft", default=DEFAULT_DRAFT)
    parser.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER))
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--sample-dump",
        type=int,
        default=0,
        help="Dump the first N baseline and DFlash outputs (0 disables it).",
    )
    parser.add_argument(
        "--sample-dump-file",
        type=Path,
        default=Path("sanity_samples.txt"),
    )
    parser.add_argument("--enable-thinking", action="store_true")
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
    args = parser.parse_args()

    if args.temperature >= 1e-5:
        parser.error("The OGA DFlash integration currently supports temperature=0 only")

    ensure_runtime_environment(args.hipep_root, args.rocm_root)
    args.hf_home.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(args.hf_home))

    # Initialize OGA/MorphiZen before importing PyTorch and its native libraries.
    target = OgaDFlashTarget(args.target, args.hipep_root)

    import torch
    from tqdm import tqdm
    from transformers import AutoConfig, AutoTokenizer

    from dflash.model import DFlashDraftModel

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=True,
    )
    draft_config = AutoConfig.from_pretrained(
        args.draft_model,
        trust_remote_code=True,
    )
    if not hasattr(draft_config, "block_size"):
        draft_config.block_size = draft_config.dflash_config["block_size"]
    configured_block_size = int(draft_config.block_size)
    block_size = (
        configured_block_size if args.block_size is None else args.block_size
    )
    if block_size != configured_block_size:
        print(
            f"Warning: overriding draft block size {configured_block_size} "
            f"with {block_size}.",
            flush=True,
        )

    draft_model = DFlashDraftModel.from_pretrained(
        args.draft_model,
        config=draft_config,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(args.draft_device).eval()

    dataset = load_and_process_dataset(args.dataset)
    dataset = _limit_dataset(dataset, args.max_samples)

    if args.sample_dump:
        args.sample_dump_file.parent.mkdir(parents=True, exist_ok=True)

    responses = []
    for idx, instance in enumerate(tqdm(dataset)):
        messages = []
        for user_content in instance["turns"]:
            messages.append({"role": "user", "content": user_content})
            input_text = _apply_chat_template(
                tokenizer,
                messages,
                args.enable_thinking,
            )
            input_ids = tokenizer.encode(
                input_text,
                return_tensors="pt",
            ).to(args.draft_device)

            response = {}
            for bs in [1, block_size]:
                response[bs] = dflash_generate_oga(
                    draft_model,
                    target=target,
                    input_ids=input_ids,
                    max_new_tokens=args.max_new_tokens,
                    stop_token_ids=[tokenizer.eos_token_id],
                    temperature=args.temperature,
                    block_size=bs,
                    return_stats=True,
                )

            spec_response = response[block_size]
            generated_ids = spec_response.output_ids[0, spec_response.num_input_tokens:]
            output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

            if args.sample_dump and len(responses) < args.sample_dump:
                bl_response = response[1]
                bl_ids = bl_response.output_ids[0, bl_response.num_input_tokens:]
                bl_text = tokenizer.decode(bl_ids, skip_special_tokens=True)
                with open(args.sample_dump_file, "a") as f:
                    f.write(f"\n{'=' * 80}\nSAMPLE {idx}\n{'=' * 80}\n")
                    f.write(f"\n----- PROMPT -----\n{input_text}\n")
                    f.write(f"\n----- bs=1 (baseline) -----\n{bl_text}\n")
                    f.write(f"\n----- bs={block_size} (DFlash) -----\n{output_text}\n")

            messages.append({"role": "assistant", "content": output_text})

        responses.append(response)

    _print_decode_summary(responses, block_size)
    print(
        "Rollback correction mismatches: "
        f"{target.rollback_correction_mismatch_count}"
    )


if __name__ == "__main__":
    main()
