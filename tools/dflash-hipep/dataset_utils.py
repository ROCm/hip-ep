#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Dataset loading and prompt formatting shared by DFlash profiling tools."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


DEFAULT_DRAFT = "z-lab/Qwen3.6-35B-A3B-DFlash"
DEFAULT_HF_HOME = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
CACHE_DIR = Path(__file__).parent.parent / "cache"

DATASETS = {
    "gsm8k": {
        "load_args": ("openai/gsm8k", "main"),
        "load_kwargs": {"split": "test"},
        "format": lambda row: (
            f"{row['question']}\nPlease reason step by step, and put your "
            "final answer within \\boxed{}."
        ),
    },
    "math500": {
        "load_args": ("HuggingFaceH4/MATH-500",),
        "load_kwargs": {"split": "test"},
        "format": lambda row: (
            f"{row['problem']}\nPlease reason step by step, and put your "
            "final answer within \\boxed{}."
        ),
    },
    "humaneval": {
        "load_args": ("openai/openai_humaneval",),
        "load_kwargs": {"split": "test"},
        "format": lambda row: (
            "Write a solution to the following problem and make sure that it "
            f"passes the tests:\n```python\n{row['prompt']}\n```"
        ),
    },
    "mbpp": {
        "load_args": ("google-research-datasets/mbpp", "sanitized"),
        "load_kwargs": {"split": "test"},
        "format": lambda row: row["prompt"],
    },
    "mt-bench": {
        "load_args": ("HuggingFaceH4/mt_bench_prompts",),
        "load_kwargs": {"split": "train"},
        "format": lambda row: row["prompt"],
        "multi_turn": True,
    },
}


def _prepare_dataset(name: str) -> Path:
    from datasets import load_dataset

    config = DATASETS[name]
    CACHE_DIR.mkdir(exist_ok=True)
    output_path = CACHE_DIR / f"{name}.jsonl"
    temporary_path = output_path.with_name(f"{output_path.name}.{os.getpid()}.tmp")

    print(f"[download] {name} ...")
    dataset = load_dataset(*config["load_args"], **config["load_kwargs"])
    with temporary_path.open("w", encoding="utf-8") as output:
        for row in dataset:
            turns = (
                config["format"](row)
                if config.get("multi_turn")
                else [config["format"](row)]
            )
            item = {"turns": turns}
            if name == "humaneval":
                item.update(
                    task_id=row["task_id"],
                    task_prompt=row["prompt"],
                    entry_point=row["entry_point"],
                    test=row["test"],
                    canonical_solution=row.get("canonical_solution"),
                )
            elif name == "gsm8k":
                item.update(
                    question=row["question"],
                    answer=row["answer"],
                )
            output.write(json.dumps(item) + "\n")
    os.replace(temporary_path, output_path)
    print(f"[cached] {output_path}  ({len(dataset)} samples)")
    return output_path


def load_and_process_dataset(data_name: str) -> list[dict[str, Any]]:
    if data_name not in DATASETS:
        raise ValueError(f"Unknown dataset {data_name!r}. Available: {list(DATASETS)}")

    path = CACHE_DIR / f"{data_name}.jsonl"
    if not path.exists():
        _prepare_dataset(data_name)
    with path.open(encoding="utf-8") as source:
        dataset = [json.loads(line) for line in source]

    required_key = {
        "gsm8k": "answer",
        "humaneval": "test",
    }.get(data_name)
    if required_key and any(required_key not in item for item in dataset):
        _prepare_dataset(data_name)
        with path.open(encoding="utf-8") as source:
            dataset = [json.loads(line) for line in source]
    return dataset


def apply_chat_template(
    tokenizer: Any,
    messages: list[dict[str, str]],
    enable_thinking: bool,
) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
