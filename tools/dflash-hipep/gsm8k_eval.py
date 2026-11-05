#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Score GSM8K generations saved by profile_oga_dflash.py."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def _boxed_content(text: str) -> str | None:
    marker = r"\boxed{"
    start = text.rfind(marker)
    if start < 0:
        return None
    content_start = start + len(marker)
    depth = 1
    for index in range(content_start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[content_start:index]
    return None


def _candidate_answer(text: str) -> str | None:
    boxed = _boxed_content(text)
    if boxed is not None:
        return boxed
    matches = re.findall(r"####\s*(.+)", text)
    if matches:
        return matches[-1].strip()
    answer_lines = [
        line.strip()
        for line in text.splitlines()
        if re.search(r"\b(final answer|answer)\b", line, re.IGNORECASE)
    ]
    return answer_lines[-1] if answer_lines else None


def _normalize_answer(text: str | None) -> str | None:
    if text is None:
        return None
    value = text.lower()
    value = re.sub(r"\\text(?:bf|rm|it)?\s*\{([^{}]*)\}", r"\1", value)
    value = re.sub(r"\\(?:mathrm|mathbf|text)\s*\{([^{}]*)\}", r"\1", value)
    value = value.replace("\\$", "$").replace(",", "").strip()
    value = re.sub(r"[$%]", "", value)
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"[^a-z0-9.+\-/ ]", "", value).strip()
    return value or None


def _numeric_value(text: str | None) -> str | None:
    normalized = _normalize_answer(text)
    if normalized is None:
        return None
    numbers = re.findall(r"-?(?:\d+(?:\.\d*)?|\.\d+)", normalized)
    if len(numbers) == 1:
        return numbers[0].lstrip("+")
    return None


def _score_output(output: str, reference: str) -> dict:
    candidate = _candidate_answer(output)
    gold = reference.split("####", 1)[-1].strip()
    candidate_numeric = _numeric_value(candidate)
    gold_numeric = _numeric_value(gold)
    return {
        "correct": (
            candidate_numeric is not None
            and gold_numeric is not None
            and candidate_numeric == gold_numeric
        ),
        "candidate": candidate,
        "candidate_normalized": _normalize_answer(candidate),
        "gold": gold,
        "gold_normalized": _normalize_answer(gold),
    }


def _mode_name(record: dict) -> str:
    return f"{record['mode']}:block={record['block_size']}"


def evaluate(
    results_path: Path,
    output_path: Path,
    block_size: int | None,
) -> None:
    results = json.loads(results_path.read_text(encoding="utf-8"))
    if results.get("dataset") != "gsm8k":
        raise ValueError("The profile must use --dataset gsm8k.")

    metadata = {
        sample["dataset_index"]: sample for sample in results["dataset_samples"]
    }
    evaluated: dict[str, list[dict]] = {}
    for record in results["generations"]:
        if (
            block_size is not None
            and record["mode"] == "dflash"
            and record["block_size"] != block_size
        ):
            continue
        sample = metadata[record["dataset_index"]]
        evaluated.setdefault(_mode_name(record), []).append(
            {
                "sample_index": record["sample_index"],
                "dataset_index": record["dataset_index"],
                **_score_output(record["output_text"], sample["answer"]),
            }
        )

    modes = {
        mode: {
            "correct": sum(item["correct"] for item in samples),
            "num_samples": len(samples),
            "samples": samples,
        }
        for mode, samples in evaluated.items()
    }
    payload = {
        "results_file": str(results_path),
        "dataset": "gsm8k",
        "modes": modes,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for mode, summary in modes.items():
        print(f"{mode}: {summary['correct']}/{summary['num_samples']}")
    print(f"Saved evaluation: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-file", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--block-size", type=int, default=None)
    args = parser.parse_args()
    evaluate(args.results_file, args.output_file, args.block_size)


if __name__ == "__main__":
    main()
