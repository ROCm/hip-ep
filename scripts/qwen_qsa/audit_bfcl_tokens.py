#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Independently retokenize every completed BFCL request and decode every response."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

from verify_handoff import WORK, sha

TASK = Path(os.environ.get("QWEN_BFCL_TASK", WORK / "bfcl-v4-50"))


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def flatten(value):
    if isinstance(value, list):
        return [v for item in value for v in flatten(item)]
    return [int(value)]


def step_limit_logs(value):
    if isinstance(value, dict):
        if value.get("role") == "handler_log" and str(
            value.get("content", "")
        ).startswith("Model has been forced to quit"):
            yield value["content"]
        for child in value.values():
            yield from step_limit_logs(child)
    elif isinstance(value, list):
        for child in value:
            yield from step_limit_logs(child)


def workload_checks(rows):
    profile = {r["id"]: r for r in load(TASK / "initial-profile.json")["cases"]}
    stops, groups = {}, {}
    for variant in sorted({r["variant"] for r in rows}):
        selected = [r for r in rows if r["variant"] == variant]
        stops[variant] = [
            r["id"] for r in selected if list(step_limit_logs(r["metadata"]))
        ]
        groups[variant] = {}
        for label, flag in [
            ("initially_pruned", True),
            ("initially_full_support", False),
        ]:
            subgroup = [
                r
                for r in selected
                if profile[r["id"]]["initial_prefill_exercises_pruning"] == flag
            ]
            groups[variant][label] = {
                "cases": len(subgroup),
                "correct": sum(r["correct"] for r in subgroup),
            }
    return dict(
        official_step_limited_case_ids=stops,
        official_step_limited_cases={v: len(ids) for v, ids in stops.items()},
        groups_by_frozen_initial_prompt=groups,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    import native_inference

    assert native_inference.WORK == WORK
    import torch
    from transformers import AutoTokenizer

    protocol = load(TASK / "protocol.json")
    snapshot = Path(load(WORK / "handoff-verification.json")["backbone"]["snapshot"])
    for name, expected in protocol["tokenizer_sha256"].items():
        assert sha(snapshot / name) == expected
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=False
    )
    eos = load(snapshot / "generation_config.json")["eos_token_id"]
    eos = {eos} if isinstance(eos, int) else set(eos)
    count, capped, eligible = 0, 0, 0
    raw = b"".join(
        s
        for s in (TASK / "rows.jsonl").read_bytes().splitlines(keepends=True)
        if s.endswith(b"\n")
    )
    rows = [json.loads(s) for s in raw.splitlines()]
    expected = 50 * len(protocol.get("variants", ["Original", "Changed"]))
    assert 0 < len(rows) <= expected
    assert args.partial or len(rows) == expected
    profile = {r["id"]: r for r in load(TASK / "initial-profile.json")["cases"]}
    turns = []
    for row in rows:
        inputs, outputs = [], []
        for identifier in row["request_ids"]:
            queue = native_inference.ROOT / row["queue"]
            request = load(queue / (identifier + ".request.done"))
            response = load(queue / (identifier + ".response.json"))
            text = tokenizer.apply_chat_template(
                request["messages"],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            assert text == response["prompt"]
            n = len(tokenizer(text, add_special_tokens=False).input_ids)
            tokens = response["generated_token_ids"]
            assert n == response["prompt_tokens"]
            if identifier == row["request_ids"][0]:
                assert n == profile[row["id"]]["initial_prompt_tokens"]
                assert (
                    response["prompt_sha256"]
                    == profile[row["id"]]["initial_prompt_sha256"]
                )
            assert len(tokens) == response["completion_tokens"]
            assert (
                tokenizer.decode(tokens, skip_special_tokens=True) == response["text"]
            )
            assert response["input_plus_output_tokens"] == n + len(tokens)
            assert response["pruning_eligible"] == (n + len(tokens) - 2 >= 2052)
            assert response["qsa_pruning_exercised"] == (
                row["variant"] in ["Changed", "Eight"] and response["pruning_eligible"]
            )
            if response["finish_reason"] == "eos":
                assert tokens[-1] in eos and not any(t in eos for t in tokens[:-1])
            else:
                assert response["finish_reason"] == "length"
                assert len(tokens) == response["maximum_new_tokens"] <= 1024
                assert not any(t in eos for t in tokens)
                capped += 1
            inputs.append(n)
            outputs.append(len(tokens))
            eligible += response["qsa_pruning_exercised"]
            count += 1
        assert flatten(row["metadata"]["input_token_count"]) == inputs, row["id"]
        assert flatten(row["metadata"]["output_token_count"]) == outputs, row["id"]
        counts = row["metadata"]["input_token_count"]
        turn_counts = (
            [len(flatten(t)) for t in counts] if isinstance(counts, list) else [1]
        )
        assert sum(turn_counts) == len(inputs)
        if row["category"].startswith("multi_turn_"):
            assert len(turn_counts) == len(row["result"])
        stopped = bool(list(step_limit_logs(row["metadata"])))
        offset = 0
        for turn_index, calls in enumerate(turn_counts):
            assert calls > 0
            inp, out = inputs[offset : offset + calls], outputs[offset : offset + calls]
            turns.append(
                dict(
                    variant=row["variant"],
                    id=row["id"],
                    category=row["category"],
                    turn_index_zero_based=turn_index,
                    declared_case_turns=row["declared_turns"],
                    model_calls=calls,
                    total_prompt_tokens=sum(inp),
                    minimum_prompt_tokens=min(inp),
                    maximum_prompt_tokens=max(inp),
                    total_completion_tokens=sum(out),
                    maximum_completion_tokens=max(out),
                    official_step_limit_hit=stopped
                    and turn_index == len(turn_counts) - 1,
                )
            )
            offset += calls
    assert not torch.cuda.is_initialized()
    result = dict(
        status="passed",
        cases=len({r["id"] for r in rows}),
        variant_rows=len(rows),
        complete=len(rows) == expected,
        generations=count,
        capped_generations=capped,
        actual_qsa_pruned_generations=eligible,
        every_prompt_retokenized=True,
        every_completion_decoded=True,
        official_metadata_token_counts_match=True,
        gpu_initialized=False,
        rows_sha256=hashlib.sha256(raw).hexdigest(),
        rows_prefix_bytes=len(raw),
        protocol_sha256=sha(TASK / "protocol.json"),
        observed_turn_records=len(turns),
        **workload_checks(rows),
    )
    (
        TASK / ("token-partial-audit.json" if args.partial else "token-audit.json")
    ).write_text(json.dumps(result, indent=2) + "\n")
    with (TASK / ("turns-partial.csv" if args.partial else "turns.csv")).open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(turns[0]))
        writer.writeheader()
        writer.writerows(turns)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
