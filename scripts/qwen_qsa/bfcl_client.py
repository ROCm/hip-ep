#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Pinned official BFCL component selection, tool simulation and scoring for Qwen."""
# ruff: noqa: E402

import argparse
import copy
import hashlib
import json
import os
import random
import subprocess
import sys
import time
import types
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TASK = ROOT / "_local/qwen-qsa/bfcl-v4-50"
os.environ["BFCL_PROJECT_ROOT"] = str(TASK)
os.environ["HIP_EP_BFCL_TASK"] = str(TASK)
sys.path.insert(0, str(ROOT / "scripts/qsa"))
import bfcl_native_eval as official
from overrides import override

# Importing the existing adapter installs the minimal, offline model registry.
# All benchmark prompts, parsing, simulator code and checkers stay unmodified.
official.TASK = TASK
os.environ["BFCL_PROJECT_ROOT"] = str(TASK)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    deadline = time.monotonic() + 10
    while True:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (PermissionError, FileNotFoundError):
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.05)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=True, indent=2) + "\n")
    temporary.replace(path)


def source_control():
    assert (
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=official.SOURCE, text=True
        ).strip()
        == official.REVISION
    )
    assert not subprocess.check_output(
        ["git", "-c", "core.autocrlf=true", "diff", "--name-only"],
        cwd=official.SOURCE,
        text=True,
    ).strip(), "Pinned BFCL source has local changes"


class QwenHandler(official.NativeHandler):
    @override
    def _query_prompting(self, inference_data):
        identifier = uuid.uuid4().hex
        request = {
            "request_id": identifier,
            "variant": self.variant,
            "case_id": self.case_id,
            "category": self.category,
            "generation_index": self.generation_index,
            "messages": copy.deepcopy(inference_data["message"]),
            "max_new_tokens": 1024,
            "temperature": 0,
        }
        self.generation_index += 1
        write(self.queue / (identifier + ".request.json"), request)
        path = self.queue / (identifier + ".response.json")
        started = time.monotonic()
        while not path.exists():
            assert not (TASK / "abort.json").exists(), (
                "Campaign stopped; retain partial evidence"
            )
            assert time.time() - load(TASK / "lease.json")["unix"] < 90, (
                "Supervisor lease expired"
            )
            assert time.monotonic() - started < 1800, (
                "Generation exceeded 30-minute limit"
            )
            time.sleep(0.1)
        response = load(path)
        assert response["status"] == "complete", response
        assert response["variant"] == self.variant
        assert response["request_id"] == identifier
        assert (
            response["prompt_sha256"]
            == hashlib.sha256(response["prompt"].encode("utf-8")).hexdigest()
        )
        inference_data["inference_input_log"] = {
            "formatted_prompt": response["prompt"],
            "request_id": identifier,
        }
        self.requests.append(identifier)
        return response, time.monotonic() - started


def handler(variant, queue=None):
    name = "HipEP-Qwen27B-" + variant
    return QwenHandler("Qwen/Qwen3.8-27B", 0, name, False, queue=queue)


for variant in ["Original", "Changed"]:
    name = "HipEP-Qwen27B-" + variant
    official.registry.MODEL_CONFIG_MAPPING[name] = types.SimpleNamespace(
        model_name="Qwen/Qwen3.8-27B",
        display_name=name,
        url="https://huggingface.co/Qwen/Qwen3.8-27B",
        org="local experiment",
        license="Apache-2.0",
        model_handler=QwenHandler,
        input_price=None,
        output_price=None,
        is_fc_model=False,
        underscore_to_dot=False,
    )


def prepare():
    TASK.mkdir(parents=True, exist_ok=True)
    model = handler("Original")
    entry = official.load_dataset_entry("simple_python")[0]
    gold = official.load_ground_truth_entry("simple_python")[0]["ground_truth"]
    assert official.score(
        model,
        entry,
        "[calculate_triangle_area(base=10, height=5)]",
        gold,
        "simple_python",
    )["valid"]
    assert not official.score(
        model,
        entry,
        "[calculate_triangle_area(base=999, height=5)]",
        gold,
        "simple_python",
    )["valid"]
    from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
        execute_multi_turn_func_call,
    )

    config = official.load_dataset_entry("multi_turn_base")[0]["initial_config"]
    states = []
    for variant in ["Original", "Changed", "Original"]:
        _, state = execute_multi_turn_func_call(
            [],
            config,
            ["GorillaFileSystem"],
            handler(variant).model_name_underline_replaced,
            "qwen-pilot-isolation-control",
        )
        states.append(state["GorillaFileSystem"])
    assert states[0] is states[2] and states[0] is not states[1]
    rng, cases, initial = random.Random(42), [], []
    for category in official.SINGLE_TURN_CATEGORY + official.MULTI_TURN_CATEGORY:
        entries = official.load_dataset_entry(category)
        count = 2 if category in official.SINGLE_TURN_CATEGORY else 6
        eligible = range(1 if category == "simple_python" else 0, len(entries))
        for index in sorted(rng.sample(eligible, count)):
            entry = entries[index]
            case = dict(
                category=category,
                index=index,
                id=entry["id"],
                pair_index=len(cases),
                declared_turns=len(entry["question"]),
                declared_user_messages=sum(
                    m["role"] == "user" for t in entry["question"] for m in t
                ),
                function_count=len(entry["function"]),
            )
            cases.append(case)
            copied = copy.deepcopy(entry)
            data = model._pre_query_processing_prompting(copied)
            data = model.add_first_turn_message_prompting(data, copied["question"][0])
            initial.append({**case, "messages": data["message"]})
    assert len(cases) == 50 and len({c["id"] for c in cases}) == 50
    manifest = dict(
        bfcl_revision=official.REVISION,
        seed=42,
        cases=cases,
        scope="50-case stratified component pilot: 26 single-turn, 24 multi-turn",
        excluded=["web_search", "memory", "format_sensitivity"],
        official_overall_score=False,
    )
    for name, data in [("manifest.json", manifest), ("initial-messages.json", initial)]:
        target = TASK / name
        if target.exists():
            assert load(target) == data, "Preserve existing frozen selection"
        write(target, data)
    write(
        TASK / "checker-self-test.json",
        dict(
            status="passed",
            bfcl_revision=official.REVISION,
            positive_negative_ast=True,
            variant_tool_state_isolation=True,
        ),
    )
    print(
        "Prepared 50 fixed cases; official checker and tool-state isolation controls passed",
        flush=True,
    )


def run(variant, queue):
    assert load(TASK / "checker-self-test.json")["status"] == "passed"
    protocol_sha = sha(TASK / "protocol.json")
    path = TASK / "rows.jsonl"
    previous = (
        [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
    )
    done = {(r["variant"], r["id"]) for r in previous}
    cached = {}
    model = handler(variant, queue)
    for case in load(TASK / "manifest.json")["cases"]:
        if (variant, case["id"]) in done:
            continue
        category, index = case["category"], case["index"]
        if category not in cached:
            cached[category] = (
                official.load_dataset_entry(category),
                official.load_dataset_entry(
                    category, include_prereq=False, include_language_specific_hint=False
                ),
                []
                if "relevance" in category
                else official.load_ground_truth_entry(category),
            )
        entries, scoring, truth = cached[category]
        assert entries[index]["id"] == scoring[index]["id"] == case["id"]
        model.case_id, model.category = case["id"], category
        model.generation_index, model.requests = 0, []
        result, metadata = model.inference(copy.deepcopy(entries[index]), True, False)
        checked = official.score(
            model,
            scoring[index],
            result,
            truth[index]["ground_truth"] if truth else None,
            category,
        )
        row = dict(
            **case,
            variant=variant,
            result=result,
            metadata=metadata,
            check=checked,
            correct=checked["valid"],
            request_ids=model.requests,
            attempt_sha256=sha(queue / "attempt.json"),
            queue=queue.relative_to(ROOT).as_posix(),
            protocol_sha256=protocol_sha,
            bfcl_revision=official.REVISION,
        )
        with path.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(official.make_json_serializable(row), ensure_ascii=True)
                + "\n"
            )
        print(
            case["id"],
            variant,
            checked["valid"],
            "generations",
            len(model.requests),
            flush=True,
        )
    write(queue / "stop.json", {"status": "complete"})


def audit():
    from collections import Counter

    rows = [json.loads(s) for s in (TASK / "rows.jsonl").read_text().splitlines()]
    cases = load(TASK / "manifest.json")["cases"]
    assert len(rows) == 100 and len({(r["variant"], r["id"]) for r in rows}) == 100
    indexed = {(r["variant"], r["id"]): r for r in rows}
    cached, totals = {}, Counter()
    for case in cases:
        first_prompts = []
        for variant in ["Original", "Changed"]:
            row = indexed[variant, case["id"]]
            assert row["protocol_sha256"] == sha(TASK / "protocol.json")
            category, index = case["category"], case["index"]
            if category not in cached:
                cached[category] = (
                    official.load_dataset_entry(
                        category,
                        include_prereq=False,
                        include_language_specific_hint=False,
                    ),
                    []
                    if "relevance" in category
                    else official.load_ground_truth_entry(category),
                )
            entries, gold = cached[category]
            checked = official.score(
                handler(variant),
                entries[index],
                row["result"],
                copy.deepcopy(gold[index]["ground_truth"]) if gold else None,
                category,
            )
            assert checked["valid"] == row["correct"]
            queue = ROOT / row["queue"]
            assert sha(queue / "attempt.json") == row["attempt_sha256"]
            for name, expected in load(queue / "attempt.json")["source_sha256"].items():
                assert sha(queue / "source" / name) == expected, name
            responses = [
                load(queue / (identifier + ".response.json"))
                for identifier in row["request_ids"]
            ]
            assert responses
            for identifier, response in zip(row["request_ids"], responses):
                request = load(queue / (identifier + ".request.done"))
                assert (
                    response["status"] == "complete" and response["variant"] == variant
                )
                assert request["case_id"] == row["id"] and request["variant"] == variant
                assert (
                    response["prompt_sha256"]
                    == hashlib.sha256(response["prompt"].encode()).hexdigest()
                )
                assert response["completion_tokens"] == len(
                    response["generated_token_ids"]
                )
                assert response["attempt_sha256"] == row["attempt_sha256"]
                assert response["request_sha256"] == sha(
                    queue / (identifier + ".request.done")
                )
            first_prompts.append(responses[0]["prompt"])
            totals[variant] += int(row["correct"])
        assert first_prompts[0] == first_prompts[1], case["id"]
    write(
        TASK / "audit.json",
        dict(
            status="passed",
            cases=50,
            rows=100,
            official_scores_recomputed=True,
            first_prompts_identical=True,
            native_request_response_identity_verified=True,
            correct=dict(totals),
            manifest_sha256=sha(TASK / "manifest.json"),
            protocol_sha256=sha(TASK / "protocol.json"),
            rows_sha256=sha(TASK / "rows.jsonl"),
        ),
    )
    print(
        "All 100 variant rows independently rescored; request identities and initial prompts verified",
        dict(totals),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--variant", choices=["Original", "Changed"])
    parser.add_argument("--queue", type=Path)
    args = parser.parse_args()
    source_control()
    if args.prepare:
        prepare()
    elif args.audit:
        audit()
    else:
        assert args.variant and args.queue
        run(args.variant, args.queue)


if __name__ == "__main__":
    main()
