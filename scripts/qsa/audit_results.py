#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Independent final audit of immutable inputs, parser results and paired rows."""

import json
import math
import re

from prepare_baseline import (
    REFERENCE,
    ROOT,
    SELECTOR,
    SELECTOR_SHA,
    WORK,
    original_folder,
    sha,
)
from report_results import main as render_report


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def parse(text):
    match = re.search(r"####\s*(-?[0-9.,]+)", text)
    return match.group(1).replace(",", "").rstrip(".") if match else None


def main():
    assert sha(SELECTOR) == sha(WORK / "selectors/selector-bf16.pt") == SELECTOR_SHA
    export = load(original_folder() / "export.json")
    for name, receipt in export["artifacts"].items():
        assert sha(original_folder() / name) == receipt["sha256"]
    native = load(WORK / "native/build.json")
    assert sha(WORK / "native/qsa_kernels_gfx1151.dll") == native["library_sha256"]
    for name, digest in native["source_sha256"].items():
        assert sha(ROOT / name) == digest
    quality = WORK / "paired-quality"
    decision = load(WORK / "acceptance-decision.json")
    assert decision["gsm8k_role"] == "diagnostic" and decision["gsm8k_user_accepted"]
    assert decision["dolma_ppl_increase_gate_percent"] == 5
    assert load(quality / "dolma-summary.json")["complete"]
    assert load(quality / "gsm8k-summary.json")["complete"]
    documents = json.loads(
        (REFERENCE / "data/selector-extended-v1/test.json").read_text(encoding="utf-8")
    )
    for row in rows(quality / "dolma.jsonl"):
        assert row["document_id"] == documents[row["document"]]["document_id"]
        assert row["targets"] == 4095 and row["cached_targets"] == 64
        assert len(row["cached_token_nll"]) == len(row["cached_predictions"]) == 64
        assert all(math.isfinite(x) and x >= 0 for x in row["cached_token_nll"])
        assert abs(row["cached_nll_sum"] - sum(row["cached_token_nll"])) < 1e-9
    gsm_data = load(REFERENCE / "data/gsm8k.json")
    prefix = "\n\n".join(
        f"Question: {x['question']}\nAnswer: {x['answer']}" for x in gsm_data["fewshot"]
    )
    historical_path = REFERENCE / "runs/300258/gsm8k.jsonl"
    historical = (
        {r["id"]: r["prompt"] for r in rows(historical_path)}
        if historical_path.exists()
        else {}
    )
    historical_compared = 0
    correction_folder = WORK / "excluded-gsm8k-cp1252"
    correction = load(correction_folder / "correction.json")
    for name, digest in correction["original_sha256"].items():
        assert sha(correction_folder / name) == digest
    prior = {
        (r["variant"], r["index"]): r for r in rows(correction_folder / "gsm8k.jsonl")
    }
    affected = {(r["variant"], r["index"]) for r in correction["affected_rows"]}
    assert len(affected) == 6 and correction["retained_identical_prompt_rows"] == 250
    assert load(quality / "gsm8k-protocol.json")["source_text_encoding"] == "UTF-8"
    for row in rows(quality / "gsm8k.jsonl"):
        item = gsm_data["items"][row["index"]]
        expected_prompt = prefix + "\n\nQuestion: " + item["question"] + "\nAnswer:"
        assert row["id"] == item["id"] and row["prompt"] == expected_prompt
        assert row["prediction"] == parse(row["completion"].split("Question:", 1)[0])
        assert row["gold"] == parse(item["answer"])
        assert row["correct"] == (row["prediction"] == row["gold"])
        key = (row["variant"], row["index"])
        if key in affected:
            assert row["prompt"] != prior[key]["prompt"]
        else:
            assert row == prior[key]
        if row["id"] in historical:
            assert row["prompt"] == historical[row["id"]]
            historical_compared += 1
    accepted_rows = 0
    for context in [4096, 16384, 32768, 65472]:
        folder = WORK / "paired-performance" / str(context)
        status, protocol = load(folder / "status.json"), load(folder / "protocol.json")
        assert (
            status["status"] == "complete"
            and status["native_original_session_unchanged"]
        )
        assert protocol["original_execution"] == "complete native hip-ep graph"
        assert protocol["batch"] == 1 and protocol["capacity"] == context + 64
        assert protocol["selector_sha256"] == SELECTOR_SHA
        assert protocol["kernel_sha256"] == native["library_sha256"]
        if protocol.get("process_isolation"):
            assert status["isolated_workers_sessions_unchanged"]
            for variant, trials in protocol["worker_schedule"]:
                worker = load(folder / f"worker-{variant}-{trials[0][0]}.json")
                assert worker["variant"] == variant and worker["trials"] == trials
                assert (
                    worker["sessions_unchanged"] and worker["cache_pointers_unchanged"]
                )
                assert worker["validation"]["finite"]
                assert worker["validation"]["replay_bit_exact"]
                if variant == "Changed":
                    assert worker["selector_integrity"]["parameters_unchanged"]
                    assert worker["selector_integrity"]["training_updates"] == 0
        for row in rows(folder / "rows.jsonl"):
            assert row["batch"] == 1 and row["decode_calls"] == 64
            assert all(
                math.isfinite(row[k]) and row[k] > 0
                for k in ["prefill_seconds", "decode_seconds"]
            )
            assert (
                abs(row["prefill_tokens_s"] * row["prefill_seconds"] - context) < 1e-7
            )
            assert (
                abs(row["decode_aggregate_tokens_s"] * row["decode_seconds"] - 64)
                < 1e-7
            )
            accepted_rows += 1
        validation = load(folder / "model-validation.json")
        assert all(
            validation[v]["finite"] and validation[v]["replay_bit_exact"]
            for v in ["Original", "Changed"]
        )
    for phase in ["dolma", "gsm8k"]:
        integrity = load(quality / (phase + "-integrity.json"))
        assert (
            integrity["parameters_unchanged"]
            and integrity["selector_parameter_dtype"] == "BF16"
        )
        assert integrity["training_updates"] == 0
    render_report()
    receipt = {
        "status": "passed",
        "batch": 1,
        "training_updates": 0,
        "acceptance": {
            "dolma_ppl_gate_pass": load(quality / "dolma-summary.json")[
                "gate_point_estimate_pass"
            ],
            "gsm8k_gate_enabled": False,
            "gsm8k_user_accepted": True,
        },
        "dolma_rows": 128,
        "gsm8k_rows": 256,
        "timing_rows": accepted_rows,
        "historical_prompt_rows_verified": historical_compared,
        "gsm8k_prompt_rows_corrected": len(affected),
        "selector_sha256": SELECTOR_SHA,
        "native_library_sha256": native["library_sha256"],
        "report_sha256": sha(ROOT / "docs/experiments/olmo-qsa-results.md"),
    }
    (WORK / "final-audit.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
