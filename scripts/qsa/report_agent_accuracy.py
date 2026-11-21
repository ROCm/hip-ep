#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Report native GSM8K, audited BFCL component pilot and SWE-bench readiness."""

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TASK = ROOT / "_local/qsa/native-gqa"


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def token_counts(value):
    if isinstance(value, list):
        return [n for item in value for n in token_counts(item)]
    return [int(value)]


def main():
    gsm = load(TASK / "gsm8k/gsm8k-summary.json")
    assert gsm["complete"] and not gsm["acceptance_gate_enabled"]
    assert load(TASK / "final-audit.json")["gsm8k_rows_verified"] == 256
    lines = [
        "<!--",
        "Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.",
        "Licensed under the MIT License.",
        "-->",
        "# Native QSA accuracy: GSM8K and agent benchmarks",
        "",
        "Default and QSA use the same pinned OLMo-3-1025-7B base model, native RTN W4A16 group128 backbone, physical Radeon 8060S GPU and single-session execution. Batch is one. QSA changes only layers 7/11/15/23; frozen BF16 selectors and all backbone weights are unchanged. Training, recovery, distillation and learned calibration are zero.",
        "",
        "## Native GSM8K: complete 128-question pilot",
        "",
        "| Default | QSA | Change | Paired 95% interval | Gate |",
        "|---:|---:|---:|---|---|",
    ]
    a, b = [gsm["variants"][v] for v in ["Original", "Changed"]]
    ci = gsm["paired_95_interval_percentage_points"]
    lines += [
        f"| {a['correct']}/128 ({a['accuracy_percent']:.2f}%) | {b['correct']}/128 ({b['accuracy_percent']:.2f}%) | {gsm['accuracy_change_percentage_points']:+.3f} pp | [{ci[0]:+.3f}, {ci[1]:+.3f}] pp | Disabled |",
        "",
        "Fresh native runs reproduce the earlier accepted 94/128 versus 91/128 scores. This is the retained five-shot pilot, not the full GSM8K test split. Both variants use greedy generation, EOS/Question stopping, a 512-token cap and the same strict #### parser. Three generations per variant reached the cap. The native report audit independently verified all 256 UTF-8 prompts and recomputed parsed answers and correctness from raw completions. GSM8K remains diagnostic and nongating.",
        "",
        "## BFCL v4: stratified component pilot",
        "",
        "The fixed seed-42 manifest selects eight cases from each of 13 single-turn categories and six from each of four multi-turn categories: 128 cases, 256 variant rows. The public simple Python transport/checker control is excluded. Variant order alternates across pairs. The official source is pinned at `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8`.",
        "",
        "This pilot covers single-turn tool calling, relevance/irrelevance and multi-turn components. It excludes web-search, memory and format-sensitivity categories. Its pooled score is not the official weighted BFCL v4 overall score.",
        "",
    ]
    bfcl = TASK / "bfcl-v4"
    receipt_path = bfcl / "pilot-audit.json"
    audited = receipt_path.exists()
    if audited:
        receipt = load(receipt_path)
        assert receipt["status"] == "passed" and receipt["rows"] == 256
        assert sha(bfcl / "pilot-manifest.json") == receipt["manifest_sha256"]
        assert sha(bfcl / "protocol.json") == receipt["protocol_sha256"]
        for name, expected in receipt["row_files_sha256"].items():
            assert sha(bfcl / name) == expected
        manifest = load(bfcl / "pilot-manifest.json")
        all_rows = {}
        for path in bfcl.glob("*/rows.jsonl"):
            for raw in path.read_text(encoding="utf-8").splitlines():
                row = json.loads(raw)
                all_rows[row["variant"], row["id"]] = row
        categories = list(dict.fromkeys(c["category"] for c in manifest["cases"]))
        calls = []
        generated = {"Original": 0, "Changed": 0}
        for case in manifest["cases"]:
            for variant in ["Original", "Changed"]:
                metadata = all_rows[variant, case["id"]]["metadata"]
                inputs = token_counts(metadata["input_token_count"])
                outputs = token_counts(metadata["output_token_count"])
                assert len(inputs) == len(outputs)
                calls.extend(zip(inputs, outputs))
                generated[variant] += sum(outputs)
        differences, counts = [], {"Original": 0, "Changed": 0}
        lines += [
            "| Component | Cases | Default correct | QSA correct |",
            "|---|---:|---:|---:|",
        ]
        for category in categories:
            cases = [c for c in manifest["cases"] if c["category"] == category]
            values = {
                v: np.array(
                    [all_rows[v, c["id"]]["correct"] for c in cases], dtype=np.int32
                )
                for v in counts
            }
            for variant in counts:
                counts[variant] += int(values[variant].sum())
            differences.append(values["Changed"] - values["Original"])
            lines.append(
                f"| `{category}` | {len(cases)} | {values['Original'].sum()}/{len(cases)} | {values['Changed'].sum()}/{len(cases)} |"
            )
        assert sum(len(d) for d in differences) == 128
        rng = np.random.default_rng(42)
        samples = (
            sum(
                d[rng.integers(0, len(d), size=(10000, len(d)))].sum(1)
                for d in differences
            )
            / 128
        )
        ci = 100 * np.quantile(samples, [0.025, 0.975])
        delta = 100 * (counts["Changed"] - counts["Original"]) / 128
        lines += [
            "",
            f"**Pooled component pilot:** Default {counts['Original']}/128 ({100 * counts['Original'] / 128:.2f}%), QSA {counts['Changed']}/128 ({100 * counts['Changed'] / 128:.2f}%); change {delta:+.3f} percentage points. Paired stratified-bootstrap 95% interval: [{ci[0]:+.3f}, {ci[1]:+.3f}] pp (10,000 resamples, seed 42, sampling within each component). This interval describes this pilot, not full-leaderboard uncertainty.",
            "",
            "Every selected row was independently rescored with official evaluation inputs, including original Java/JavaScript types, and each pair's initial prompt was checked for equality. Native runtime/infrastructure failures were not scored as wrong answers. Setup-failure logs and completed-row provenance are retained separately.",
            "",
            f"The pilot made {len(calls):,} native generation requests. The largest observed input was {max(p for p, _ in calls):,} tokens, and the largest input plus generated continuation was {max(p + n for p, n in calls):,} tokens, within a 65,536-token allocation. Including a long-context category does not by itself establish accuracy at 64K: a model can fail tool-use steps before reaching that context length.",
            "",
            f"Recorded generated tokens total {generated['Original']:,} for Default and {generated['Changed']:,} for QSA. Variant continuations and tool paths can diverge; accuracy-run latency is therefore not a fixed-token throughput comparison. The separate native performance report uses identical teacher-forced token inputs for that comparison.",
        ]
    else:
        lines += [
            "The paired pilot is still running or awaiting its independent scoring audit. No completed BFCL score is reported yet."
        ]
    lines += [
        "",
        "The adapter uses official BFCL system prompts, parsers, simulated tool backends and scoring functions. A registry containing only the two local model variants avoids unrelated commercial API SDK imports. Tool histories are isolated by variant and test case. The base tokenizer has no chat template: both variants use the same fixed plain System/User/Assistant transcript, greedy generation, a 4096-token cap (bounded by the 65,536-token context), EOS or fixed transcript-role stopping, and no constrained decoding. No benchmark answers or demonstrations are injected into generation prompts. Official positive/negative AST controls and variant tool-history isolation passed before generation.",
        "",
        "Full BFCL v4's web-search component needs a configured SerpAPI backend; none was found. Memory and web-search results are not included in the component pilot. Missing categories are not assigned zero or represented as an official overall score. See the [official BFCL setup](https://github.com/ShishirPatil/gorilla/tree/6ea57973c7a6097fd7c5915698c54c17c5b1b6c8/berkeley-function-call-leaderboard).",
        "",
        "## SWE-bench Verified: blocked before execution",
        "",
        "Neither Windows nor the existing Ubuntu WSL environment has Docker or Podman installed. An existing container host/configuration has been requested. No SWE-bench predictions or resolved-task score have been produced; this is an infrastructure block, not a zero model score. The native mini-SWE adapter (`mini_native_model.py`) and Docker-only override (`swebench_native.yaml`) are prepared. CPU fixture controls accepted one shell action and rejected multiple actions using the official text-action parser; these controls perform no real model inference or shell execution and are not SWE-bench results. End-to-end container validation remains blocked.",
        "",
        "Preparation pins the Verified test dataset to `c104f840cc67f8b6eec6f759ebc8b2693d585d4a`, mini-SWE-agent's v2.0.0 tag to `ce91a8c6449bc5a686df8f6b7ee8ca9433d893bc`, and the official evaluator source to `02e7a74ffd0b707aab73d203fe87bdc7c76afc8e`. A text-based agent configuration is selected for the unchanged base model; no instruction-tuned replacement is substituted. See the [official container evaluation guide](https://www.swebench.com/SWE-bench/guides/evaluation/).",
        "",
        "## Local evidence and reproduction",
        "",
        "Native GSM8K: `scripts/qsa/run_native_gsm8k.py`; artifacts under `_local/qsa/native-gqa/gsm8k/`. Its four serial stages preserve exact commands, script snapshots and frozen-selector checks.",
        "",
        "BFCL: install `scripts/qsa/bfcl_cpu_requirements.txt` in an isolated Linux/Python 3.12 environment, then run `bfcl_native_eval.py --self-test` and `--prepare-pilot` with the pinned official source available. `run_native_bfcl_stage.py --label <fresh-label> --pilot --cpu-python <WSL-python-path>` runs a serial native generation worker and CPU client under `run_stage.py`. CPU dependencies are isolated from the ROCm environment. Queue files retain request prompts, native responses/token IDs and worker/client logs. `audit_native_bfcl.py` independently rescores all pilot rows before this report emits a score. Source/data, protocol, manifest and row-file hashes are retained under `_local/qsa/native-gqa/bfcl-v4/` and `_local/qsa/stages/`.",
        "",
        "SWE-bench preparation and blocker details are recorded in `_local/qsa/benchmarks/swebench-plan.json`. The [native integration results](olmo-qsa-native-gqa-results.md) retain Dolma quality and paired 4K–64K performance. Long-context timing does not establish broad long-context accuracy.",
        "",
    ]
    output = ROOT / "docs/experiments/olmo-qsa-agent-accuracy.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    (TASK / "agent-accuracy-report.json").write_text(
        json.dumps(
            {
                "gsm8k_complete": True,
                "bfcl_pilot_audited": audited,
                "swebench_state": "blocked",
                "report_sha256": sha(output),
            },
            indent=2,
        )
        + "\n"
    )
    print(output)


if __name__ == "__main__":
    main()
