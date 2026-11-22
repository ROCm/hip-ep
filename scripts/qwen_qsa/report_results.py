#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Audit fresh default quality rows and render the native Qwen evidence record."""

import hashlib
import json
import math
from pathlib import Path

from audit_memory import audit_memory
from audit_performance import audit_performance
from evaluate_default import answer
from verify_handoff import ROOT, SOURCE, WORK, sha


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def compare_native_quality(quality):
    if not all(item["state"] == "complete" for item in quality.values()):
        return {"state": "pending"}
    folder = WORK / "native-gqa/quality"
    protocols = {
        variant: load(folder / variant / "protocol.json") for variant in quality
    }
    original, changed = protocols["Original"], protocols["Changed"]
    differences = {
        key
        for key in original.keys() | changed.keys()
        if original.get(key) != changed.get(key)
    }
    assert differences == {"variant", "selectors_loaded", "export_onnx_sha256"}
    rows = {
        variant: {
            row["index"]: row
            for row in map(
                json.loads,
                (folder / variant / "gsm8k.jsonl")
                .read_text(encoding="utf-8")
                .splitlines(),
            )
        }
        for variant in quality
    }
    token_differences = sum(
        rows["Original"][i]["generated_token_ids"]
        != rows["Changed"][i]["generated_token_ids"]
        for i in range(128)
    )
    ppl_change = 100 * (quality["Changed"]["ppl"] / quality["Original"]["ppl"] - 1)
    gsm_change = (
        100
        * (quality["Changed"]["gsm_correct"] - quality["Original"]["gsm_correct"])
        / 128
    )
    limits = original["quality_gates_for_later_matched_qsa"]
    return {
        "state": "complete",
        "ppl_change_percent": ppl_change,
        "gsm_change_percentage_points": gsm_change,
        "gsm_token_sequence_differences": token_differences,
        "ppl_gate_passed": ppl_change <= limits["ppl_increase_percent"],
        "gsm_gate_passed": gsm_change >= -limits["gsm_drop_percentage_points"],
        "limits": limits,
        "training_updates": 0,
    }


def audit_quality(folder=None, variant="Original", export_folder=None):
    folder = folder or WORK / "quality/Original"
    if not (folder / "protocol.json").exists():
        return {"state": "not_started", "ppl_documents": 0, "gsm_questions": 0}
    protocol_path = folder / "protocol.json"
    protocol = load(protocol_path)
    exported = load((export_folder or WORK / "default-rtn-w4a16-64") / "export.json")
    assert protocol["export_onnx_sha256"] == exported["onnx_sha256"]
    assert protocol["export_data_sha256"] == exported["data_sha256"]
    directory = SOURCE / "data/qwen-selector-v1"
    for name, expected in protocol["data_sha256"].items():
        assert sha(directory / name) == expected
    documents, questions = [
        load(directory / name) for name in ["test.json", "gsm8k.json"]
    ]
    datasets = {}
    for name in ["ppl", "gsm8k"]:
        path = folder / (name + ".jsonl")
        rows = (
            [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            if path.exists()
            else []
        )
        assert len({r["index"] for r in rows}) == len(rows)
        assert all(
            r["protocol_sha256"] == sha(protocol_path) and r["variant"] == variant
            for r in rows
        )
        datasets[name] = rows
    for row in datasets["ppl"]:
        assert 0 <= row["index"] < 64
        document = documents[row["index"]]
        assert row["document_id"] == document["document_id"]
        assert row["targets"] == len(document["input_ids"]) - 1 == 4095
        assert math.isfinite(row["nll_sum"]) and row["nll_sum"] > 0
    if datasets["gsm8k"]:
        from transformers import AutoTokenizer

        snapshot = Path(
            load(WORK / "handoff-verification.json")["backbone"]["snapshot"]
        )
        for name, expected in protocol["tokenizer_sha256"].items():
            assert sha(snapshot / name) == expected
        tokenizer = AutoTokenizer.from_pretrained(
            snapshot, local_files_only=True, trust_remote_code=False
        )
    for row in datasets["gsm8k"]:
        assert 0 <= row["index"] < 128
        question = questions[row["index"]]
        assert (
            row["question"] == question["question"]
            and row["gold_answer"] == question["answer"]
        )
        assert (
            hashlib.sha256(row["prompt"].encode()).hexdigest() == row["prompt_sha256"]
        )
        expected_prompt = tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": question["question"]
                    + "\nSolve the problem and end with #### followed by the numerical answer.",
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        assert row["prompt"] == expected_prompt
        assert row["prompt_tokens"] == len(
            tokenizer(expected_prompt, add_special_tokens=False).input_ids
        )
        assert row["completion"] == tokenizer.decode(
            row["generated_token_ids"], skip_special_tokens=True
        )
        assert row["predicted"] == answer(row["completion"])
        assert row["gold"] == answer(question["answer"])
        assert row["correct"] == (
            row["predicted"] is not None and row["predicted"] == row["gold"]
        )
        assert 0 < len(row["generated_token_ids"]) <= 512
        assert all(
            isinstance(t, int) and 0 <= t < 248320 for t in row["generated_token_ids"]
        )
        assert not set(row["generated_token_ids"][:-1]) & set(protocol["eos_token_ids"])
        assert row["stopped_on_eos"] == (
            row["generated_token_ids"][-1] in protocol["eos_token_ids"]
        )
        assert row["reached_cap"] == (
            not row["stopped_on_eos"] and len(row["generated_token_ids"]) == 512
        )
    ppl, gsm = datasets["ppl"], datasets["gsm8k"]
    complete = len(ppl) == 64 and len(gsm) == 128
    receipt = {
        "state": "complete" if complete else "partial",
        "ppl_documents": len(ppl),
        "gsm_questions": len(gsm),
        "protocol_sha256": sha(protocol_path),
        "row_files_sha256": {
            name: sha(folder / (name + ".jsonl"))
            for name in datasets
            if (folder / (name + ".jsonl")).exists()
        },
        "gsm_correct": sum(r["correct"] for r in gsm),
        "gsm_caps": sum(r["reached_cap"] for r in gsm),
        "training_updates": 0,
    }
    if ppl:
        receipt["targets"] = sum(r["targets"] for r in ppl)
        receipt["nll"] = sum(r["nll_sum"] for r in ppl) / receipt["targets"]
        receipt["ppl"] = math.exp(receipt["nll"])
    (folder / "audit.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def audit_long_quality(folder=None, variant="Original", export_folder=None):
    folder = folder or WORK / "long-quality-default"
    if not (folder / "rows.jsonl").exists():
        return []
    protocol_path = folder / "protocol.json"
    protocol = load(protocol_path)
    data = WORK / "long-quality-v1"
    manifest = load(data / "manifest.json")
    assert sha(data / "manifest.json") == protocol["dataset_manifest_sha256"]
    assert sha(data / "documents.json") == protocol["documents_sha256"]
    assert protocol["documents_sha256"] == manifest["documents_sha256"]
    exported = load((export_folder or WORK / "default-rtn-w4a16-64") / "export.json")
    assert protocol["export_onnx_sha256"] == exported["onnx_sha256"]
    assert protocol["export_data_sha256"] == exported["data_sha256"]
    documents = load(data / "documents.json")
    rows = [
        json.loads(line) for line in (folder / "rows.jsonl").read_text().splitlines()
    ]
    assert len({(r["context"], r["index"]) for r in rows}) == len(rows)
    for row in rows:
        assert row["protocol_sha256"] == sha(protocol_path)
        assert row["variant"] == variant and row["training_updates"] == 0
        assert row["context"] in manifest["contexts"] and 0 <= row["index"] < 4
        document = documents[row["index"]]
        for name in ["document_sha256", "window_sha256"]:
            assert row[name] == document[name]
        assert len(document["input_ids"]) >= row["context"]
        assert row["targets"] == protocol["scored_suffix_targets"] == 1023
        assert row["finite"] and math.isfinite(row["nll_sum"]) and row["nll_sum"] > 0
    summary = []
    for context in sorted(manifest["contexts"]):
        selected = [r for r in rows if r["context"] == context]
        if len(selected) == 4:
            targets = sum(r["targets"] for r in selected)
            nll = sum(r["nll_sum"] for r in selected) / targets
            summary.append(
                {
                    "context": context,
                    "documents": 4,
                    "targets": targets,
                    "nll": nll,
                    "ppl": math.exp(nll),
                }
            )
    (folder / "audit.json").write_text(
        json.dumps(
            {
                "state": "complete"
                if len(summary) == len(manifest["contexts"])
                else "partial",
                "protocol_sha256": sha(protocol_path),
                "rows_sha256": sha(folder / "rows.jsonl"),
                "rows": summary,
                "training_updates": 0,
            },
            indent=2,
        )
        + "\n"
    )
    return summary


def main():
    verified = load(WORK / "handoff-verification.json")
    support = load(WORK / "native-support.json")
    probe = load(WORK / "default-4-support.json")
    reference = load(WORK / "default-4-reference.json")
    assert all(x["status"] == "passed" for x in [verified, support, probe, reference])
    quality = audit_quality()
    long_quality = audit_long_quality()
    native_quality = {
        variant: audit_quality(
            WORK / "native-gqa/quality" / variant,
            variant,
            WORK / "native-gqa/models" / variant,
        )
        for variant in ["Original", "Changed"]
    }
    comparison = compare_native_quality(native_quality)
    performance = audit_performance()
    memory = audit_memory()
    native_long_quality = {
        variant: audit_long_quality(
            WORK / "native-gqa/long-quality" / variant,
            variant,
            WORK / "native-gqa/models" / variant,
        )
        for variant in ["Original", "Changed"]
    }
    trained_path = SOURCE / "docs/reproducibility/qwen-trained-performance.json"
    trained = load(trained_path)
    assert trained["state"] == "passed" and trained["job"] == "303195"
    assert trained["training_job"] == "303042" and trained["selected_step"] == 128
    assert (
        trained["selector_sha256"]
        == "2d2672cf7108d233bc4ff1f7259a4aff4f674434199165d966b493817a1d5edd"
    )
    assert sha(SOURCE / "runs/303195/performance.jsonl") == trained["raw_sha256"]
    assert all(r["before"] == r["measured"] for r in trained["runtime"].values())
    long_manifest_path = WORK / "long-quality-v1/manifest.json"
    long_manifest = load(long_manifest_path)
    assert long_manifest["state"] == "locked"
    assert (
        sha(WORK / "long-quality-v1/documents.json")
        == long_manifest["documents_sha256"]
    )
    timing_path = WORK / "timing-v1/manifest.json"
    timing = load(timing_path)
    assert timing["state"] == "locked"
    assert sha(WORK / "timing-v1/streams.json") == timing["streams_sha256"]
    lines = [
        "<!--",
        "Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.",
        "Licensed under the MIT License.",
        "-->",
        "# Qwen trained-selector port: native default and QSA",
        "",
        "## Current evidence",
        "",
        "Fresh 64-block native default quality is complete and independently audited. Four-block architecture diagnostics pass native HIP and frozen Hugging Face numerical controls. Native QSA integration passes kernel, complete-operator and full-model numerical/dispatch controls. Matched 4K PPL/GSM results are below; long-document QSA accuracy and performance remain separate qualifications.",
        "",
        "Work is isolated on `codex/qwen-native-qsa`, created after OLMo accuracy report commit `7e0e1e3`. The earlier OLMo branch, binaries and results are preserved. The source sparse-attn project is read-only. Training, fine-tuning, recovery training, distillation, learned calibration and selector updates are zero.",
        "",
        "## Frozen identities",
        "",
        "| Input | Identity |",
        "|---|---|",
        f"| Model | `{verified['model']}` |",
        f"| Revision | `{verified['revision']}` |",
        f"| Trained selector payload | `{verified['selector_sha256']}` |",
        "| Selector checkpoint | Job 303042, step 128; layers 27/31/35/39 |",
        "| Selector tensors | 16 BF16 tensors, 13,108,736 parameters; every shape and raw-byte hash verified |",
        "| Backbone | All 18 official BF16 shard byte hashes match the source identity manifest |",
        "| Full text backbone | 26,895,998,464 parameters; 53,791,996,928 BF16 bytes, excluding vision/MTP |",
        "",
        "The official local snapshot and selector/data payloads remain unchanged. No OLMo or community selector is substituted. Source MI355X results are references: BF16 PPL 5.787418 → 5.791103 and GSM 109/128 → 111/128. They are not hip-ep measurements or a substitute for a fresh converted default.",
        "",
        "## Target representation and native behavior",
        "",
        f"The target is **{probe['gpu']}**, gfx1151/wave32, with {support['ram_total_bytes'] / 2**30:.2f} GiB physical shared memory. The native probe uses PyTorch {probe['torch']}, HIP {probe['hip']} and Transformers {support['transformers']}. The default measurements use `_local/qwen-qsa/install`, which preserves the unchanged four hash-verified OLMo native compiler/runtime binaries. The Qwen port uses a separate build/install directory.",
        "",
        "Native GQA currently supports FP16/FP32 rather than BF16. Full text FP16 weights alone occupy about 50.10 GiB before GPU cache/workspaces and host/compiler copies. A fresh deterministic RTN W4A16 group128 backbone is therefore selected for this machine; small Gated DeltaNet a/b projections, embedding and output head remain FP16. Both later variants must use these exact exported backbone bytes. This is a new conversion from the official checkpoint, with its own default accuracy; it is not a restart of Qwen MXFP4 or reuse of earlier quantized accuracy claims.",
        "",
        "The pinned release ONNX supplies topology and static RoPE/shape constants only. Every learned constant is regenerated from the verified official BF16 weights. Its topology hash is `44b712df872250c1c1614c3aa717788a103f3a814e0cfd24b22196d3b99509cd`. Export manifests record exact source-parameter coverage, per-projection conversion errors, representation, output graph/data hashes and exporter hash. The four-block diagnostic contains 5,940,028,480 bytes of fresh constants. The full export has 64 blocks: 48 Gated DeltaNet and 16 full-attention layers; native attention is Q24/KV4/D256.",
        "",
        "The current LinearAttention ABI has one dtype for Q/K/V, gates and recurrent state. Explicit FP32 LinearAttention regions and casts retain **FP32 recurrent state** with FP16 backbone boundaries. The rest of the graph remains identical between the planned variants. Its conversion and numerical error must be evaluated rather than called BF16-exact.",
        "",
        "**Gate metadata control:** the checkpoint config includes `output_gate_type=swish`, but the actual Transformers 5.19 Qwen3_5Attention class used by the source experiment executes sigmoid gating. A CPU single-key control matches sigmoid exactly (maximum absolute error 0) and differs from SWISH by 1.761594. The export preserves the executed native **sigmoid** behavior; changing it based solely on that extra metadata field would change the source model.",
        "",
        "## Native default numerical controls",
        "",
        "The four-block diagnostic contains three Gated DeltaNet blocks and one full-attention block, using real checkpoint weights. Native sessions have CPU fallback disabled. It checks finite prefill/decode outputs, exact replay after state reset and FP32 recurrent buffers.",
        "",
        f"Full eight-token prefill versus seven-token prefill plus one decode differs by {100 * probe['full_vs_incremental_relative_rms']:.5f}% relative RMS. Replay is bit-exact.",
        "",
        "The independent HF control uses the exact unpacked RTN weights and FP16 unquantized parameters, with no fitting. The predeclared diagnostic limit is 2% relative RMS.",
        "",
        "| Prefix / continuation | Native vs frozen HF relative RMS |",
        "|---|---:|",
    ]
    for row in reference["rows"]:
        label = (
            f"{row['tokens']} tokens"
            if "tokens" in row
            else f"{row['prefill']} + {row['decode']} cached token"
        )
        lines.append(f"| {label} | {100 * row['relative_rms']:.5f}% |")
    long_path = WORK / "default-4-reference-long.json"
    if long_path.exists():
        long_reference = load(long_path)
        assert long_reference["status"] == "passed"
        for row in long_reference["rows"]:
            if row.get("tokens", 0) > 17 or row.get("prefill", 0) > 17:
                label = (
                    f"{row['tokens']} tokens"
                    if "tokens" in row
                    else f"{row['prefill']} + {row['decode']} cached token"
                )
                lines.append(f"| {label} | {100 * row['relative_rms']:.5f}% |")
    lines += [
        "",
        "These are architecture/export diagnostics, not a full-model accuracy score or performance benchmark. The HF fallback implementation is used for correctness only; no timing comparison against that fallback is claimed.",
        "",
        "## Fresh native quality",
        "",
        "The locked Qwen protocol uses 64 test windows of 4,096 tokens and 128 zero-shot, thinking-disabled GSM questions, greedy generation with a 512-token reserve and both official EOS IDs. The parser takes the last #### number, falls back to the last number, and applies Decimal normalization. It does not use OLMo's five-shot prompt/strict parser. Qwen gates remain PPL increase ≤5% and GSM drop ≤2 percentage points against the fresh default. The earlier OLMo nongating decision is not transferred to Qwen.",
        "",
    ]
    if quality["state"] == "complete":
        lines += [
            "| Metric | Native default | Native trained QSA |",
            "|---|---:|---|",
            f"| PPL, 64 × 4K windows | {quality['ppl']:.6f} | Pending |",
            f"| GSM8K, 128 questions | {quality['gsm_correct']}/128 ({100 * quality['gsm_correct'] / 128:.2f}%) | Pending |",
            "",
            f"All rows pass protocol/data identity and score/count audits. {quality['gsm_caps']} default GSM generations reached the cap. Raw completions and capped cases are retained. No native QSA quality result is implied.",
        ]
    elif quality["ppl_documents"] == 64:
        lines += [
            "| Metric | Native default | Native trained QSA |",
            "|---|---:|---|",
            f"| PPL, 64 × 4K windows | {quality['ppl']:.6f} | Pending |",
            "| GSM8K, 128 questions | In progress | Pending |",
            "",
            "All 64 native default PPL windows are complete and pass the frozen protocol/count audit. The 128-question default GSM campaign is still in progress. No native QSA quality result is established yet.",
        ]
    else:
        lines += [
            "The default quality campaign is in progress. Completed native scores will be reported after the frozen 64-window/128-question campaign and independent audit finish. Local progress and partial rows are under `_local/qwen-qsa/quality/Original/`; no completed native Qwen QSA score is established yet."
        ]
    native_control_path = WORK / "native-gqa/full-model-validation.json"
    if native_control_path.exists():
        native_control = load(native_control_path)
        kernel_control = load(WORK / "native-gqa/kernel-validation.json")
        operator_control = load(WORK / "native-gqa/operator-validation.json")
        assert all(
            control["status"] == "passed"
            for control in [native_control, kernel_control, operator_control]
        )
        lines += [
            "",
            "## Native Qwen integration controls",
            "",
            "The isolated Qwen native build passes 611 supported compiler tests (nine unsupported) and three compiler/runtime unit tests. Frozen GPU features agree with the canonical CPU fixture within 0.004% relative RMS; tested block IDs and stable ties match exactly. Incremental pools match full prefill exactly. Dedicated batch-four/eight feature and long-index controls qualify those kernels, not full-model batch/long-context feasibility.",
            "",
            "The complete native operator, including hipBLASLt BF16 projection and dense-to-sparse cache transition, matches the independent frozen CPU attention oracle within 0.030% relative RMS. Full-model Original/Changed outputs are bit-exact on the tested full-support requests (prefixes 1/17/513/2047 plus four decodes). Sparse 4092+4 versus full prefill differs by 0.07771% relative RMS and replays exactly; first-pruning continuity differs by 0.10323%. ORT profiles verify one EP node dispatch per forward: 20 Original and 39 Changed calls, with unchanged cache addresses.",
            "",
            "The validated implementation is committed as `7a71b72` on `codex/qwen-native-qsa`. Both timing variants use the isolated Qwen build and identical backbone bytes. The earlier fresh default remains preserved. Matched quality is independently evaluated below; numerical controls do not substitute for dataset accuracy.",
            "",
            "| Metric | Matched native default | Matched native QSA |",
            "|---|---:|---:|",
        ]

        def native_cell(variant, metric):
            item = native_quality[variant]
            if metric == "ppl" and item["ppl_documents"] == 64:
                return f"{item['ppl']:.6f}"
            if metric == "gsm" and item["gsm_questions"] == 128:
                return f"{item['gsm_correct']}/128 ({100 * item['gsm_correct'] / 128:.2f}%)"
            return "Pending"

        lines += [
            f"| PPL, 64 × 4K | {native_cell('Original', 'ppl')} | {native_cell('Changed', 'ppl')} |",
            f"| GSM8K, 128 questions | {native_cell('Original', 'gsm')} | {native_cell('Changed', 'gsm')} |",
        ]
        if comparison["state"] == "complete":
            assert comparison["ppl_gate_passed"] and comparison["gsm_gate_passed"]
            lines += [
                "",
                f"**Matched quality gates pass:** PPL changes by {comparison['ppl_change_percent']:+.6f}% against the 5% limit; GSM changes by {comparison['gsm_change_percentage_points']:+.2f} percentage points against the 2-point drop limit. Default/QSA retain {native_quality['Original']['gsm_caps']}/{native_quality['Changed']['gsm_caps']} capped generations. Generated GSM token sequences differ on {comparison['gsm_token_sequence_differences']}/128 questions. This GSM pilot stays entirely within the native full-support dense path; it does not measure quality after sparse pruning. The 4K PPL test exercises sparse pruning.",
            ]
    if performance:
        lines += [
            "",
            "## Matched native performance",
            "",
            "Full-session wall timings use two exact-shape warmup pairs and six balanced measured pairs, with a common 64-token teacher-forced continuation. Both variants include the same final-position output head and resident cache work. Model loading, input/output/binding allocation, recurrent-state zeroing and sampling are outside phase timers. Intervals are 10,000 paired bootstrap resamples and describe within-run variation.",
            "",
            "| Batch | Prefix | Phase | Default tok/s | QSA tok/s | Change, paired 95% interval |",
            "|---:|---:|---|---:|---:|---:|",
        ]
        for row in performance:
            for phase in ["prefill", "decode"]:
                values = row["phases"][phase]
                low, high = values["paired_95_interval_percent"]
                lines.append(
                    f"| {row['batch']} | {row['context'] // 1024}K | {phase} | {values['Original']['tokens_s']:.2f} | {values['Changed']['tokens_s']:.2f} | {values['throughput_change_percent']:+.2f}% [{low:+.2f}, {high:+.2f}] |"
                )
        lines += [
            "",
            "Every listed row has finite outputs, exact warmup replay, unchanged sessions/cache addresses, and independently audited paired raw trials and statistics. Shapes without a completed receipt remain unqualified.",
        ]
    failed_64k_path = ROOT / "_local/qsa/stages/qwen-native-perf-65536-b1/status.json"
    if failed_64k_path.exists() and load(failed_64k_path)["state"] == "failed":
        lines += [
            "",
            "**64K has no valid timing result.** The co-resident two-session attempt stopped during warmup when OS available RAM remained below the 2 GiB guard. Its memory trace and failure receipt are retained. Further 64K work was excluded at the user's request. The [separate default/QSA memory-only matrix](qwen-qsa-memory.md) covers batches 1/4/8 and prefixes 4K/16K/32K with one prefill and eight decode calls per isolated process, without rerunning the performance benchmark.",
        ]
    if memory:
        lines += [
            "",
            "## Earlier paired-run application memory",
            "",
            "A separate CPU-only sampler reads OS process/system and Windows PDH GPU memory gauges every second without creating a GPU context. The worker holds Original and Changed sessions together, sharing KV serially; these are joint application footprints, not isolated per-variant allocations. The 32K trace attaches mid-run; 64K sampling starts before application startup. Sampled peaks can miss sub-second spikes.",
            "",
            "| Prefix | Coverage | Peak process RSS GiB | Peak private commitment GiB | Peak WDDM dedicated usage GiB | Peak WDDM shared usage GiB | Minimum system RAM available GiB |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        for row in memory["rows"]:
            fields = [
                "peak_sampled_rss_tree_bytes",
                "peak_sampled_private_commit_tree_bytes",
                "peak_sampled_wddm_dedicated_usage_bytes",
                "peak_sampled_wddm_shared_usage_bytes",
                "minimum_system_ram_available_bytes",
            ]
            cells = [
                f"{row[field] / 2**30:.2f}" if row[field] is not None else "Unavailable"
                for field in fields
            ]
            lines.append(
                f"| {row['context'] // 1024}K | {row['coverage']} | "
                + " | ".join(cells)
                + " |"
            )
        lines += [
            "",
            "WDDM values are driver-reported usage/commitment; they are not a proven physical-residency total. Process RSS, private commitment, GPU dedicated/shared gauges and system RAM overlap and must not be added. Raw timestamped samples, OS lifetime high-water counters, per-request completion markers and immutable source/snapshot identities are retained under `_local/qwen-qsa/memory-v1/`. Memory sampling begins after the earlier uninstrumented trials and partway through 32K; 64K has the observer throughout both variants.",
        ]
    lines += [
        "",
        "## Separate locked long-context inputs",
        "",
        "Before native QSA integration or kernel tuning, four natural documents from the pinned Dolma PDF subcorpus were selected with seed 20261124, one per new shard. Prior Qwen selector-data shards and all train/validation/test document hashes are excluded. The bounded selection also rejects shared 16-token fingerprints with the 128 GSM questions. This is not a semantic contamination guarantee or a corpus-representative sample.",
        "",
        f"Each document supplies nested 16K and 32K prefix trials, scoring the final 1,023 next-token targets at each length. The four locked windows contain 32,896 tokens each, including a 128-token reserve. The document file SHA256 is `{long_manifest['documents_sha256']}`. These suffix scores are reported separately from the 4K and GSM diagnostics.",
        "",
        f"The timing inputs are separately frozen under `_local/qwen-qsa/timing-v1/`, SHA256 `{timing['streams_sha256']}`. Eight common 65,600-token streams concatenate the earlier locked 4K test windows. They support candidate prefix lengths 4K/16K/32K/64K with 64 teacher-forced decode calls, and candidate batches 1/4/8. The protocol declares two warmup pairs and six measured alternating pairs. These are candidate shapes rather than claims of memory/support, and these concatenations are timing inputs rather than natural long-document quality evidence.",
    ]
    if long_quality:
        lines += [
            "",
            "| Natural-document prefix | Default suffix PPL | Native trained QSA |",
            "|---|---:|---|",
        ]
        for row in long_quality:
            lines.append(
                f"| {row['context'] // 1024}K, four documents / {row['targets']:,} targets | {row['ppl']:.6f} | Pending |"
            )
        lines += [
            "",
            "All listed long-context default outputs were finite and pass document, protocol, row-count and target-count audits. This four-document diagnostic is not a corpus-wide accuracy claim.",
        ]
    if any(native_long_quality.values()):
        lines += [
            "",
            "| Matched natural-document prefix | Default suffix PPL | QSA suffix PPL | Change |",
            "|---|---:|---:|---:|",
        ]
        for context in [16384, 32768]:
            values = {
                variant: next((r for r in rows if r["context"] == context), None)
                for variant, rows in native_long_quality.items()
            }
            cells = {
                variant: f"{row['ppl']:.6f}" if row else "Pending"
                for variant, row in values.items()
            }
            change = (
                f"{100 * (values['Changed']['ppl'] / values['Original']['ppl'] - 1):+.4f}%"
                if all(values.values())
                else "Pending"
            )
            lines.append(
                f"| {context // 1024}K, four documents / 4,092 targets | {cells['Original']} | {cells['Changed']} | {change} |"
            )
        lines += [
            "",
            "Matched long-context scores use the same native binaries and unchanged backbone bytes, with all frozen document identities, output finiteness and suffix target counts audited. Four nested documents are a diagnostic, with no 64K quality or population-wide claim.",
        ]
    lines += [
        "",
        "## Updated MI355X trained reference",
        "",
        "Source job 303195 now times the exact trained step-128 export used for source accuracy, with an unchanged backbone and six unchanged inference/kernel sources. Source packet commit is `3b0c30ea4976085798a36d22cefa4089bbef6543`. This evidence update does not change the transferred selector or native quality protocol. Selected batch-one median paired changes are:",
        "",
        "| Source prefix | Source decode TPS change | Source prefill TPS change |",
        "|---|---:|---:|",
    ]
    for row in trained["rows"]:
        if row["batch"] == 1:
            lines.append(
                f"| {row['context'] // 1024}K | {row['decode']['paired_throughput_gain_pct']:+.2f}% | {row['prefill']['paired_throughput_gain_pct']:+.2f}% |"
            )
    lines += [
        "",
        "These are BF16 MI355X results, not hip-ep measurements or a forecast. The source has two warmup pairs and six measured alternating pairs per shape. Its independent wall phase timers include per-step greedy argmax in decode, and variants can follow different token streams. That protocol remains distinct from the planned native fixed-token timing. Full source tables, raw-trial linkage and ranges are in the read-only `../sparse-attn/QWEN_PERFORMANCE.md` and `docs/reproducibility/qwen-trained-performance.json`. Historical untrained timing remains separate.",
    ]
    lines += [
        "",
        "## Quality and performance work remaining",
        "",
        "The native wave32 integration and feature/index/cache/numerical controls above are complete. Only layers 27/31/35/39 are replaced, using the frozen Qwen projection, three zero-centered norms, token-key normalization before four-token FP32 pooling, partial 64-coordinate RoPE, stable 512 historical-block selection and one causal current block. Full KV storage, selected-address reads and all Gated DeltaNet/default work are preserved. The matched 4K PPL/GSM campaign is complete; long-document quality is qualified only where completed matched rows appear above.",
        "",
        "Prefill and decode use independent fixed-token phase timers, identical session/head/cache work, exact-shape warmups and balanced paired trials. Only completed, audited shapes in the performance table establish measured throughput or support; remaining batch/long-context candidates require actual memory and correctness checks.",
        "",
        "## Evidence and reproduction",
        "",
        "Scripts are under `scripts/qwen_qsa/`. CPU identities/support receipts and export/conversion manifests are under `_local/qwen-qsa/`. Bounded stages use the same serial GPU lock as OLMo, with retained command/status logs under `_local/qsa/stages/qwen-*`. Four-block support and numerical receipts retain binary, graph, validator and runtime identities. `report_results.py` audits complete quality rows before reporting a full default score.",
        "",
        "The [native Qwen GQA design contract](../design/qwen-qsa-native-gqa.md) records the replacement boundary, precision, cache ownership, selection semantics and validation/timing requirements before implementation.",
        "",
        "See the preserved [OLMo native accuracy report](olmo-qsa-agent-accuracy.md) and [OLMo native performance report](olmo-qsa-native-gqa-results.md). GSM/BFCL pilots are complete; SWE-bench remains blocked by the missing container host.",
        "",
    ]
    output = ROOT / "docs/experiments/qwen-qsa-native-gqa-results.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    (WORK / "report.json").write_text(
        json.dumps(
            {
                "default_quality": quality,
                "matched_native_quality": native_quality,
                "matched_native_comparison": comparison,
                "matched_native_performance": performance,
                "application_memory": memory,
                "default_long_quality": long_quality,
                "matched_native_long_quality": native_long_quality,
                "native_qsa_results": comparison["state"] == "complete",
                "baseline_established": quality["state"] == "complete",
                "long_quality_manifest_sha256": sha(long_manifest_path),
                "timing_manifest_sha256": sha(timing_path),
                "trained_source_evidence_sha256": sha(trained_path),
                "trained_source_raw_sha256": trained["raw_sha256"],
                "report_sha256": sha(output),
            },
            indent=2,
        )
        + "\n"
    )
    print(output)


if __name__ == "__main__":
    main()
