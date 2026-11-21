#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Audit fresh default quality rows and render the native Qwen evidence record."""

import hashlib
import json
import math
from pathlib import Path

from evaluate_default import answer
from verify_handoff import ROOT, SOURCE, WORK, sha


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def audit_quality():
    folder = WORK / "quality/Original"
    if not (folder / "protocol.json").exists():
        return {"state": "not_started", "ppl_documents": 0, "gsm_questions": 0}
    protocol_path = folder / "protocol.json"
    protocol = load(protocol_path)
    exported = load(WORK / "default-rtn-w4a16-64/export.json")
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
            r["protocol_sha256"] == sha(protocol_path) and r["variant"] == "Original"
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


def main():
    verified = load(WORK / "handoff-verification.json")
    support = load(WORK / "native-support.json")
    probe = load(WORK / "default-4-support.json")
    reference = load(WORK / "default-4-reference.json")
    assert all(x["status"] == "passed" for x in [verified, support, probe, reference])
    quality = audit_quality()
    lines = [
        "<!--",
        "Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.",
        "Licensed under the MIT License.",
        "-->",
        "# Qwen trained-selector port: native default and QSA",
        "",
        "## Current evidence",
        "",
        "The verified frozen Qwen payload has been imported on CPU, and a four-block default architecture diagnostic passes native HIP and frozen Hugging Face numerical controls. Full 64-block default quality must be established before QSA integration. No native Qwen QSA accuracy or performance result is established yet.",
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
        f"The target is **{probe['gpu']}**, gfx1151/wave32, with {support['ram_total_bytes'] / 2**30:.2f} GiB physical shared memory. The native probe uses PyTorch {probe['torch']}, HIP {probe['hip']} and Transformers {support['transformers']}. The isolated `_local/qwen-qsa/install` copies the unchanged four hash-verified OLMo native compiler/runtime binaries; no Qwen kernel change or rebuild has occurred at this stage.",
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
    else:
        lines += [
            f"Default quality is **{quality['state'].replace('_', ' ')}**: {quality['ppl_documents']}/64 PPL windows and {quality['gsm_questions']}/128 GSM questions are recorded. No completed native Qwen default or QSA accuracy score is reported yet."
        ]
    lines += [
        "",
        "## QSA and performance work still required",
        "",
        "After default quality is established, integrate only layers 27/31/35/39 using the frozen Qwen-specific projection, three zero-centered norms, token-key normalization before four-token FP32 pooling, partial 64-coordinate RoPE, stable 512 historical-block selection and one causal current block. Preserve full KV storage with selected-address reads and all Gated DeltaNet/default work. The MI355X wave64 implementation needs a faithful wave32 port and dedicated feature/index/cache/numerical controls.",
        "",
        "Prefill and decode will be measured independently with a fixed common teacher-forced decode stream, identical session/head/cache work, exact-shape warmups, balanced paired trials and uncertainty. Start with feasible batch-one shapes, expanding only after actual support/memory tests. No native Qwen speedup, slowdown or feasible long-context/batch limit has been measured yet. Source timing with untrained selectors is not trained-export performance evidence.",
        "",
        "## Evidence and reproduction",
        "",
        "Scripts are under `scripts/qwen_qsa/`. CPU identities/support receipts and export/conversion manifests are under `_local/qwen-qsa/`. Bounded stages use the same serial GPU lock as OLMo, with retained command/status logs under `_local/qsa/stages/qwen-*`. Four-block support and numerical receipts retain binary, graph, validator and runtime identities. `report_results.py` audits complete quality rows before reporting a full default score.",
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
                "native_qsa_results": False,
                "report_sha256": sha(output),
            },
            indent=2,
        )
        + "\n"
    )
    print(output)


if __name__ == "__main__":
    main()
