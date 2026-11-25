#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Matched Original/Changed inference accuracy with the exact frozen data/parser."""
# isort: skip_file

import argparse
import json
import math
import time

from runtime import WORK
import numpy as np
import torch
from transformers import AutoTokenizer

from evaluate_original import TEST_SHA, extract, nll
from partitioned_inference import PartitionedOriginal
from prepare_baseline import REFERENCE, SELECTOR_SHA, original_folder, sha

VARIANTS = ["Original", "Changed"]


def write_row(path, row):
    with path.open("a") as stream:
        stream.write(json.dumps(row) + "\n")


def read_rows(path):
    return (
        [json.loads(line) for line in path.read_text().splitlines()]
        if path.exists()
        else []
    )


def order(index):
    return VARIANTS if index % 2 == 0 else VARIANTS[::-1]


@torch.inference_mode()
def dolma(model, folder, limit, validate_baseline=True):
    source = REFERENCE / "data/selector-extended-v1/test.json"
    assert sha(source) == TEST_SHA
    documents = json.loads(source.read_text())
    path = folder / "dolma.jsonl"
    done = {(r["variant"], r["document"]) for r in read_rows(path)}
    baseline = {
        r["document"]: r for r in read_rows(WORK / "baseline-w4-rope-v2/dolma.jsonl")
    }
    for index, document in enumerate(documents[:limit]):
        ids = torch.tensor([document["input_ids"]], device="cuda", dtype=torch.int64)
        for variant in order(index):
            if (variant, index) in done:
                continue
            model.variant = variant
            started = time.perf_counter()
            logits = model.forward(ids, all_logits=True)
            loss = nll(logits[:, :-1], ids[:, 1:])
            del logits
            output = model.forward(ids[:, :4032].contiguous())
            losses, predictions = [], []
            for step in range(64):
                losses.append(
                    torch.nn.functional.cross_entropy(
                        output[:, -1].float(), ids[:, 4032 + step]
                    ).item()
                )
                predictions.append(int(output[:, -1].argmax(-1)))
                if step < 63:
                    output = model.forward(
                        ids[:, 4032 + step : 4033 + step].contiguous(), past=4032 + step
                    )
            row = {
                "variant": variant,
                "source": "fresh_test",
                "document": index,
                "document_id": document["document_id"],
                "targets": 4095,
                "nll_sum": loss,
                "cached_targets": 64,
                "cached_nll_sum": sum(losses),
                "cached_token_nll": losses,
                "cached_predictions": predictions,
                "evaluation_seconds": time.perf_counter() - started,
            }
            if variant == "Original":
                row["nll_drift_from_established_baseline"] = (
                    loss - baseline[index]["nll_sum"]
                )
                if validate_baseline:
                    assert abs(row["nll_drift_from_established_baseline"]) < 0.01
            write_row(path, row)
            print(
                "Dolma",
                index + 1,
                variant,
                "PPL",
                round(math.exp(loss / 4095), 6),
                flush=True,
            )
    rows = read_rows(path)
    summary = {"complete": len(rows) == 128, "variants": {}}
    for variant in VARIANTS:
        selected = [r for r in rows if r["variant"] == variant]
        loss = sum(r["nll_sum"] for r in selected) / sum(r["targets"] for r in selected)
        summary["variants"][variant] = {
            "documents": len(selected),
            "nll": loss,
            "ppl": math.exp(loss),
            "cached_nll": sum(r["cached_nll_sum"] for r in selected)
            / sum(r["cached_targets"] for r in selected),
        }
    paired = {
        v: {r["document"]: r["nll_sum"] / 4095 for r in rows if r["variant"] == v}
        for v in VARIANTS
    }
    indices = sorted(set(paired["Original"]) & set(paired["Changed"]))
    delta = np.array([paired["Changed"][i] - paired["Original"][i] for i in indices])
    rng = np.random.default_rng(42)
    samples = rng.choice(delta, size=(10000, len(delta)), replace=True).mean(1)
    summary["ppl_change_percent"] = float(100 * np.expm1(delta.mean()))
    summary["paired_95_interval_percent"] = (
        100 * np.expm1(np.quantile(samples, [0.025, 0.975]))
    ).tolist()
    summary["gate_limit_percent"] = 5
    summary["gate_point_estimate_pass"] = summary["ppl_change_percent"] <= 5
    (folder / "dolma-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


@torch.inference_mode()
def gsm(model, folder, limit):
    path = folder / "gsm8k.jsonl"
    source = REFERENCE / "data/gsm8k.json"
    identity = json.loads((REFERENCE / "data/inputs.json").read_text())
    assert sha(source) == identity["sha256"]["gsm8k.json"]
    data = json.loads(source.read_text(encoding="utf-8"))
    assert len(data["items"]) == 128 and len(data["fewshot"]) == 5
    tokenizer = AutoTokenizer.from_pretrained(
        WORK / "base-model", local_files_only=True
    )
    prefix = "\n\n".join(
        f"Question: {r['question']}\nAnswer: {r['answer']}" for r in data["fewshot"]
    )
    done = {(r["variant"], r["index"]) for r in read_rows(path)}
    for index, item in enumerate(data["items"][:limit]):
        prompt = prefix + "\n\nQuestion: " + item["question"] + "\nAnswer:"
        ids = tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False
        ).input_ids.cuda()
        length = ids.shape[1]
        assert length + 512 <= model.capacity
        for variant in order(index):
            if (variant, index) in done:
                continue
            model.variant = variant
            started = time.perf_counter()
            output = model.forward(ids)
            generated = []
            for step in range(512):
                token = int(output[:, -1].argmax(-1))
                generated.append(token)
                recent = tokenizer.decode(
                    generated[max(0, len(generated) - 16) :], skip_special_tokens=True
                )
                if token == tokenizer.eos_token_id or "Question:" in recent:
                    break
                if step < 511:
                    output = model.forward(
                        torch.tensor([[token]], device="cuda", dtype=torch.int64),
                        past=length + step,
                    )
            text = tokenizer.decode(generated, skip_special_tokens=True)
            prediction, gold = (
                extract(text.split("Question:", 1)[0]),
                extract(item["answer"]),
            )
            write_row(
                path,
                {
                    "variant": variant,
                    "id": item["id"],
                    "index": index,
                    "prompt": prompt,
                    "completion": text,
                    "generated_token_ids": generated,
                    "prediction": prediction,
                    "gold": gold,
                    "correct": prediction == gold,
                    "prompt_tokens": length,
                    "generated_tokens": len(generated),
                    "hit_token_cap": len(generated) == 512,
                    "evaluation_seconds": time.perf_counter() - started,
                },
            )
            print(
                "GSM8K", index + 1, variant, "correct", prediction == gold, flush=True
            )
        model.release_prompt_buffers()
    rows = read_rows(path)
    summary = {"complete": len(rows) == 256, "variants": {}}
    for variant in VARIANTS:
        selected = [r for r in rows if r["variant"] == variant]
        correct = sum(r["correct"] for r in selected)
        summary["variants"][variant] = {
            "items": len(selected),
            "correct": correct,
            "accuracy_percent": 100 * correct / len(selected),
            "capped": sum(r["hit_token_cap"] for r in selected),
        }
    paired = {
        v: {r["index"]: int(r["correct"]) for r in rows if r["variant"] == v}
        for v in VARIANTS
    }
    indices = sorted(set(paired["Original"]) & set(paired["Changed"]))
    delta = np.array([paired["Changed"][i] - paired["Original"][i] for i in indices])
    rng = np.random.default_rng(42)
    samples = rng.choice(delta, size=(10000, len(delta)), replace=True).mean(1)
    summary["accuracy_change_percentage_points"] = float(100 * delta.mean())
    summary["paired_95_interval_percentage_points"] = (
        100 * np.quantile(samples, [0.025, 0.975])
    ).tolist()
    summary["acceptance_gate_enabled"] = False
    summary["role"] = "diagnostic"
    (folder / "gsm8k-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["dolma", "gsm8k"])
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    torch.set_num_threads(8)
    assert (
        json.loads((WORK / "changed-validation.json").read_text())["status"] == "passed"
    )
    folder = WORK / "paired-quality"
    folder.mkdir(exist_ok=True)
    protocol = {
        "phase": args.phase,
        "variants": VARIANTS,
        "batch": 1,
        "training_updates": 0,
        "backbone": "shared native RTN W4A16 group128; DP4A disabled",
        "selector_dtype": "BF16",
        "selector_sha256": SELECTOR_SHA,
        "backbone_export_sha256": sha(original_folder() / "export.json"),
        "kernel_library_sha256": sha(WORK / "native/qsa_kernels_gfx1151.dll"),
        "order": "alternate Original-first / Changed-first per document or item",
        "dolma_targets": "4095 prefill and 64 cached per document; prefix4032 + 63 calls",
        "gsm8k": "same reused 128-item five-shot pilot; greedy EOS/Question stop; strict #### parser; cap512",
    }
    protocol_path = folder / (args.phase + "-protocol.json")
    if args.phase == "gsm8k":
        protocol["source_text_encoding"] = "UTF-8"
    if protocol_path.exists():
        assert json.loads(protocol_path.read_text()) == protocol
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    model = PartitionedOriginal(4608)
    model.enable_qsa()
    try:
        if args.phase == "dolma":
            dolma(model, folder, args.limit or 64)
        else:
            gsm(model, folder, args.limit or 128)
    finally:
        (folder / (args.phase + "-integrity.json")).write_text(
            json.dumps(model.bank.integrity(), indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
