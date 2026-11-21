#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Inference-only Original baseline on the frozen Dolma test and GSM8K pilot."""
# isort: skip_file

import argparse
import json
import math
import re
import time

from runtime import WORK
import torch
from transformers import AutoTokenizer

from original_inference import Original
from prepare_baseline import REFERENCE, SELECTOR, SELECTOR_SHA, original_folder, sha

TEST_SHA = "64b40ea48f14a0bda01e4700b3a7e91ba82f547a5cc7b20cf57f087bf20cf764"


def append(path, row):
    with path.open("a") as stream:
        stream.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)


def extract(text):
    # Exact strict #### parser from sparse-attn/scripts/train_selector.py.
    match = re.search(r"####\s*(-?[0-9.,]+)", text)
    return match.group(1).replace(",", "").rstrip(".") if match else None


def nll(logits, targets):
    total = 0.0
    for start in range(0, targets.shape[1], 128):
        block = logits[:, start : start + 128].float()
        gold = targets[:, start : start + 128]
        total += torch.nn.functional.cross_entropy(
            block.flatten(0, 1), gold.flatten(), reduction="sum"
        ).item()
    assert math.isfinite(total)
    return total


@torch.inference_mode()
def language(model, folder, limit, cached):
    path = REFERENCE / "data/selector-extended-v1/test.json"
    assert sha(path) == TEST_SHA
    documents = json.loads(path.read_text())
    assert len(documents) == 64
    rows_path = folder / "dolma.jsonl"
    previous = (
        [json.loads(line) for line in rows_path.read_text().splitlines()]
        if rows_path.exists()
        else []
    )
    done = {row["document"] for row in previous}
    for index, doc in enumerate(documents[:limit]):
        if index in done:
            continue
        started = time.perf_counter()
        ids = torch.tensor([doc["input_ids"]], dtype=torch.int64, device="cuda")
        logits = model.forward(ids, all_logits=True)
        loss = nll(logits[:, :-1], ids[:, 1:])
        del logits
        row = {
            "variant": "Original",
            "source": "fresh_test",
            "document": index,
            "document_id": doc["document_id"],
            "targets": 4095,
            "nll_sum": loss,
        }
        if cached:
            output = model.forward(ids[:, :4032].contiguous())
            losses, predictions = [], []
            for step in range(64):
                token = ids[:, 4032 + step]
                losses.append(
                    torch.nn.functional.cross_entropy(
                        output[:, -1].float(), token
                    ).item()
                )
                predictions.append(int(output[:, -1].argmax(-1)))
                if step < 63:
                    output = model.forward(
                        ids[:, 4032 + step : 4033 + step].contiguous(), past=4032 + step
                    )
            row.update(
                cached_targets=64,
                cached_nll_sum=sum(losses),
                cached_token_nll=losses,
                cached_predictions=predictions,
            )
        row["evaluation_seconds"] = time.perf_counter() - started
        append(rows_path, row)
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    summary = {
        "variant": "Original",
        "documents": len(rows),
        "required_documents": 64,
        "complete": len(rows) == 64,
        "nll": sum(r["nll_sum"] for r in rows) / sum(r["targets"] for r in rows),
    }
    summary["ppl"] = math.exp(summary["nll"])
    if cached:
        summary["cached_nll"] = sum(r["cached_nll_sum"] for r in rows) / sum(
            r["cached_targets"] for r in rows
        )
    (folder / "dolma-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


@torch.inference_mode()
def gsm(model, folder, limit):
    source = REFERENCE / "data/gsm8k.json"
    identity = json.loads((REFERENCE / "data/inputs.json").read_text())
    assert sha(source) == identity["sha256"]["gsm8k.json"]
    data = json.loads(source.read_text(encoding="utf-8"))
    assert len(data["items"]) == 128 and len(data["fewshot"]) == 5
    tokenizer = AutoTokenizer.from_pretrained(
        WORK / "base-model", local_files_only=True
    )
    prefix = "\n\n".join(
        f"Question: {x['question']}\nAnswer: {x['answer']}" for x in data["fewshot"]
    )
    rows_path = folder / "gsm8k.jsonl"
    previous = (
        [json.loads(line) for line in rows_path.read_text().splitlines()]
        if rows_path.exists()
        else []
    )
    done = {row["index"] for row in previous}
    for index, item in enumerate(data["items"][:limit]):
        if index in done:
            continue
        prompt = prefix + "\n\nQuestion: " + item["question"] + "\nAnswer:"
        ids = tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False
        ).input_ids.cuda()
        length = ids.shape[1]
        assert length + 512 <= model.capacity
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
        raw = tokenizer.decode(generated, skip_special_tokens=True)
        prediction, gold = (
            extract(raw.split("Question:", 1)[0]),
            extract(item["answer"]),
        )
        append(
            rows_path,
            {
                "variant": "Original",
                "id": item["id"],
                "index": index,
                "prompt": prompt,
                "completion": raw,
                "prediction": prediction,
                "gold": gold,
                "correct": prediction == gold,
                "prompt_tokens": length,
                "generated_tokens": len(generated),
                "hit_token_cap": len(generated) == 512,
                "evaluation_seconds": time.perf_counter() - started,
            },
        )
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    summary = {
        "variant": "Original",
        "items": len(rows),
        "complete": len(rows) == 128,
        "correct": sum(r["correct"] for r in rows),
    }
    summary["accuracy"] = summary["correct"] / summary["items"]
    (folder / "gsm8k-summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["dolma", "gsm8k"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--cached", action="store_true")
    parser.add_argument("--bits", type=int, choices=[4, 8], default=4)
    args = parser.parse_args()
    torch.set_num_threads(8)
    assert sha(SELECTOR) == SELECTOR_SHA
    original = original_folder(args.bits)
    assert json.loads((original / "validation.json").read_text())["status"] == "passed"
    folder = WORK / f"baseline-w{args.bits}-rope-v2"
    folder.mkdir(parents=True, exist_ok=True)
    protocol = {
        "training_updates": 0,
        "variant": "Original",
        "phase": args.phase,
        "model_export_sha256": sha(original / "export.json"),
        "fresh_test_sha256": TEST_SHA,
        "selector_sha256_unchanged": SELECTOR_SHA,
        "precision": f"RTN W{args.bits}A16 group128; DP4A disabled",
        "cached_loss": args.cached,
        "gsm8k": "reused 128-item pilot; five-shot greedy; strict #### parser; EOS/Question stop; cap512",
    }
    protocol_path = folder / (args.phase + "-protocol.json")
    if protocol_path.exists():
        assert json.loads(protocol_path.read_text()) == protocol, (
            "Cannot mix baseline protocols in one result directory"
        )
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    model = Original(original, 4608)
    if args.phase == "dolma":
        language(model, folder, args.limit or 64, args.cached)
    else:
        gsm(model, folder, args.limit or 128)


if __name__ == "__main__":
    main()
