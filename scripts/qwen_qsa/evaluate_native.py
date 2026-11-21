#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Run matched native Qwen Original/Changed PPL and GSM quality stages."""

import argparse
import hashlib
import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path

from verify_handoff import PAYLOAD_SHA, ROOT, SOURCE, WORK, sha


def answer(text):
    marked = re.findall(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)", text)
    numbers = marked or re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    if not numbers:
        return None
    try:
        return str(Decimal(numbers[-1].replace(",", "")).normalize())
    except InvalidOperation:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["Original", "Changed"], required=True)
    parser.add_argument("--task", choices=["ppl", "gsm8k"], required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    args = parser.parse_args()
    from native_inference import Native

    import torch
    import torch.nn.functional as functional
    from transformers import AutoTokenizer

    torch.set_num_threads(2)
    verified = json.loads((WORK / "handoff-verification.json").read_text())
    selector_path = (
        SOURCE / "checkpoints/hip-ep-qwen/303042-step-0128/selector-bf16.safetensors"
    )
    assert sha(selector_path) == PAYLOAD_SHA
    snapshot = Path(verified["backbone"]["snapshot"])
    directory = SOURCE / "data/qwen-selector-v1"
    for name in ["test.json", "gsm8k.json"]:
        assert sha(directory / name) == verified["data_sha256"][name]
    documents = json.loads((directory / "test.json").read_text(encoding="utf-8"))
    questions = json.loads((directory / "gsm8k.json").read_text(encoding="utf-8"))
    assert len(documents) == 64 and len(questions) == 128
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=False
    )
    prompts = [
        tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": row["question"]
                    + "\nSolve the problem and end with #### followed by the numerical answer.",
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for row in questions
    ]
    prompt_ids = [tokenizer(p, add_special_tokens=False).input_ids for p in prompts]
    assert max(map(len, prompt_ids)) + 512 <= 4096
    eos = json.loads((snapshot / "generation_config.json").read_text())["eos_token_id"]
    eos = {eos} if isinstance(eos, int) else set(eos)
    assert len(eos) == 2
    folder = WORK / "native-gqa/models" / args.variant
    exported = json.loads((folder / "export.json").read_text())
    assert exported["layers"] == 64
    assert sha(folder / "default.onnx") == exported["onnx_sha256"]
    assert sha(folder / "default.weights.bin") == exported["data_sha256"]
    diagnostic = json.loads((WORK / "default-4-reference.json").read_text())
    assert diagnostic["status"] == "passed"
    output = WORK / "native-gqa/quality" / args.variant
    output.mkdir(parents=True, exist_ok=True)
    validation = json.loads(
        (WORK / "native-gqa/full-model-validation.json").read_text()
    )
    assert validation["status"] == "passed"
    assert (
        json.loads((WORK / "native-gqa/operator-validation.json").read_text())["status"]
        == "passed"
    )
    binaries = validation["binary_sha256"]
    for name, expected in binaries.items():
        assert sha(WORK / "native-gqa/install/bin" / name) == expected
    protocol = {
        "variant": args.variant,
        "model": verified["model"],
        "revision": verified["revision"],
        "export_onnx_sha256": exported["onnx_sha256"],
        "export_data_sha256": exported["data_sha256"],
        "data_sha256": {
            name: verified["data_sha256"][name] for name in ["test.json", "gsm8k.json"]
        },
        "tokenizer_sha256": {
            name: sha(snapshot / name)
            for name in [
                "tokenizer.json",
                "tokenizer_config.json",
                "chat_template.jinja",
            ]
        },
        "eos_token_ids": sorted(eos),
        "max_new_tokens": 512,
        "thinking": False,
        "few_shot": 0,
        "sampling": "greedy",
        "gsm_prompt_max_tokens": max(map(len, prompt_ids)),
        "capacity": 4096,
        "batch": 1,
        "training_updates": 0,
        "selectors_loaded": args.variant == "Changed",
        "recurrent_state": "FP32",
        "ppl_window_tokens": 4096,
        "ppl_loss_chunk": 128,
        "quality_gates_for_later_matched_qsa": {
            "ppl_increase_percent": 5,
            "gsm_drop_percentage_points": 2,
        },
        "evaluator_sha256": sha(Path(__file__)),
        "inference_sha256": sha(Path(__file__).with_name("native_inference.py")),
        "common_runtime_sha256": sha(ROOT / "scripts/qsa/runtime.py"),
        "binary_sha256": binaries,
        "frozen_selector_sha256": PAYLOAD_SHA,
    }
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        assert json.loads(protocol_path.read_text()) == protocol
    else:
        protocol_path.write_text(
            json.dumps(protocol, indent=2) + "\n", encoding="utf-8"
        )
    protocol_sha = sha(protocol_path)
    path = output / (args.task + ".jsonl")
    existing = (
        [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        if path.exists()
        else []
    )
    assert len({row["index"] for row in existing}) == len(existing)
    assert all(row["protocol_sha256"] == protocol_sha for row in existing)
    done = {row["index"] for row in existing}
    data = documents if args.task == "ppl" else questions
    assert 0 <= args.start < args.start + args.count <= len(data)
    selected = [i for i in range(args.start, args.start + args.count) if i not in done]
    if not selected:
        print(
            "All requested rows already complete under the same frozen protocol",
            flush=True,
        )
        return
    print("Loading 64-block native", args.variant, flush=True)
    model = Native(folder, capacity=4096)
    print(args.variant, "ready", flush=True)
    with torch.inference_mode():
        for index in selected:
            if args.task == "ppl":
                document = documents[index]
                ids = torch.tensor(
                    [document["input_ids"]], device="cuda", dtype=torch.int64
                )
                assert ids.shape[1] == 4096
                logits = model.forward(ids, all_logits=True)
                assert torch.isfinite(logits).all()
                loss = 0.0
                for start in range(0, ids.shape[1] - 1, 128):
                    end = min(start + 128, ids.shape[1] - 1)
                    loss += functional.cross_entropy(
                        logits[:, start:end].float().reshape(-1, logits.shape[-1]),
                        ids[:, start + 1 : end + 1].reshape(-1),
                        reduction="sum",
                    ).item()
                row = {
                    "index": index,
                    "document_id": document["document_id"],
                    "nll_sum": loss,
                    "targets": ids.shape[1] - 1,
                }
                del logits
            else:
                row = questions[index]
                ids = torch.tensor(
                    [prompt_ids[index]], device="cuda", dtype=torch.int64
                )
                logits = model.forward(ids)
                generated = []
                for step in range(512):
                    assert torch.isfinite(logits).all()
                    token = logits[:, -1].argmax(-1, keepdim=True)
                    value = int(token.item())
                    generated.append(value)
                    if value in eos or step == 511:
                        break
                    logits = model.forward(token, past=len(prompt_ids[index]) + step)
                text = tokenizer.decode(generated, skip_special_tokens=True)
                predicted, gold = answer(text), answer(row["answer"])
                row = {
                    "index": index,
                    "question": row["question"],
                    "gold_answer": row["answer"],
                    "prompt": prompts[index],
                    "prompt_sha256": hashlib.sha256(
                        prompts[index].encode()
                    ).hexdigest(),
                    "prompt_tokens": len(prompt_ids[index]),
                    "generated_token_ids": generated,
                    "completion": text,
                    "predicted": predicted,
                    "gold": gold,
                    "correct": predicted is not None and predicted == gold,
                    "stopped_on_eos": generated[-1] in eos,
                    "reached_cap": generated[-1] not in eos and len(generated) == 512,
                }
            row.update(
                variant=args.variant, protocol_sha256=protocol_sha, training_updates=0
            )
            with path.open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                )
            print(
                args.task,
                index,
                {k: row[k] for k in ["nll_sum", "correct", "reached_cap"] if k in row},
                flush=True,
            )
    assert sha(selector_path) == PAYLOAD_SHA


if __name__ == "__main__":
    main()
