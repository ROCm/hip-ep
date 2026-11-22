#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""One resident native Qwen variant; actual chat-template/token workload receipts."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from verify_handoff import WORK, sha

TASK = WORK / "bfcl-v4-50"


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
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=True, indent=2) + "\n")
    deadline = time.monotonic() + 10
    while True:
        try:
            temporary.replace(path)
            return
        except PermissionError:
            assert time.monotonic() < deadline
            time.sleep(0.05)


def prompt(tokenizer, messages):
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--variant", choices=["Original", "Changed"])
    parser.add_argument("--queue", type=Path)
    parser.add_argument("--capacity", type=int, default=32768)
    args = parser.parse_args()
    # Configure the native DLL environment before importing Torch/Transformers.
    from native_inference import Native
    import torch
    from transformers import AutoTokenizer

    torch.set_num_threads(2)
    snapshot = Path(load(WORK / "handoff-verification.json")["backbone"]["snapshot"])
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=False
    )
    assert tokenizer.chat_template
    eos = load(snapshot / "generation_config.json")["eos_token_id"]
    eos = {eos} if isinstance(eos, int) else set(eos)
    if args.profile:
        rows = []
        for case in load(TASK / "initial-messages.json"):
            text = prompt(tokenizer, case["messages"])
            ids = tokenizer(text, add_special_tokens=False).input_ids
            assert 0 < len(ids) <= args.capacity - 1026, (case["id"], len(ids))
            rows.append(
                {k: v for k, v in case.items() if k != "messages"}
                | {
                    "initial_prompt_tokens": len(ids),
                    "initial_prompt_characters": len(text),
                    "initial_prompt_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "initial_prefill_exercises_pruning": len(ids) > 2052,
                }
            )
        assert not torch.cuda.is_initialized()
        profile = {"cases": rows, "tokenizer_gpu_initialized": False}
        profile_path = TASK / "initial-profile.json"
        if profile_path.exists():
            assert load(profile_path) == profile, (
                "Preserve the frozen pre-run prompt profile"
            )
        else:
            write(profile_path, profile)
        print(
            "Profiled 50 initial prompts with the actual Qwen tokenizer; GPU uninitialized",
            flush=True,
        )
        return
    assert args.variant and args.queue
    assert not os.environ.get("QWEN_QSA_INSTALL"), "Use the preserved measured install"
    assert not os.environ.get("QWEN_QSA_MODEL_CACHE"), (
        "Use the preserved measured cache"
    )
    validation = load(WORK / "native-gqa/full-model-validation.json")
    for name, expected in validation["binary_sha256"].items():
        assert sha(WORK / "native-gqa/install/bin" / name) == expected, name
    model = Native(WORK / "native-gqa/models" / args.variant, args.capacity)
    pointers = {name: tensor.data_ptr() for name, tensor in model.caches.items()}
    write(
        args.queue / "ready.json",
        {
            "pid": os.getpid(),
            "variant": args.variant,
            "capacity": args.capacity,
            "binary_sha256": validation["binary_sha256"],
            "eos_token_ids": sorted(eos),
            "training_updates": 0,
        },
    )
    idle = time.monotonic()
    with torch.inference_mode():
        while not (args.queue / "stop.json").exists():
            assert not (TASK / "abort.json").exists(), "Campaign aborted"
            assert time.time() - load(TASK / "lease.json")["unix"] < 90, (
                "Supervisor lease expired"
            )
            pending = sorted(args.queue.glob("*.request.json"))
            if not pending:
                assert time.monotonic() - idle < 600, "No request for ten minutes"
                time.sleep(0.1)
                continue
            path = pending[0]
            request = load(path)
            assert request["variant"] == args.variant and request["temperature"] == 0
            assert request["max_new_tokens"] == 1024
            text = prompt(tokenizer, request["messages"])
            ids = tokenizer(text, add_special_tokens=False).input_ids
            length = len(ids)
            assert 0 < length <= args.capacity - 2, (
                "Context capacity exceeded; infrastructure failure, not wrong answer"
            )
            maximum = min(1024, args.capacity - length - 2)
            assert maximum > 0
            tokens, reason = [], "length"
            started = time.perf_counter()
            try:
                logits = model.forward(
                    torch.tensor([ids], dtype=torch.int64, device="cuda")
                )
                for step in range(maximum):
                    assert torch.isfinite(logits).all(), "Nonfinite logits"
                    token = int(logits[:, -1].argmax(-1))
                    tokens.append(token)
                    if token in eos:
                        reason = "eos"
                        break
                    if step + 1 < maximum:
                        logits = model.forward(
                            torch.tensor([[token]], device="cuda", dtype=torch.int64),
                            past=length + step,
                        )
                torch.cuda.synchronize()
                assert pointers == {
                    name: tensor.data_ptr() for name, tensor in model.caches.items()
                }
                response = dict(
                    status="complete",
                    request_id=request["request_id"],
                    variant=args.variant,
                    attempt_sha256=sha(args.queue / "attempt.json"),
                    case_id=request["case_id"],
                    category=request["category"],
                    generation_index=request["generation_index"],
                    prompt=text,
                    prompt_sha256=hashlib.sha256(text.encode()).hexdigest(),
                    request_sha256=sha(path),
                    prompt_tokens=length,
                    prompt_characters=len(text),
                    text=tokenizer.decode(tokens, skip_special_tokens=True),
                    generated_token_ids=tokens,
                    completion_tokens=len(tokens),
                    finish_reason=reason,
                    maximum_new_tokens=maximum,
                    context_budget_reduced=maximum < 1024,
                    input_plus_output_tokens=length + len(tokens),
                    generation_seconds=time.perf_counter() - started,
                    prefill_pruning_eligible=length > 2052,
                    pruning_eligible=length + len(tokens) - 2 >= 2052,
                    qsa_pruning_exercised=args.variant == "Changed"
                    and length + len(tokens) - 2 >= 2052,
                    message_role_counts={
                        role: sum(m["role"] == role for m in request["messages"])
                        for role in ["system", "user", "assistant", "tool"]
                    },
                    cache_pointers_unchanged=True,
                    finite=True,
                )
                # Publish the immutable request before its response so the CPU
                # scorer can immediately inspect a complete evidence pair.
                deadline = time.monotonic() + 10
                while True:
                    try:
                        path.rename(path.with_suffix(".done"))
                        break
                    except PermissionError:
                        assert time.monotonic() < deadline
                        time.sleep(0.05)
                write(args.queue / (request["request_id"] + ".response.json"), response)
                print(
                    args.variant,
                    request["case_id"],
                    length,
                    len(tokens),
                    reason,
                    flush=True,
                )
                idle = time.monotonic()
            except BaseException as exc:
                write(
                    args.queue / (request["request_id"] + ".response.json"),
                    {"status": "failed", "error": repr(exc)},
                )
                raise
    write(
        args.queue / "stopped.json",
        {"status": "complete", "variant": args.variant, "training_updates": 0},
    )


if __name__ == "__main__":
    main()
