#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Serve serial frozen native generations to a file queue shared with WSL."""
# isort: skip_file
# Initialize the selected native EP before importing GPU libraries.
# ruff: noqa: E402

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

TASK = Path(__file__).resolve().parents[2] / "_local/qsa/native-gqa"
os.environ["QSA_HIP_EP_INSTALL"] = str(TASK / "install")
os.environ["QSA_MODEL_CACHE"] = str(TASK / "model-cache")
os.environ.pop("QSA_ORT_PROFILE_DIR", None)

import runtime
import torch
from transformers import AutoTokenizer

from original_inference import Original
from prepare_baseline import SELECTOR, SELECTOR_SHA, WORK, sha


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=True) + "\n", encoding="utf-8")
    share_retry(lambda: temporary.replace(path))


def share_retry(operation):
    """Wait briefly for WSL/Windows to release a just-published file handle."""
    deadline = time.monotonic() + 10
    while True:
        try:
            return operation()
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.05)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--capacity", type=int, default=65536)
    args = parser.parse_args()
    assert runtime.INSTALL == TASK / "install"
    assert 1 <= args.capacity <= 65536
    assert sha(SELECTOR) == SELECTOR_SHA
    torch.set_num_threads(8)
    args.queue.mkdir(parents=True, exist_ok=True)
    assert not (args.queue / "ready.json").exists(), "Use a fresh stage queue"
    tokenizer = AutoTokenizer.from_pretrained(
        WORK / "base-model", local_files_only=True
    )
    original = Original(TASK / "models/Original", args.capacity)
    changed = Original(TASK / "models/Changed", args.capacity, caches=original.caches)
    models = {"Original": original, "Changed": changed}
    pointers = [v.data_ptr() for v in original.caches.values()]
    write_json(
        args.queue / "ready.json",
        {
            "pid": os.getpid(),
            "capacity": args.capacity,
            "selector_sha256": SELECTOR_SHA,
            "training_updates": 0,
            "sampling": "greedy",
            "has_chat_template": bool(tokenizer.chat_template),
        },
    )
    idle = time.monotonic()
    while not (args.queue / "stop.json").exists():
        pending = sorted(args.queue.glob("*.request.json"))
        if not pending:
            if time.monotonic() - idle > 600:
                raise TimeoutError("No benchmark request for ten minutes")
            time.sleep(0.1)
            continue
        request_path = pending[0]
        request = json.loads(
            share_retry(lambda: request_path.read_text(encoding="utf-8"))
        )
        response_path = args.queue / (request["request_id"] + ".response.json")
        try:
            variant = request["variant"]
            assert variant in models
            prompt = request["prompt"]
            encoded = tokenizer(
                prompt, return_tensors="pt", add_special_tokens=False
            ).input_ids.cuda()
            length = encoded.shape[1]
            assert 0 < length <= args.capacity - 2, (
                "Prompt exceeds native context capacity"
            )
            maximum = min(request["max_new_tokens"], args.capacity - length - 2)
            assert 0 < maximum <= 4096
            assert request.get("temperature", 0) == 0
            stops = request.get("stop", [])
            assert all(isinstance(s, str) and s for s in stops)
            started = time.perf_counter()
            logits = models[variant].forward(encoded)
            tokens, reason = [], "length"
            for step in range(maximum):
                assert torch.isfinite(logits).all(), "Non-finite native logits"
                token = int(logits[:, -1].argmax(-1))
                tokens.append(token)
                if token == tokenizer.eos_token_id:
                    reason = "eos"
                    break
                text = tokenizer.decode(tokens, skip_special_tokens=True)
                if any(s in text for s in stops):
                    reason = "stop"
                    break
                if step + 1 < maximum:
                    logits = models[variant].forward(
                        torch.tensor([[token]], device="cuda", dtype=torch.int64),
                        past=length + step,
                    )
            text = tokenizer.decode(tokens, skip_special_tokens=True)
            if reason == "stop":
                text = text[: min(text.index(s) for s in stops if s in text)]
            response = {
                "status": "complete",
                "request_id": request["request_id"],
                "variant": variant,
                "text": text,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "prompt_tokens": length,
                "generated_token_ids": tokens,
                "completion_tokens": len(tokens),
                "finish_reason": reason,
                "generation_seconds": time.perf_counter() - started,
            }
            assert sha(SELECTOR) == SELECTOR_SHA
            assert pointers == [v.data_ptr() for v in changed.caches.values()]
            write_json(response_path, response)
            share_retry(lambda: request_path.rename(request_path.with_suffix(".done")))
            print(variant, length, len(tokens), reason, flush=True)
            idle = time.monotonic()
        except Exception as exc:
            write_json(response_path, {"status": "failed", "error": repr(exc)})
            raise
    write_json(
        args.queue / "stopped.json",
        {"selector_sha256": sha(SELECTOR), "training_updates": 0},
    )


if __name__ == "__main__":
    main()
