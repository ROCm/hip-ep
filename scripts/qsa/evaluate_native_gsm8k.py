#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Run the retained GSM8K pilot on matched single-session native models."""
# isort: skip_file
# Configure the native EP before importing Torch or ONNX Runtime.
# ruff: noqa: E402

import argparse
import json
import os
from pathlib import Path

TASK = Path(__file__).resolve().parents[2] / "_local/qsa/native-gqa"
os.environ["QSA_HIP_EP_INSTALL"] = str(TASK / "install")
os.environ["QSA_MODEL_CACHE"] = str(TASK / "model-cache")
os.environ.pop("QSA_ORT_PROFILE_DIR", None)

import runtime
import torch

from evaluate_pair import gsm
from original_inference import Original
from prepare_baseline import selector_fingerprint
from prepare_baseline import SELECTOR_SHA, sha


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=128)
    args = parser.parse_args()
    assert 1 <= args.limit <= 128
    assert runtime.INSTALL == TASK / "install"
    assert json.loads((TASK / "validation.json").read_text())["status"] == "passed"
    assert selector_fingerprint() == SELECTOR_SHA
    torch.set_num_threads(8)
    folder = TASK / "gsm8k"
    folder.mkdir(exist_ok=True)
    protocol = {
        "benchmark": "GSM8K retained 128-item five-shot pilot",
        "batch": 1,
        "capacity": 4608,
        "generation": "greedy; EOS/Question stop; 512-token cap; strict #### parser",
        "source_text_encoding": "UTF-8",
        "execution": "one native full-graph session per variant; one EP dispatch per forward",
        "weights": "identical RTN W4A16 group128 backbone; DP4A disabled",
        "selector_sha256": SELECTOR_SHA,
        "training_updates": 0,
        "acceptance_gate_enabled": False,
        "export_receipt_sha256": {
            v: sha(TASK / "models" / v / "export.json") for v in ["Original", "Changed"]
        },
    }
    target = folder / "protocol.json"
    if target.exists():
        assert json.loads(target.read_text()) == protocol
    target.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    original = Original(TASK / "models/Original", 4608)
    changed = Original(TASK / "models/Changed", 4608, caches=original.caches)
    models = {"Original": original, "Changed": changed}
    pointers = [v.data_ptr() for v in original.caches.values()]

    class Pair:
        variant = "Original"
        capacity = 4608

        def forward(self, *a, **kw):
            return models[self.variant].forward(*a, **kw)

        def release_prompt_buffers(self):
            # Original.forward retains no prompt-dependent tensor/binding cache.
            pass

    try:
        gsm(Pair(), folder, args.limit)
    finally:
        assert selector_fingerprint() == SELECTOR_SHA
        assert pointers == [v.data_ptr() for v in changed.caches.values()]
        (folder / "integrity.json").write_text(
            json.dumps(
                {
                    "selector_sha256": SELECTOR_SHA,
                    "training_updates": 0,
                    "cache_pointers_unchanged": True,
                    "exports_unchanged": all(
                        sha(TASK / "models" / v / "export.json") == digest
                        for v, digest in protocol["export_receipt_sha256"].items()
                    ),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
