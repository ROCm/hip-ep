#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Matched full-session Qwen inference using the isolated Qwen native build."""
# ruff: noqa: E402

import json
import os
import sys
from pathlib import Path

from verify_handoff import sha

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "_local/qwen-qsa"
os.environ["QSA_HIP_EP_INSTALL"] = str(WORK / "native-gqa/install")
os.environ["QSA_MODEL_CACHE"] = str(WORK / "native-gqa/model-cache")
sys.path.insert(0, str(ROOT / "scripts/qsa"))
from runtime import binding, run, session

import torch


class Native:
    def __init__(self, folder, capacity, batch=1, caches=None):
        self.folder, self.capacity, self.batch = folder, capacity, batch
        self.export = json.loads((folder / "export.json").read_text())
        assert sha(folder / "default.onnx") == self.export["onnx_sha256"]
        if self.export["variant"] == "Changed":
            assert (
                sha(folder / "qsa.selectors.bin") == self.export["selector_data_sha256"]
            )
        self.layers = self.export["layers"]
        self.sess = session(folder / "default.onnx")
        self.caches = {} if caches is None else caches
        if caches is None:
            for layer in range(self.layers):
                if layer % 4 == 3:
                    for kind in ["key", "value"]:
                        self.caches[f"past_key_values.{layer}.{kind}"] = torch.empty(
                            batch, 4, capacity, 256, device="cuda", dtype=torch.float16
                        )
                else:
                    self.caches[f"past_key_values.{layer}.conv_state"] = torch.empty(
                        batch, 10240, 3, device="cuda", dtype=torch.float16
                    )
                    self.caches[f"past_key_values.{layer}.recurrent_state"] = (
                        torch.empty(
                            batch, 48, 128, 128, device="cuda", dtype=torch.float32
                        )
                    )
        self.mask = torch.empty(batch, capacity, device="cuda", dtype=torch.int64)
        expected = set(self.caches) | {
            "input_ids",
            "position_ids",
            "attention_mask",
            "logits_positions",
        }
        assert {x.name for x in self.sess.get_inputs()} == expected
        self.seen = None

    @torch.inference_mode()
    def forward(self, ids, past=0, all_logits=False):
        batch, count = ids.shape
        assert batch == self.batch and count > 0 and past + count <= self.capacity
        if past == 0:
            for name, tensor in self.caches.items():
                if name.endswith((".conv_state", ".recurrent_state")):
                    tensor.zero_()
        else:
            assert self.seen == past, (self.seen, past)
        self.mask.zero_()
        self.mask[:, : past + count] = 1
        positions = torch.arange(past, past + count, device="cuda", dtype=torch.int64)
        positions = positions[None, None].expand(3, batch, -1).contiguous()
        selected = (
            torch.arange(count, device="cuda", dtype=torch.int64)
            if all_logits
            else torch.tensor([count - 1], device="cuda", dtype=torch.int64)
        )
        logits = torch.empty(
            batch, len(selected), 248320, device="cuda", dtype=torch.float16
        )
        inputs = {
            **self.caches,
            "input_ids": ids,
            "position_ids": positions,
            "attention_mask": self.mask,
            "logits_positions": selected,
        }
        outputs = {
            "logits": logits,
            **{
                name.replace("past_key_values.", "present."): tensor
                for name, tensor in self.caches.items()
            },
        }
        run(self.sess, binding(self.sess, inputs, outputs))
        self.seen = past + count
        return logits
