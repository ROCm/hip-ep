#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Select suffix output-head rows without changing full-model context or caches."""
# ruff: noqa: E402

from native_inference import Native
from runtime import binding, run

import torch


class LongNative(Native):
    @torch.inference_mode()
    def suffix(self, ids, scored=1023):
        batch, count = ids.shape
        assert batch == self.batch and scored < count <= self.capacity
        for name, tensor in self.caches.items():
            if name.endswith((".conv_state", ".recurrent_state")):
                tensor.zero_()
        self.mask.zero_()
        self.mask[:, :count] = 1
        positions = torch.arange(count, device="cuda", dtype=torch.int64)
        positions = positions[None, None].expand(3, batch, -1).contiguous()
        selected = torch.arange(
            count - scored - 1, count - 1, device="cuda", dtype=torch.int64
        )
        logits = torch.empty(batch, scored, 248320, device="cuda", dtype=torch.float16)
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
        self.seen = count
        return logits
