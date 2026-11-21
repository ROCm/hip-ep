#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Shared hip-ep backbone sessions with four controlled attention middles."""
# isort: skip_file

import json

from runtime import binding, run, session
import torch

from prepare_baseline import original_folder


class PartitionedOriginal:
    def __init__(self, capacity, batch=1):
        assert batch == 1, "This OLMo comparison is qualified for batch one only"
        self.folder = original_folder() / "partitioned"
        self.manifest = json.loads((self.folder / "manifest.json").read_text())
        self.capacity, self.batch, self.variant = capacity, batch, "Original"
        self.sessions = []
        for index, part in enumerate(self.manifest["parts"]):
            print("Loading shared Original part", index, part["kind"], flush=True)
            self.sessions.append(session(self.folder / part["file"]))
        self.caches = {
            f"past_{i}_{kind}": torch.empty(
                (batch, 32, capacity, 128), device="cuda", dtype=torch.float16
            )
            for i in range(32)
            for kind in ["key", "value"]
        }
        self.cap = torch.tensor(capacity, dtype=torch.int32, device="cuda")
        self.buffers, self.roles = {}, {"logits": "logits"}
        self.bank, self.qsa = None, {}
        for number, layer in enumerate(self.manifest["layers"]):
            boundary = self.manifest["boundaries"][str(layer)]
            for label in ["query", "key", "value"]:
                self.roles[boundary[label]] = label
            self.roles[boundary["hidden"]] = "residual_" + str(number % 2)
            self.roles[boundary["output"]] = "attention"

    def _output(self, info, count, logits_count):
        name = info["name"]
        if name.startswith("present_"):
            return self.caches[name.replace("present_", "past_", 1)]
        # Q/K/V buffers are reused between middles. Residuals alternate so a
        # segment never aliases an input and an output unintentionally.
        role = self.roles[name]
        dimensions = {
            "batch": self.batch,
            "sequence": count,
            "logits_sequence": logits_count,
            "cache_capacity": self.capacity,
        }
        shape = tuple(dimensions[d] if isinstance(d, str) else d for d in info["shape"])
        assert info["dtype"] == 10, "All controlled boundary outputs must be FP16"
        key = (role, shape)
        if key not in self.buffers:
            self.buffers[key] = torch.empty(shape, dtype=torch.float16, device="cuda")
        return self.buffers[key]

    @torch.inference_mode()
    def prepare(self, ids, past=0, all_logits=False):
        count = ids.shape[1]
        assert ids.shape[0] == self.batch and ids.is_contiguous()
        assert 0 <= past and past + count <= self.capacity
        positions = (
            torch.arange(count, device="cuda", dtype=torch.int64)
            if all_logits
            else torch.tensor([count - 1], device="cuda", dtype=torch.int64)
        )
        world = {
            **self.caches,
            "capacity": self.cap,
            "input_ids": ids,
            "position_ids": torch.arange(
                past, past + count, device="cuda", dtype=torch.int64
            )[None]
            .expand(self.batch, -1)
            .contiguous(),
            "seqlens_k": torch.full(
                (self.batch,), past + count - 1, device="cuda", dtype=torch.int32
            ),
            "logits_positions": positions,
        }
        jobs = []
        for index, (part, sess) in enumerate(
            zip(self.manifest["parts"], self.sessions, strict=True)
        ):
            inputs = {info["name"]: world[info["name"]] for info in part["inputs"]}
            outputs = {
                info["name"]: self._output(info, count, len(positions))
                for info in part["outputs"]
            }
            job = {
                "kind": part["kind"],
                "session": sess,
                "io": binding(sess, inputs, outputs),
                "inputs": inputs,
                "outputs": outputs,
            }
            if part["kind"] == "attention":
                layer = part["layer"]
                boundary = self.manifest["boundaries"][str(layer)]
                job.update(
                    layer=layer, hidden=world[boundary["hidden"]], boundary=boundary
                )
            jobs.append(job)
            world.update(outputs)
        return {
            "jobs": jobs,
            "owners": list(world.values()),
            "past": past,
            "count": count,
            "logits": world["logits"],
        }

    @torch.inference_mode()
    def invoke(self, plan):
        assert self.variant in ["Original", "Changed"]
        for job in plan["jobs"]:
            if self.variant == "Changed" and job["kind"] == "attention":
                assert self.qsa, "Enable and validate frozen QSA first"
                boundary = job["boundary"]
                torch.cuda.synchronize()
                self.qsa[job["layer"]].run(
                    job["hidden"],
                    job["inputs"][boundary["query"]],
                    job["inputs"][boundary["key"]],
                    job["inputs"][boundary["value"]],
                    job["outputs"][boundary["output"]],
                    plan["past"],
                )
                torch.cuda.synchronize()
            else:
                run(job["session"], job["io"])
        return plan["logits"]

    def enable_qsa(self):
        from qsa_runtime import FrozenBank, FrozenQSA

        assert not self.qsa
        self.bank = FrozenBank()
        self.qsa = {
            layer: FrozenQSA(
                self.bank,
                layer,
                self.capacity,
                self.caches[f"past_{layer}_key"],
                self.caches[f"past_{layer}_value"],
            )
            for layer in self.manifest["layers"]
        }

    def release_prompt_buffers(self):
        """Bound memory across independent, differently sized GSM requests."""
        self.buffers = {
            key: value
            for key, value in self.buffers.items()
            if len(key[1]) < 2 or key[1][1] == 1
        }

    @torch.inference_mode()
    def forward(self, ids, past=0, all_logits=False):
        return self.invoke(self.prepare(ids, past, all_logits))
