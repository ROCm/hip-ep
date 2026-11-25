#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Original OLMo3 hip-ep inference and pinned-source numerical validation."""
# isort: skip_file

import argparse
import json
import time

from runtime import INSTALL, WORK, binding, run, session
import numpy as np
import torch
from transformers import AutoConfig, Olmo3ForCausalLM
from transformers.models.olmo3.modeling_olmo3 import Olmo3RotaryEmbedding

from export_original import Export, quantize
from prepare_baseline import REFERENCE, original_folder, sha


class Original:
    def __init__(self, folder, capacity, batch=1, caches=None):
        self.folder = folder
        self.export = json.loads((folder / "export.json").read_text())
        self.layers = self.export["layers"]
        self.capacity, self.batch = capacity, batch
        print(f"Creating strict Original session: {self.layers} layers", flush=True)
        self.sess = session(folder / "original.onnx")
        print("Original session ready", flush=True)
        self.caches = (
            caches
            if caches is not None
            else {
                f"past_{i}_{kind}": torch.empty(
                    (batch, 32, capacity, 128), device="cuda", dtype=torch.float16
                )
                for i in range(self.layers)
                for kind in ["key", "value"]
            }
        )
        assert len(self.caches) == 2 * self.layers
        assert all(
            t.shape == (batch, 32, capacity, 128) and t.dtype == torch.float16
            for t in self.caches.values()
        )
        self.lengths = torch.empty(batch, dtype=torch.int32, device="cuda")
        self.cap = torch.tensor(capacity, dtype=torch.int32, device="cuda")

    @torch.inference_mode()
    def forward(self, ids, past=0, all_logits=False):
        b, count = ids.shape
        assert b == self.batch and 0 <= past and past + count <= self.capacity
        self.lengths.fill_(past + count - 1)
        positions = (
            torch.arange(count, device="cuda", dtype=torch.int64)
            if all_logits
            else torch.tensor([count - 1], device="cuda", dtype=torch.int64)
        )
        logits = torch.empty(
            (b, len(positions), 100278), dtype=torch.float16, device="cuda"
        )
        inputs = {
            **self.caches,
            "input_ids": ids,
            "position_ids": torch.arange(
                past, past + count, device="cuda", dtype=torch.int64
            )[None]
            .expand(b, -1)
            .contiguous(),
            "seqlens_k": self.lengths,
            "capacity": self.cap,
            "logits_positions": positions,
        }
        outputs = {
            "logits": logits,
            **{
                name.replace("past_", "present_", 1): value
                for name, value in self.caches.items()
            },
        }
        io = binding(self.sess, inputs, outputs)
        run(self.sess, io)
        return logits


@torch.inference_mode()
def reference_model(folder, quantized=True):
    """Load the pinned model with this export's dequantized weights; no updates."""
    receipt = json.loads((folder / "export.json").read_text())
    config = AutoConfig.from_pretrained(WORK / "base-model", local_files_only=True)
    config.num_hidden_layers = receipt["layers"]
    config.layer_types = config.layer_types[: config.num_hidden_layers]
    config._attn_implementation = "sdpa"
    with torch.device("meta"):
        model = Olmo3ForCausalLM(config)
    model.model.rotary_emb = Olmo3RotaryEmbedding(config, device="cuda")
    # Reuse the exporter's read-only safetensors access without opening outputs.
    reader = object.__new__(Export)
    reader.weights = json.loads(
        (WORK / "base-model/model.safetensors.index.json").read_text()
    )["weight_map"]
    for name, parameter in list(model.named_parameters()):
        tensor = reader.source(name)
        if quantized and name.removesuffix(".weight") in receipt["projections"]:
            bits = receipt.get("bits", 4)
            packed, scales = quantize(tensor, bits=bits)
            if bits == 4:
                codes = np.empty((*packed.shape[:-1], packed.shape[-1] * 2), np.uint8)
                codes[..., ::2], codes[..., 1::2] = packed & 15, packed >> 4
            else:
                codes = packed
            tensor = torch.from_numpy(
                (
                    (codes.astype(np.float32) - (1 << (bits - 1)))
                    * scales.astype(np.float32)[..., None]
                ).reshape(parameter.shape)
            )
        parent, leaf = name.rsplit(".", 1)
        setattr(
            model.get_submodule(parent),
            leaf,
            torch.nn.Parameter(
                tensor.to(device="cuda", dtype=torch.float16), requires_grad=False
            ),
        )
        if name.endswith("post_feedforward_layernorm.weight"):
            print("Loaded reference", parent, flush=True)
    return model.eval().requires_grad_(False)


def stats(actual, expected):
    a, e = actual.float(), expected.float()
    return {
        "relative_l2": float((a - e).norm() / e.norm().clamp_min(1e-12)),
        "cosine": float(
            torch.nn.functional.cosine_similarity(a.flatten(), e.flatten(), dim=0)
        ),
        "max_abs": float((a - e).abs().max()),
        "finite": bool(a.isfinite().all()),
    }


@torch.inference_mode()
def validate(folder, source_reference, length=521):
    out = folder / "validation.json"
    receipt = {
        "status": "running",
        "install": str(INSTALL),
        "training_updates": 0,
        "dp4a": False,
        "cpu_fallback_disabled": True,
        "tokens": length,
        "checks": [],
    }
    started = time.perf_counter()
    try:
        assert 8 <= length <= 4096
        model = Original(folder, max(768, length + 64))
        tokens = json.loads(
            (REFERENCE / "data/selector-extended-v1/test.json").read_text()
        )[0]["input_ids"][:length]
        ids = torch.tensor([tokens], device="cuda", dtype=torch.int64)
        full = model.forward(ids, all_logits=True)
        assert full.isfinite().all()
        prefix_length = length - 4
        prefix = model.forward(ids[:, :prefix_length].contiguous())
        parts = [prefix]
        for pos in range(prefix_length, length):
            parts.append(model.forward(ids[:, pos : pos + 1].contiguous(), past=pos))
        cached = torch.cat(parts, dim=1)
        parity = stats(cached, full[:, prefix_length - 1 : length])
        receipt["checks"].append(
            {"name": f"full_vs_cached_{prefix_length}_plus_four", **parity}
        )
        assert parity["relative_l2"] < 0.02 and parity["cosine"] > 0.999, parity
        replay = model.forward(ids, all_logits=True)
        exact = torch.equal(full, replay)
        receipt["checks"].append({"name": "prefill_replay", "bit_exact": exact})
        assert exact
        if source_reference:
            reference = reference_model(folder)
            ref = reference(ids, use_cache=False).logits
            parity = stats(full, ref)
            receipt["checks"].append(
                {"name": "pinned_transformers_same_rtn_weights", **parity}
            )
            assert parity["relative_l2"] < 0.02 and parity["cosine"] > 0.999, parity
        receipt["artifacts"] = {
            str(p): sha(p)
            for p in [
                INSTALL / "bin/hipgpu.dll",
                INSTALL / "bin/custom_kernels_gfx1151.dll",
                folder / "original.onnx",
            ]
        }
        receipt["status"] = "passed"
    except BaseException as exc:
        receipt.update(status="failed", error=repr(exc))
        raise
    finally:
        receipt["elapsed_seconds"] = time.perf_counter() - started
        out.write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", type=int, default=32)
    parser.add_argument("--source-reference", action="store_true")
    parser.add_argument("--bits", type=int, choices=[4, 8], default=4)
    parser.add_argument("--length", type=int, default=521)
    args = parser.parse_args()
    torch.set_num_threads(8)
    folder = original_folder(args.bits, args.layers)
    validate(folder, args.source_reference, args.length)
