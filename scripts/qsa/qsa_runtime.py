#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Frozen BF16 selectors and staged, persistent-workspace QSA C ABI bindings."""
# isort: skip_file

import ctypes
import hashlib
import json

from runtime import ROOT, WORK
import numpy as np
import onnx
import torch

from prepare_baseline import SELECTOR_SHA, original_folder, selector_fingerprint, sha
from selector_assets import get_selector_asset

LAYERS = [7, 11, 15, 23]
CHUNK = 32


class KernelLibrary:
    def __init__(self):
        assert torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx1151")
        receipt = json.loads((WORK / "native/build.json").read_text())
        for path, digest in receipt["source_sha256"].items():
            assert sha(ROOT / path) == digest, "Rebuild stale QSA native sources"
        path = WORK / "native/qsa_kernels_gfx1151.dll"
        assert sha(path) == receipt["library_sha256"]
        self.library = ctypes.CDLL(str(path))
        p, i = ctypes.c_void_p, ctypes.c_int
        self.features = self.library.hip_qsa_features
        self.features.argtypes = [p] * 13 + [i] * 5 + [ctypes.c_float]
        self.select = self.library.hip_qsa_select
        self.select.argtypes = [p] * 5 + [i] * 6
        self.attend = self.library.hip_qsa_attend
        self.attend.argtypes = [p] * 8 + [i] * 6
        for function in [self.features, self.select, self.attend]:
            function.restype = i
        self.receipt = receipt


def tensor_digest(tensors):
    digest = hashlib.sha256()
    for name, value in sorted(tensors.items()):
        digest.update(name.encode())
        digest.update(
            value.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
        )
    return digest.hexdigest()


def rotary_tables():
    folder = original_folder()
    model = onnx.load(folder / "original.onnx", load_external_data=False)
    tables = {}
    for node in model.graph.initializer:
        if node.name not in ["cos_full_attention", "sin_full_attention"]:
            continue
        assert node.data_type == onnx.TensorProto.FLOAT
        metadata = {item.key: item.value for item in node.external_data}
        mapped = np.memmap(
            folder / metadata["location"],
            dtype=np.float32,
            mode="r",
            offset=int(metadata["offset"]),
            shape=tuple(node.dims),
        )
        tables[node.name] = torch.from_numpy(np.array(mapped, copy=True)).cuda()
    assert len(tables) == 2
    return tables["cos_full_attention"], tables["sin_full_attention"]


class FrozenBank:
    def __init__(self):
        self.asset = get_selector_asset("olmo-four")
        assert self.asset.spec["compatibility_sha256"] == SELECTOR_SHA
        assert self.asset.config["layers"] == LAYERS
        tensors = self.asset.tensors()
        assert all(t.dtype == torch.bfloat16 for t in tensors.values())
        self.tensors = {
            name: value.detach().cuda().contiguous().requires_grad_(False)
            for name, value in tensors.items()
        }
        assert tensor_digest(self.tensors) == tensor_digest(tensors)
        self.before = tensor_digest(self.tensors)
        self.cos, self.sin = rotary_tables()
        self.kernel = KernelLibrary()

    def weights(self, layer):
        assert layer in LAYERS
        return [
            self.tensors[f"selectors.{layer}.{name}.weight"]
            for name in ["projection", "q_norm", "k_norm"]
        ]

    def integrity(self):
        assert selector_fingerprint() == SELECTOR_SHA
        assert tensor_digest(self.tensors) == self.before
        assert all(
            t.dtype == torch.bfloat16 and not t.requires_grad and t.grad is None
            for t in self.tensors.values()
        )
        return {
            "export_sha256": SELECTOR_SHA,
            "selector_asset": self.asset.receipt(),
            "runtime_selector_digest": self.before,
            "parameters_unchanged": True,
            "selector_parameter_dtype": "BF16",
            "training_updates": 0,
        }


class FrozenQSA:
    def __init__(self, bank, layer, capacity, keys, values):
        self.bank, self.layer, self.capacity = bank, layer, capacity
        self.weight, self.q_weight, self.k_weight = bank.weights(layer)
        self.keys, self.values, self.seen = keys, values, 0
        assert keys.shape == values.shape == (1, 32, capacity, 128)
        assert keys.dtype == values.dtype == torch.float16
        self.pool_capacity = (capacity + 3) // 4
        self.hidden = torch.empty((capacity, 4096), dtype=torch.bfloat16, device="cuda")
        self.projected = torch.empty(
            (capacity, 640), dtype=torch.bfloat16, device="cuda"
        )
        self.features = torch.empty(
            (capacity, 4, 128), dtype=torch.bfloat16, device="cuda"
        )
        self.pool = torch.empty(
            (self.pool_capacity, 128), dtype=torch.bfloat16, device="cuda"
        )
        self.tail = torch.empty((4, 128), dtype=torch.bfloat16, device="cuda")
        self.scores = torch.empty(
            (CHUNK, self.pool_capacity), dtype=torch.float32, device="cuda"
        )
        self.selected = torch.empty((CHUNK, 128), dtype=torch.int32, device="cuda")
        self.partial = torch.empty(
            (CHUNK, 32, 8, 128), dtype=torch.float32, device="cuda"
        )
        self.stats = torch.empty((CHUNK, 32, 8, 2), dtype=torch.float32, device="cuda")
        self.workspace_pointers = self.pointers()
        self.capture = None

    def pointers(self):
        return [
            getattr(self, name).data_ptr()
            for name in [
                "hidden",
                "projected",
                "features",
                "pool",
                "tail",
                "scores",
                "selected",
                "partial",
                "stats",
                "keys",
                "values",
            ]
        ]

    @torch.inference_mode()
    def run(self, hidden, query, key, value, output, past):
        b, count, width = hidden.shape
        assert b == 1 and width == 4096 and past + count <= self.capacity
        assert past == 0 or (past == self.seen and count == 1)
        for tensor in [hidden, query, key, value, output]:
            assert (
                tensor.is_cuda
                and tensor.is_contiguous()
                and tensor.dtype == torch.float16
            )
            assert tensor.shape == hidden.shape
        self.hidden[:count].copy_(hidden[0])
        torch.mm(self.hidden[:count], self.weight.t(), out=self.projected[:count])
        stream = torch.cuda.current_stream().cuda_stream

        def ptr(*items):
            return [t.data_ptr() for t in items]

        status = self.bank.kernel.features(
            stream,
            *ptr(
                self.projected,
                self.q_weight,
                self.k_weight,
                self.bank.cos,
                self.bank.sin,
                self.features,
                self.pool,
                self.tail,
                key,
                value,
                self.keys,
                self.values,
            ),
            1,
            count,
            past,
            self.capacity,
            self.pool_capacity,
            1e-6,
        )
        if status:
            raise RuntimeError(f"QSA layer {self.layer} features HIP error {status}")
        for first in range(0, count, CHUNK):
            chunk = min(CHUNK, count - first)
            status = self.bank.kernel.select(
                stream,
                *ptr(self.features, self.pool, self.scores, self.selected),
                1,
                count,
                past,
                first,
                chunk,
                self.pool_capacity,
            )
            if status:
                raise RuntimeError(f"QSA layer {self.layer} select HIP error {status}")
            if self.capture is not None:
                self.capture.append(
                    {"first": first, "selected": self.selected[:chunk].clone()}
                )
            status = self.bank.kernel.attend(
                stream,
                *ptr(
                    query,
                    self.keys,
                    self.values,
                    self.selected,
                    self.partial,
                    self.stats,
                    output,
                ),
                1,
                count,
                past,
                first,
                chunk,
                self.capacity,
            )
            if status:
                raise RuntimeError(
                    f"QSA layer {self.layer} attention HIP error {status}"
                )
        self.seen = past + count
        assert self.pointers() == self.workspace_pointers
