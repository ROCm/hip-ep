#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Export full Original/Changed graphs with four native QSA GQA nodes."""

import json
import os

import onnx
import torch
from onnx import TensorProto, helper
from prepare_baseline import SELECTOR, SELECTOR_SHA, WORK, original_folder, sha


def main():
    assert sha(SELECTOR) == SELECTOR_SHA
    bank = torch.load(SELECTOR, map_location="cpu", weights_only=True)
    assert bank["architecture"]["layers"] == [7, 11, 15, 23]
    tensors = bank["selector"]
    assert all(t.dtype == torch.bfloat16 for t in tensors.values())
    source = original_folder()
    manifest = json.loads((source / "partitioned/manifest.json").read_text())
    for variant in ["Original", "Changed"]:
        folder = WORK / "native-gqa/models" / variant
        folder.mkdir(parents=True, exist_ok=True)
        assert not (folder / "original.onnx").exists(), "Preserve existing exports"
        os.link(source / "original.weights.bin", folder / "original.weights.bin")
        graph = onnx.load(source / "original.onnx", load_external_data=False)
        # Both variants have identical B1 specialization and unchanged QKV/O/MLPs.
        for info in [*graph.graph.input, *graph.graph.output, *graph.graph.value_info]:
            for dim in info.type.tensor_type.shape.dim:
                if dim.dim_param == "batch":
                    dim.dim_value = 1
        frozen_hashes = {}
        if variant == "Changed":
            graph.opset_import.append(helper.make_opsetid("com.amd", 1))
            with (folder / "qsa.selectors.bin").open("wb") as data:
                for layer in [7, 11, 15, 23]:
                    b = manifest["boundaries"][str(layer)]
                    node = next(n for n in graph.graph.node if b["output"] in n.output)
                    assert node.op_type == "GroupQueryAttention"
                    node.op_type, node.domain = "QsaGroupQueryAttention", "com.amd"
                    del node.attribute[:]
                    names = []
                    for kind in ["projection", "q_norm", "k_norm"]:
                        name = f"selectors.{layer}.{kind}.weight"
                        value = tensors[name].contiguous()
                        raw = value.view(torch.uint8).numpy().tobytes()
                        offset = data.tell()
                        data.write(raw)
                        tensor = TensorProto(name=name, data_type=TensorProto.BFLOAT16)
                        tensor.dims.extend(value.shape)
                        tensor.data_location = TensorProto.EXTERNAL
                        for key, val in [
                            ("location", "qsa.selectors.bin"),
                            ("offset", str(offset)),
                            ("length", str(len(raw))),
                        ]:
                            tensor.external_data.add(key=key, value=val)
                        graph.graph.initializer.append(tensor)
                        names.append(name)
                        import hashlib

                        frozen_hashes[name] = hashlib.sha256(raw).hexdigest()
                    del node.input[:]
                    node.input.extend(
                        [
                            b["query"],
                            b["key"],
                            b["value"],
                            b["past_key"],
                            b["past_value"],
                            "seqlens_k",
                            "capacity",
                            b["hidden"],
                            *names,
                            "cos_full_attention",
                            "sin_full_attention",
                        ]
                    )
        assert sum(n.op_type == "MatMulNBits" for n in graph.graph.node) == 224
        source_nodes = {
            n.name: n.SerializeToString()
            for n in onnx.load(
                source / "original.onnx", load_external_data=False
            ).graph.node
            if n.op_type == "MatMulNBits"
        }
        assert all(
            n.SerializeToString() == source_nodes[n.name]
            for n in graph.graph.node
            if n.op_type == "MatMulNBits"
        )
        onnx.save_model(graph, folder / "original.onnx")
        # The established source uses hip-ep's default-domain
        # SimplifiedLayerNormalization, absent from stock ONNX's schema table.
        # Preserve it byte-for-byte; target-EP compilation validates execution.
        assert sum(n.op_type == "QsaGroupQueryAttention" for n in graph.graph.node) == (
            4 if variant == "Changed" else 0
        )
        receipt = json.loads((source / "export.json").read_text())
        receipt.update(
            variant=variant,
            batch=1,
            selector_sha256=SELECTOR_SHA,
            selector_tensor_sha256=frozen_hashes,
            execution="one full ONNX graph/session",
        )
        receipt["artifacts"] = {
            p.name: {"bytes": p.stat().st_size, "sha256": sha(p)}
            for p in [
                folder / "original.onnx",
                folder / "original.weights.bin",
                *folder.glob("qsa.selectors.bin"),
            ]
        }
        (folder / "export.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(variant, sha(folder / "original.onnx"), flush=True)
    assert sha(SELECTOR) == SELECTOR_SHA


if __name__ == "__main__":
    main()
