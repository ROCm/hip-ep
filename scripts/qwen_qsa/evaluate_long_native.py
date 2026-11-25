#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Score locked natural long-document suffixes with matched native Qwen variants."""

import argparse
import json
from pathlib import Path

from verify_handoff import WORK, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["Original", "Changed"], required=True)
    parser.add_argument("--context", type=int, choices=[16384, 32768], required=True)
    args = parser.parse_args()
    from native_long_inference import LongNative

    import torch
    import torch.nn.functional as functional

    torch.set_num_threads(2)
    data = WORK / "long-quality-v1"
    manifest = json.loads((data / "manifest.json").read_text())
    assert (
        manifest["state"] == "locked"
        and manifest["locked_before_native_qsa_integration"]
    )
    assert sha(data / "documents.json") == manifest["documents_sha256"]
    documents = json.loads((data / "documents.json").read_text())
    assert len(documents) == 4
    folder = WORK / "native-gqa/models" / args.variant
    exported = json.loads((folder / "export.json").read_text())
    assert sha(folder / "default.onnx") == exported["onnx_sha256"]
    assert sha(folder / "default.weights.bin") == exported["data_sha256"]
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
    output = WORK / "native-gqa/long-quality" / args.variant
    output.mkdir(parents=True, exist_ok=True)
    protocol = {
        "variant": args.variant,
        "export_onnx_sha256": exported["onnx_sha256"],
        "export_data_sha256": exported["data_sha256"],
        "binary_sha256": binaries,
        "dataset_manifest_sha256": sha(data / "manifest.json"),
        "documents_sha256": manifest["documents_sha256"],
        "contexts": manifest["contexts"],
        "scored_suffix_targets": 1023,
        "batch": 1,
        "recurrent_state": "FP32",
        "head": "FP16",
        "evaluator_sha256": sha(Path(__file__)),
        "suffix_inference_sha256": sha(
            Path(__file__).with_name("native_long_inference.py")
        ),
        "base_inference_sha256": sha(Path(__file__).with_name("native_inference.py")),
        "training_updates": 0,
        "selectors_loaded": args.variant == "Changed",
        "scope": "Four locked natural documents per prefix length; final 1023 next-token targets only",
    }
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        assert json.loads(protocol_path.read_text()) == protocol
    else:
        protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    path = output / "rows.jsonl"
    rows = (
        [json.loads(line) for line in path.read_text().splitlines()]
        if path.exists()
        else []
    )
    assert all(row["protocol_sha256"] == sha(protocol_path) for row in rows)
    assert len({(row["context"], row["index"]) for row in rows}) == len(rows)
    done = {(row["context"], row["index"]) for row in rows}
    selected = [i for i in range(4) if (args.context, i) not in done]
    if not selected:
        print("All requested frozen rows already complete", flush=True)
        return
    print("Loading native long-context", args.variant, args.context, flush=True)
    model = LongNative(folder, capacity=args.context)
    with torch.inference_mode():
        for index in selected:
            document = documents[index]
            ids = torch.tensor(
                [document["input_ids"][: args.context]],
                device="cuda",
                dtype=torch.int64,
            )
            logits = model.suffix(ids)
            assert torch.isfinite(logits).all()
            targets = ids[:, -1023:]
            loss = 0.0
            for start in range(0, 1023, 128):
                end = min(start + 128, 1023)
                loss += functional.cross_entropy(
                    logits[:, start:end].float().reshape(-1, logits.shape[-1]),
                    targets[:, start:end].reshape(-1),
                    reduction="sum",
                ).item()
            row = {
                "variant": args.variant,
                "context": args.context,
                "index": index,
                "document_sha256": document["document_sha256"],
                "window_sha256": document["window_sha256"],
                "targets": 1023,
                "nll_sum": loss,
                "finite": True,
                "protocol_sha256": sha(protocol_path),
                "training_updates": 0,
            }
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            print(row, flush=True)
            del logits


if __name__ == "__main__":
    main()
