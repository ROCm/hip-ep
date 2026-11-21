#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Probe default Qwen prefill/decode, replay and retained FP32 state on native HIP."""

import argparse
import json
import shutil
from pathlib import Path

from verify_handoff import ROOT, WORK, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, required=True)
    args = parser.parse_args()
    source = ROOT / "_local/qsa/native-gqa/install"
    destination = WORK / "install"
    frozen = json.loads((ROOT / "_local/qsa/native-gqa/final-audit.json").read_text())
    for name, expected in frozen["binary_sha256"].items():
        assert sha(source / "bin" / name) == expected
    if not destination.exists():
        shutil.copytree(source, destination)
    for name, expected in frozen["binary_sha256"].items():
        assert sha(destination / "bin" / name) == expected
    from default_inference import Default

    import torch

    folder = args.folder
    exported = json.loads((folder / "export.json").read_text())
    assert sha(folder / "default.onnx") == exported["onnx_sha256"]
    assert sha(folder / "default.weights.bin") == exported["data_sha256"]
    model = Default(folder, capacity=128)
    ids = torch.tensor(
        [[1, 42, 83, 109, 27, 512, 33, 69]], device="cuda", dtype=torch.int64
    )
    first = model.forward(ids[:, :7].contiguous()).clone()
    continuation = model.forward(ids[:, 7:].contiguous(), past=7).clone()
    states = {
        name: value.clone()
        for name, value in model.caches.items()
        if name.endswith("state")
    }
    replay = model.forward(ids[:, :7].contiguous()).clone()
    repeated_continuation = model.forward(ids[:, 7:].contiguous(), past=7).clone()
    full = model.forward(ids).clone()
    for value in [first, continuation, replay, repeated_continuation, full]:
        assert torch.isfinite(value).all()
    assert torch.equal(first, replay) and torch.equal(
        continuation, repeated_continuation
    )
    recurrent = [
        value
        for name, value in model.caches.items()
        if name.endswith("recurrent_state")
    ]
    assert len(recurrent) == 3 and all(x.dtype == torch.float32 for x in recurrent)
    error = (
        (
            (full.float() - continuation.float()).square().mean()
            / full.float().square().mean()
        )
        .sqrt()
        .item()
    )
    assert error < 0.02, error
    receipt = {
        "status": "passed",
        "scope": "Four-block architecture/support diagnostic; not the 64-block baseline or accuracy",
        "layers": 4,
        "gpu": torch.cuda.get_device_name(),
        "torch": str(torch.__version__),
        "hip": torch.version.hip,
        "binary_sha256": frozen["binary_sha256"],
        "export_onnx_sha256": exported["onnx_sha256"],
        "prefill_tokens": 7,
        "decode_calls": 1,
        "capacity": 128,
        "replay_bit_exact": True,
        "full_vs_incremental_relative_rms": error,
        "recurrent_state_dtype": "FP32",
        "finite": True,
        "retained_state_tensor_count": len(states),
        "source_sha256": {p.name: sha(p) for p in Path(__file__).parent.glob("*.py")},
        "training_updates": 0,
        "selectors_loaded": False,
        "baseline_established": False,
    }
    (WORK / "default-4-support.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
