#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Check invalid native QSA cache history returns an ORT error, then recovers."""
# isort: skip_file
# Native DLL selection must precede runtime/Torch imports.
# ruff: noqa: E402

import json
import os
from pathlib import Path

TASK = Path(__file__).resolve().parents[2] / "_local/qsa/native-gqa"
os.environ["QSA_HIP_EP_INSTALL"] = str(TASK / "install")
os.environ["QSA_MODEL_CACHE"] = str(TASK / "model-cache")
os.environ.pop("QSA_ORT_PROFILE_DIR", None)

from runtime import INSTALL
import torch

from original_inference import Original


@torch.inference_mode()
def main():
    assert INSTALL.resolve() == (TASK / "install").resolve()
    torch.set_num_threads(8)
    model = Original(TASK / "models/Changed", 1024)
    token = torch.tensor([[42]], dtype=torch.int64, device="cuda")
    try:
        model.forward(token, past=17)
    except RuntimeError as exc:
        assert "inference_compute" in str(exc)
        message = str(exc)
    else:
        raise AssertionError("Invalid QSA history returned success")
    # The next full prefill must clear the error flag and rebuild selector state.
    first = model.forward(token).clone()
    assert first.isfinite().all()
    assert model.forward(token, past=1).isfinite().all()
    assert torch.equal(first, model.forward(token))
    (TASK / "error-validation.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "cache_without_prefill_rejected": True,
                "error": message,
                "next_prefill_recovers": True,
                "training_updates": 0,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
