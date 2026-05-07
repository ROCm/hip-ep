#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Standalone driver to pre-fetch the gpt-oss-20b model files.

Avoids having to spin up pytest just to trigger the dynamic_model_path
fixture. Imports the same `ensure_model` helper so we exercise the
multi-data-file code path in conftest.
"""
import sys
import time

sys.path.insert(0, "test/python")
from conftest import REPO_ROOT, ensure_model  # noqa: E402

_ONNX = "model_q4f16.onnx"
_DATA = [
    "model_q4f16.onnx_data",
    "model_q4f16.onnx_data_1",
    "model_q4f16.onnx_data_2",
    "model_q4f16.onnx_data_3",
    "model_q4f16.onnx_data_4",
    "model_q4f16.onnx_data_5",
    "model_q4f16.onnx_data_6",
]
_HF = (
    "https://huggingface.co/onnxruntime/gpt-oss-20b-onnx/resolve/main/"
    "webgpu/webgpu-int4-rtn-block-32"
)
_MODEL_DIR = REPO_ROOT / "models" / "gpt-oss-20b-int4-rtn-block-32"

t0 = time.perf_counter()
path = ensure_model(_MODEL_DIR, _ONNX, _DATA, _HF)
elapsed = time.perf_counter() - t0
print(f"\nDownload complete: {path} ({elapsed:.1f}s)")
