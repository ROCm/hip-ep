#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Quick test: does disabling ORT graph optimizations fix the standalone bug?"""

import os
import sys
from pathlib import Path
import numpy as np
import onnxruntime as ort

# register_morphizen_ep lives in test/python/conftest.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
from conftest import register_morphizen_ep

REPO_ROOT = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
MODEL = str(
    REPO_ROOT
    / "models"
    / "Qwen3.5-9B-rtn-int4-int8-128gs-fp16-onnx-gpu"
    / "vision.onnx"
)
GRID = [2, 8, 8]


def make_inputs():
    t, h, w = GRID
    n = t * h * w
    rng = np.random.default_rng(0)
    return {
        "pixel_values": (rng.standard_normal((n, 1536)) * 0.1).astype(np.float16),
        "image_grid_thw": np.array([GRID], dtype=np.int64),
    }


def cos(a, b):
    af = a.astype(np.float32).flatten()
    bf = b.astype(np.float32).flatten()
    m = np.isfinite(af) & np.isfinite(bf)
    af, bf = af[m], bf[m]
    return float(np.dot(af, bf) / (np.linalg.norm(af) * np.linalg.norm(bf)))


devices = register_morphizen_ep(REPO_ROOT)
inputs = make_inputs()

# CPU baseline
cpu_sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
cpu_out = cpu_sess.run(None, inputs)[0]
del cpu_sess
print(f"CPU output shape: {cpu_out.shape}", flush=True)

for opt_level, label in [
    (ort.GraphOptimizationLevel.ORT_DISABLE_ALL, "DISABLE_ALL"),
    (ort.GraphOptimizationLevel.ORT_ENABLE_BASIC, "ENABLE_BASIC"),
    (ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED, "ENABLE_EXTENDED"),
    (ort.GraphOptimizationLevel.ORT_ENABLE_ALL, "ENABLE_ALL"),
]:
    so = ort.SessionOptions()
    so.graph_optimization_level = opt_level
    so.add_provider_for_devices(devices, {})
    sess = ort.InferenceSession(MODEL, sess_options=so)
    ep_out = sess.run(None, inputs)[0]
    del sess
    c = cos(ep_out, cpu_out)
    print(
        f"opt_level={label:20s}: cos={c:.4f} ep_norm={np.linalg.norm(ep_out.astype(np.float32)):.2f}",
        flush=True,
    )
