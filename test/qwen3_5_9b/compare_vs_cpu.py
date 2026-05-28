#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Compare EP standalone vs CPU per-row of image_features. Diagnose where
the residual 5% gap (cos=0.948 after the loop-accumulator fix) comes from."""

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


inputs = make_inputs()

# CPU
cpu_sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
cpu_out = cpu_sess.run(None, inputs)[0]
del cpu_sess

# EP standalone
devices = register_morphizen_ep(REPO_ROOT)
so = ort.SessionOptions()
so.add_provider_for_devices(devices, {})
sess = ort.InferenceSession(MODEL, sess_options=so)
ep_out = sess.run(None, inputs)[0]
del sess

print(f"Shapes: cpu={cpu_out.shape} ep={ep_out.shape}")
af = ep_out.astype(np.float32)
bf = cpu_out.astype(np.float32)

# Per-row cosine
print("\nPer-row cosine:")
for r in range(min(32, af.shape[0])):
    a = af[r]
    b = bf[r]
    a_n = np.linalg.norm(a)
    b_n = np.linalg.norm(b)
    if a_n == 0 or b_n == 0:
        cos = 0.0
    else:
        cos = float(np.dot(a, b) / (a_n * b_n))
    max_diff = float(np.max(np.abs(a - b)))
    print(
        f"  row {r:3d}: cos={cos:+.4f} max_diff={max_diff:.3e} cpu_norm={b_n:.2f} ep_norm={a_n:.2f}"
    )
