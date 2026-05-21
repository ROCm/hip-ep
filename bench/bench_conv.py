#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Conv backend microbenchmark.

Times the EP-side Conv wrapper over a small matrix of shapes for whatever
backend HIPDNN_EP_CONV selects (default miopen). Run twice -- once per
backend -- and compare the per-shape latencies.

    HIPDNN_EP_CONV=miopen python bench/bench_conv.py
    HIPDNN_EP_CONV=ck     python bench/bench_conv.py

Both backends now honour fp16; the bench builds fp16 ONNX so it's
apples-to-apples. Output is a per-shape table of avg / min / max ms per
inference, plus throughput in iterations / second.

Methodology:
- Build a one-node Conv ONNX in memory (no disk I/O).
- Create the EP session ONCE per shape (compiles model.dll once).
- IOBinding reused across iterations to avoid binding cost in the timed loop.
- N_WARMUP iterations to settle autotune + caches; then N_RUNS timed.
- IOBinding inputs / outputs are HOST-allocated numpy arrays bound directly
  via bind_cpu_input / bind_output. The EP path's H2D / D2H is identical
  for both backends so comparison stays fair.

Don't run with HIPDNN_EP_PERF=1 -- profiling overhead masks 5%-class
differences and can invert relative perf (CLAUDE.md).
"""

import gc
import os
import pathlib
import sys
import time

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
# conftest.py lives next to the python tests; reuse register_morphizen_ep
# rather than duplicate the EP-loading boilerplate here.
sys.path.insert(0, str(REPO_ROOT / "test" / "python"))
from conftest import register_morphizen_ep  # noqa: E402


# (id, N, C, H, W, K, ky, kx, stride, pad, dilation, group)
#
# One representative shape per workload class -- enough to spot perf
# regressions across the surface that real models actually hit, without
# pad-waiting through 25 redundant variants. If you add a row, justify
# what behavior or workload class it covers that none of these do.
SHAPES = [
    # Tiny smoke -- per-call overhead dominates; useful as the bottom
    # of the dispatch curve.
    ("smoke_3x3", 1, 3, 32, 32, 16, 3, 3, 1, 1, 1, 1),
    # ResNet-50 stem -- canonical large-spatial 7x7 stride-2 entry layer.
    ("rn50_stem", 1, 3, 224, 224, 64, 7, 7, 2, 3, 1, 1),
    # ResNet-50 mid-stage 3x3 -- canonical interior conv (most CV models).
    ("rn50_s3_3x3", 1, 256, 14, 14, 256, 3, 3, 1, 1, 1, 1),
    # 1x1 pointwise -- exercises the pointwise / projection path.
    ("1x1", 1, 16, 32, 32, 32, 1, 1, 1, 0, 1, 1),
    # CLIP ViT-L/14 patch conv -- representative of all VL/ViT patch stems
    # (large-K, large-spatial, big-stride single-shot). 1x3x224x224 ->
    # 1x1024x16x16.
    ("vit_l14_224", 1, 3, 224, 224, 1024, 14, 14, 14, 0, 1, 1),
    # SDXL UNet bottom block (1280 channels @ 16x16) -- the deepest /
    # widest residual conv in SDXL inference. Big weights (~28 MB fp16),
    # small spatial -- the canonical generative-diffusion hot loop.
    ("sdxl_unet_1280", 1, 1280, 16, 16, 1280, 3, 3, 1, 1, 1, 1),
    # 2K image patchify (ViT-B/16 over a 2K image) -- representative of
    # high-res segmentation / DiT / vision-encoder input stages. 4K is a
    # better stress-test for spatial scaling but takes ~30 ms/iter, which
    # blows up the bench's 200-iter run; 2K trades half the spatial axis
    # for ~4x lower wall time. 1x3x2048x2048 -> 1x768x128x128.
    ("2k_vit_b16", 1, 3, 2048, 2048, 768, 16, 16, 16, 0, 1, 1),
]

N_WARMUP = 20
N_RUNS = 200


def make_conv_onnx_fp16(N, C, H, W, K, ky, kx, stride, pad, dilation, group, *, seed=0):
    rng = np.random.default_rng(seed)
    inp = rng.standard_normal((N, C, H, W), dtype=np.float32).astype(np.float16)
    weights = (
        rng.standard_normal((K, C // group, ky, kx), dtype=np.float32) * 0.1
    ).astype(np.float16)
    Ho = (H + 2 * pad - dilation * (ky - 1) - 1) // stride + 1
    Wo = (W + 2 * pad - dilation * (kx - 1) - 1) // stride + 1

    weight_init = numpy_helper.from_array(weights, name="W")
    inp_proto = helper.make_tensor_value_info("X", TensorProto.FLOAT16, (N, C, H, W))
    out_proto = helper.make_tensor_value_info("Y", TensorProto.FLOAT16, (N, K, Ho, Wo))
    node = helper.make_node(
        "Conv",
        ["X", "W"],
        ["Y"],
        kernel_shape=[ky, kx],
        strides=[stride, stride],
        pads=[pad, pad, pad, pad],
        dilations=[dilation, dilation],
        group=group,
    )
    g = helper.make_graph([node], "g", [inp_proto], [out_proto], [weight_init])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)], ir_version=10)
    onnx.checker.check_model(m)
    return m.SerializeToString(), inp, (N, K, Ho, Wo)


def bench_shape(name, *args):
    """Returns dict with avg_ms, min_ms, max_ms, iter_per_sec, gflops, name."""
    devices = register_morphizen_ep(REPO_ROOT)
    if not devices:
        raise SystemExit("MorphiZen EP not found -- run python build.py first")

    model_bytes, inp, _out_shape = make_conv_onnx_fp16(*args, seed=42)
    so = ort.SessionOptions()
    so.add_provider_for_devices(devices, {})
    sess = ort.InferenceSession(model_bytes, sess_options=so)

    # IOBinding: bind once, reuse in the loop. bind_cpu_input copies into
    # whichever device memory the EP wants; that copy is constant across
    # both backends so it cancels in relative comparison.
    bnd = sess.io_binding()
    for _ in range(N_WARMUP):
        bnd.bind_cpu_input("X", inp)
        bnd.bind_output("Y", "cpu")
        sess.run_with_iobinding(bnd)
        # Force materialization so timing isn't masked by lazy copy-on-read.
        bnd.get_outputs()[0].numpy()
        bnd.clear_binding_inputs()
        bnd.clear_binding_outputs()

    times = []
    for _ in range(N_RUNS):
        bnd.bind_cpu_input("X", inp)
        bnd.bind_output("Y", "cpu")
        t0 = time.perf_counter()
        sess.run_with_iobinding(bnd)
        bnd.get_outputs()[0].numpy()
        times.append(time.perf_counter() - t0)
        bnd.clear_binding_inputs()
        bnd.clear_binding_outputs()

    ms = np.array(times) * 1000.0
    # Conv FLOPs (2 * N * K * Ho * Wo * (C/g) * ky * kx)
    N, C, H, W, K, ky, kx, stride, pad, dilation, group = args
    Ho = (H + 2 * pad - dilation * (ky - 1) - 1) // stride + 1
    Wo = (W + 2 * pad - dilation * (kx - 1) - 1) // stride + 1
    flops = 2 * N * K * Ho * Wo * (C // group) * ky * kx
    gflops = (flops / 1e9) / (ms.mean() / 1000.0)

    del sess, bnd
    gc.collect()
    return {
        "name": name,
        "avg_ms": float(ms.mean()),
        "min_ms": float(ms.min()),
        "max_ms": float(ms.max()),
        "p50_ms": float(np.percentile(ms, 50)),
        "p99_ms": float(np.percentile(ms, 99)),
        "iter_per_sec": 1000.0 / float(ms.mean()),
        "gflops": float(gflops),
    }


def main():
    backend = os.environ.get("HIPDNN_EP_CONV", "miopen")
    print(f"\nbench_conv.py  backend={backend}  warmup={N_WARMUP} runs={N_RUNS}\n")
    print(
        f"{'shape':<14} {'avg_ms':>9} {'min_ms':>9} {'p50_ms':>9} "
        f"{'p99_ms':>9} {'it/s':>9} {'GFLOPS':>9}"
    )
    print("-" * 80)
    for shape in SHAPES:
        r = bench_shape(*shape)
        print(
            f"{r['name']:<14} {r['avg_ms']:>9.3f} {r['min_ms']:>9.3f} "
            f"{r['p50_ms']:>9.3f} {r['p99_ms']:>9.3f} "
            f"{r['iter_per_sec']:>9.1f} {r['gflops']:>9.1f}"
        )
    print()


if __name__ == "__main__":
    main()
