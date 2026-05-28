"""Run the Qwen 3.5 vision encoder twice (standalone EP, then EP with
add_92 as an extra output) and dump the per-element diff of the final
`image_features`. Use to pinpoint WHICH parts of the output get corrupted
when buffer-pool reuses make a different decision."""
import os
import sys
import numpy as np
import onnx
import onnxruntime as ort
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from conftest import register_morphizen_ep

REPO_ROOT = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
MODEL_DIR = REPO_ROOT / "models" / "Qwen3.5-9B-rtn-int4-int8-128gs-fp16-onnx-gpu"
MODEL = str(MODEL_DIR / "vision.onnx")
GRID = [2, 8, 8]


def make_inputs():
    t, h, w = GRID
    n = t * h * w
    rng = np.random.default_rng(0)
    return {
        "pixel_values": (rng.standard_normal((n, 1536)) * 0.1).astype(np.float16),
        "image_grid_thw": np.array([GRID], dtype=np.int64),
    }


def run(model_path):
    devices = register_morphizen_ep(REPO_ROOT)
    so = ort.SessionOptions()
    so.add_provider_for_devices(devices, {})
    sess = ort.InferenceSession(model_path, sess_options=so)
    out = sess.run(["image_features"], make_inputs())[0]
    del sess
    return out


def main():
    # Standalone: just image_features
    print("Running standalone EP...", flush=True)
    ep_standalone = run(MODEL)

    # With add_92 pinned
    print("Building probe with add_92 added as output...", flush=True)
    m = onnx.load(MODEL, load_external_data=False)
    vi_map = {v.name: v for v in list(m.graph.value_info) + list(m.graph.output) + list(m.graph.input)}
    if "add_92" not in {o.name for o in m.graph.output}:
        m.graph.output.append(vi_map["add_92"])
    probe_path = str(MODEL_DIR / "vision_probe_add92.onnx")
    onnx.save(m, probe_path, save_as_external_data=False)
    print("Running EP with add_92 pinned...", flush=True)
    ep_pinned = run(probe_path)
    try:
        os.remove(probe_path)
    except OSError:
        pass

    print(f"\nStandalone shape: {ep_standalone.shape}, dtype: {ep_standalone.dtype}")
    print(f"Pinned     shape: {ep_pinned.shape}, dtype: {ep_pinned.dtype}")

    af = ep_standalone.astype(np.float32)
    bf = ep_pinned.astype(np.float32)
    diff = np.abs(af - bf)
    print(f"\n|standalone - pinned| stats:")
    print(f"  max:    {diff.max():.4e}")
    print(f"  mean:   {diff.mean():.4e}")
    print(f"  nonzero: {(diff > 0).sum()} / {diff.size}")
    print(f"  count > 0.01:  {(diff > 0.01).sum()}")
    print(f"  count > 1.0:   {(diff > 1.0).sum()}")

    # Identify rows where differences cluster
    per_row_max = diff.max(axis=1)
    bad_rows = np.argsort(per_row_max)[::-1][:10]
    print(f"\nTop 10 rows by max-diff:")
    for r in bad_rows:
        print(f"  row {r:3d}: max_diff={per_row_max[r]:.4e}  "
              f"standalone[0:3]={af[r, :3]}  pinned[0:3]={bf[r, :3]}")
    print(f"\nDistribution of rows by max-diff bucket:")
    buckets = [0, 1e-4, 1e-2, 1.0, 10.0, 100.0, 1e6]
    for lo, hi in zip(buckets[:-1], buckets[1:]):
        n = np.sum((per_row_max >= lo) & (per_row_max < hi))
        print(f"  [{lo:.0e}, {hi:.0e}): {n} rows")


if __name__ == "__main__":
    main()
