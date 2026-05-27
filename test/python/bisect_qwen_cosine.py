"""Per-tap bisect for the Qwen 3.5 vision encoder EP-vs-CPU cosine gap.

Adds one intermediate output at a time to a copy of vision.onnx, runs
both ORT-CPU and MorphiZen-EP, and prints the cosine. Use this to find
the first op whose EP output diverges from the CPU baseline.

Cost: each probe is a fresh EP compile (~30 s) because the EP cache key
is the ONNX graph hash and each probe changes the output set. Keep
n_taps small and bisect manually after locating the first divergent
band.

Empirical:
  * The first sanity tap (`mul_288`, early grid_thw arithmetic) returns
    cos=1.0 on the current branch, so the early portion of the graph
    is correct.
  * Final output (image_features) on grid_small=[2,8,8] currently
    measures cos=0.4687 vs ORT-CPU; the divergence point is between
    these two and is what this tool exists to find.
"""
import sys, os, numpy as np, onnx
sys.path.insert(0, os.path.dirname(__file__))
import onnxruntime as ort

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                         "..", ".."))
MODEL_DIR = os.path.join(REPO_ROOT, "models",
                         "Qwen3.5-9B-rtn-int4-int8-128gs-fp16-onnx-gpu")
MODEL = os.path.join(MODEL_DIR, "vision.onnx")
GRID = [2, 8, 8]

def make_inputs():
    t,h,w = GRID
    n = t*h*w
    rng = np.random.default_rng(0)
    return {
        "pixel_values": (rng.standard_normal((n,1536))*0.1).astype(np.float16),
        "image_grid_thw": np.array([GRID], dtype=np.int64),
    }
def cosine(a, b):
    af = a.astype(np.float32).flatten()
    bf = b.astype(np.float32).flatten()
    mask = np.isfinite(af) & np.isfinite(bf)
    af, bf = af[mask], bf[mask]
    if af.size == 0: return 0.0
    return float(np.dot(af, bf) / (np.linalg.norm(af)*np.linalg.norm(bf) + 1e-30))

m = onnx.load(MODEL, load_external_data=False)
vi_map = {v.name: v for v in list(m.graph.value_info) + list(m.graph.output) + list(m.graph.input)}
# Prefer LayerNorm / Mul / Add / MatMul / Gemm outputs to avoid Squeeze-output EP issues
safe_ops = {"LayerNormalization","Mul","Add","MatMul","Gemm","Sub","Div","Cos","Sin","Conv","Transpose","Reshape","Tile","Concat","Cast"}
all_outs = []
for n in m.graph.node:
    if n.op_type in safe_ops:
        for o in n.output:
            if o and o in vi_map:
                all_outs.append((n.op_type, n.name, o))

print(f"Safe-op tapable outputs: {len(all_outs)}", flush=True)
N = len(all_outs)
n_taps = 12
idxs = [int(N * (i+1)/(n_taps+1)) for i in range(n_taps)]
taps = [all_outs[i] for i in idxs]
for tag, name, out in taps: print(f"  [{tag}] {name} -> {out}", flush=True)

from conftest import register_morphizen_ep, REPO_ROOT
devices = register_morphizen_ep(REPO_ROOT)
inputs = make_inputs()
results = []
for i, (op, nname, tap) in enumerate(taps):
    print(f"\n[{i+1}/{n_taps}] {op}({nname}) -> {tap}", flush=True)
    mp = onnx.load(MODEL)
    if tap not in {o.name for o in mp.graph.output}:
        mp.graph.output.append(vi_map[tap])
    probe = os.path.join(MODEL_DIR, f"vision_probe_{i}.onnx")
    onnx.save(mp, probe, save_as_external_data=False)
    cpu_o = ep_o = None; err = None
    try:
        cpu_sess = ort.InferenceSession(probe, providers=["CPUExecutionProvider"])
        cpu_o = cpu_sess.run([tap], inputs)[0]
        del cpu_sess
    except Exception as e:
        err = f"CPU: {e}"
    if cpu_o is not None:
        try:
            so = ort.SessionOptions()
            so.add_provider_for_devices(devices, {})
            ep_sess = ort.InferenceSession(probe, sess_options=so)
            ep_o = ep_sess.run([tap], inputs)[0]
            del ep_sess
        except Exception as e:
            err = f"EP: {e}"
    try: os.remove(probe)
    except: pass
    if err:
        print(f"  ERR: {err[:120]}", flush=True)
        results.append((tap, op, None))
    else:
        cos = cosine(ep_o, cpu_o)
        tag = "OK " if cos >= 0.999 else ("LOW" if cos >= 0.5 else "BAD")
        print(f"  [{tag}] cos={cos:.4f} shape={cpu_o.shape}", flush=True)
        results.append((tap, op, cos))

print("\n=== Summary (topo order) ===", flush=True)
for (tap, op, cos) in results:
    if cos is None:
        print(f"  ERR {op:18s} {tap}", flush=True)
    else:
        tag = "OK " if cos >= 0.999 else ("LOW" if cos >= 0.5 else "BAD")
        print(f"  [{tag}] cos={cos:.4f} {op:18s} {tap}", flush=True)
