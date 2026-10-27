#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""A 2-D float16 MatMul compiled with HIPEP_USE_DYNAMIC_DISPATCH=1 must reach
wrap_dd_matmul and complete the DynamicDispatch call sequence.

[1, 1024] x [1024, 384] -> [1, 384] is the shape wrap_dd_matmul maps onto
DD's combined_gemm {Bin=1, B=6, M=1, K=1024, N=64} kernel. DynamicDispatch's
tests/test_c_api.cpp (test_wrap_dd_matmul) issues the identical dd_* calls
directly, so a failure here that does not reproduce there points at the
hip-ep <-> DD DLL boundary.

Numerics are not checked: the runtime zero-fills the QDQ tensors and
reinterprets the f16 weights as uint8.

Requires a runtime built with DynamicDispatch, DD_ROOT pointing at the DD repo
(xclbins), and the DD and XRT DLLs reachable via DYNAMICDISPATCH_ROOT/bin and
XILINX_XRT.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from conftest import EP_DLL_NAME, _ep_runtime_dirs

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EP_BIN, THEROCK_BIN = _ep_runtime_dirs(REPO_ROOT)
EP_DLL = EP_BIN / EP_DLL_NAME

M, K, N = 1, 1024, 384
# Raise (e.g. HIPEP_DD_TEST_RUNS=20000) to keep the NPU busy long enough to
# observe in Task Manager.
NUM_RUNS = int(os.environ.get("HIPEP_DD_TEST_RUNS", "2"))

_WORKER = textwrap.dedent(
    """
    import sys
    import numpy as np
    import onnxruntime as ort
    from onnx import TensorProto, helper

    ep_dll, M, K, N, runs = sys.argv[1], *map(int, sys.argv[2:6])

    a = helper.make_tensor_value_info("A", TensorProto.FLOAT16, [M, K])
    b = helper.make_tensor_value_info("B", TensorProto.FLOAT16, [K, N])
    c = helper.make_tensor_value_info("C", TensorProto.FLOAT16, [M, N])
    node = helper.make_node("MatMul", ["A", "B"], ["C"])
    graph = helper.make_graph([node], "dd_matmul", [a, b], [c])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])

    ort.register_execution_provider_library("AMDGPUExecutionProvider", ep_dll)
    from onnxruntime.capi._pybind_state import get_ep_devices
    devices = [d for d in get_ep_devices()
               if d.ep_name == "AMDGPUExecutionProvider"]
    if not devices:
        print("NO_EP_DEVICES", file=sys.stderr)
        sys.exit(3)

    so = ort.SessionOptions()
    so.add_provider_for_devices(devices, {"profile": "hip"})
    sess = ort.InferenceSession(model.SerializeToString(), sess_options=so)

    rng = np.random.default_rng(0)
    feeds = {"A": rng.random((M, K), dtype=np.float32).astype(np.float16),
             "B": rng.random((K, N), dtype=np.float32).astype(np.float16)}
    for _ in range(runs):
        (out,) = sess.run(None, feeds)
        assert out.shape == (M, N), out.shape
    print("RUN_OK", flush=True)
    """
)


def _run_worker(use_dd=True):
    env = dict(os.environ)
    env["HIPEP_USE_DYNAMIC_DISPATCH"] = "1" if use_dd else "0"
    # A compile failure must not silently fall back to the CPU.
    env["HIPDNN_EP_STRICT"] = "1"
    env.setdefault("THEROCK_DIST", str(THEROCK_BIN.parent))
    path = [EP_BIN, THEROCK_BIN]
    if env.get("DYNAMICDISPATCH_ROOT"):
        path.append(Path(env["DYNAMICDISPATCH_ROOT"]) / "bin")
    if env.get("XILINX_XRT"):
        path.append(Path(env["XILINX_XRT"]))
    env["PATH"] = (
        os.pathsep.join(str(p) for p in path) + os.pathsep + env.get("PATH", "")
    )
    return subprocess.run(
        [
            sys.executable,
            "-c",
            _WORKER,
            str(EP_DLL),
            str(M),
            str(K),
            str(N),
            str(NUM_RUNS),
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
    )


@pytest.mark.skipif(
    not EP_DLL.exists(), reason=f"{EP_DLL} not found (set HIPEP_EP_BIN)"
)
@pytest.mark.skipif(not os.environ.get("DD_ROOT"), reason="DD_ROOT not set")
def test_dd_matmul_call_sequence():
    proc = _run_worker()
    log = proc.stdout + "\n" + proc.stderr

    if "DynamicDispatch not available (mock runtime)" in log:
        pytest.skip("runtime built without DynamicDispatch (mock wrap_dd_matmul)")

    assert "[DD] wrap_dd_matmul" in log, f"wrap_dd_matmul was never called:\n{log}"
    assert (
        "dd_combined_gemm_create model=8x4_PSMU_ST1_v3.2 Bin=1 B=6 M=1 K=1024 N=64"
        in log
    ), f"unexpected combined_gemm configuration:\n{log}"
    assert log.count("dd_combined_gemm_create model=") == 1, (
        f"handle was not cached across inferences:\n{log}"
    )
    assert log.count("dd_combined_gemm_execute rc=0") == NUM_RUNS, (
        f"dd_combined_gemm_execute did not succeed {NUM_RUNS} times:\n{log}"
    )
    assert "RUN_OK" in proc.stdout, f"inference failed (rc={proc.returncode}):\n{log}"


@pytest.mark.skipif(
    not EP_DLL.exists(), reason=f"{EP_DLL} not found (set HIPEP_EP_BIN)"
)
def test_matmul_gpu_fallback():
    """The same MatMul with HIPEP_USE_DYNAMIC_DISPATCH=0 runs on the GPU."""
    proc = _run_worker(use_dd=False)
    log = proc.stdout + "\n" + proc.stderr

    assert "[DD]" not in log, f"DynamicDispatch was used with DD disabled:\n{log}"
    assert "RUN_OK" in proc.stdout, f"inference failed (rc={proc.returncode}):\n{log}"
