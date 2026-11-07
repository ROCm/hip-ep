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

To compare devices in Task Manager, set HIPEP_DD_TEST_SECONDS (e.g. 5): the
GPU test runs first, then the NPU test, each looping for that long.

Requires a runtime built with DynamicDispatch, DD_ROOT pointing at the DD repo
(xclbins), and the DD and XRT DLLs reachable via DYNAMICDISPATCH_ROOT/bin and
XILINX_XRT.
"""

import os
import re
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
# When RUN_SECONDS is set, each worker loops for that much inference time
# (session setup excluded) and NUM_RUNS is ignored.
NUM_RUNS = int(os.environ.get("HIPEP_DD_TEST_RUNS", "2"))
RUN_SECONDS = float(os.environ.get("HIPEP_DD_TEST_SECONDS", "0"))
# The GPU test is not tied to a DD kernel shape, so it uses a square MatMul
# large enough to show up as GPU compute rather than launch overhead.
GPU_SIZE = int(os.environ.get("HIPEP_GPU_TEST_SIZE", "2048"))

_WORKER = textwrap.dedent(
    """
    import gc
    import sys
    import time
    import numpy as np
    import onnxruntime as ort
    from onnx import TensorProto, helper

    ep_dll, M, K, N, runs = sys.argv[1], *map(int, sys.argv[2:6])
    seconds = float(sys.argv[6])

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
    done = 0
    start = time.perf_counter()
    while done < runs or time.perf_counter() - start < seconds:
        (out,) = sess.run(None, feeds)
        assert out.shape == (M, N), out.shape
        done += 1
    elapsed = time.perf_counter() - start

    # Release DD ops and their XRT contexts while the process is fully alive;
    # teardown during interpreter shutdown/DLL unload can wedge the NPU driver.
    del out, sess
    gc.collect()
    print("SESSION_RELEASED", file=sys.stderr, flush=True)
    print(f"RUN_OK runs={done} seconds={elapsed:.2f} "
          f"ms_per_run={1000 * elapsed / done:.3f}", flush=True)
    """
)


def _run_worker(use_dd=True, m=M, k=K, n=N):
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
            str(m),
            str(k),
            str(n),
            str(1 if RUN_SECONDS > 0 else NUM_RUNS),
            str(RUN_SECONDS),
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
    )


def _completed_runs(proc, log):
    match = re.search(r"RUN_OK runs=(\d+).*", proc.stdout)
    assert match, f"inference failed (rc={proc.returncode}):\n{log}"
    print(match.group(0))
    return int(match.group(1))


# Defined first so pytest runs the GPU phase before the NPU phase.
@pytest.mark.skipif(
    not EP_DLL.exists(), reason=f"{EP_DLL} not found (set HIPEP_EP_BIN)"
)
def test_matmul_gpu_fallback():
    """A MatMul with HIPEP_USE_DYNAMIC_DISPATCH=0 runs on the GPU."""
    proc = _run_worker(use_dd=False, m=GPU_SIZE, k=GPU_SIZE, n=GPU_SIZE)
    log = proc.stdout + "\n" + proc.stderr

    assert "[DD]" not in log, f"DynamicDispatch was used with DD disabled:\n{log}"
    _completed_runs(proc, log)


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
    _completed_runs(proc, log)
    # The runtime logs an op's first execute and every failure.
    executes = re.findall(r"dd_op_execute rc=(-?\d+)", log)
    assert executes == ["0"], f"dd_op_execute failed:\n{log}"
    assert "copy failed" not in log, f"DD tensor staging failed:\n{log}"
    destroyed = proc.stderr.find("wrap_dd_matmul: dd_op_destroy")
    assert 0 <= destroyed < proc.stderr.find("SESSION_RELEASED"), (
        f"DD op was not destroyed when the session was released:\n{log}"
    )
