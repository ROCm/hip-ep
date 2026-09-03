#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""The compile must not abort on a constant/initializer graph output.

A graph output that is a bare constant initializer (e.g. an Identity(constant)
that ORT constant folding collapses to a producerless graph output, as in
Model-PSI-QDQ-v3_0's output_exposed_scale / output_exposed_zero_point) is not
produced by any node in the fused region. `fuse_graph` and `build_metadata_json`
(backend-mlir-compiler/level-1-pass/src/pass_main.cpp) must exclude it so the
meta_def, the compiled artifact, and the custom-op output map stay consistent;
ORT then serves the initializer-backed output directly. A regression would
either FATAL (cannot find producer / meta_def output mismatch) or segfault while
building the fused-function return.

The surface under test is compile-time (fusion + metadata build), which runs
during session creation. The worker only creates the session. Session creation
is driven in a child process so a glog CHECK abort (a regression) doesn't take
down the pytest runner.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
# AMDGPU umbrella EP (loads hip-backend.dll → hipgpu.dll underneath).
EP_DLL = REPO_ROOT / "install" / "dist" / "bin" / "amdgpu-ep.dll"


def _build_model(path: Path) -> None:
    """Graph with a real compute output plus a constant graph output.

    input  : float[1, 8]
    Relu(input)          -> compute_out : float[1, 8]   (real node output)
    Identity(const_val)  -> const_out   : uint8[1]      (producerless constant)

    const_out mirrors an exposed quant scale/zero_point: a constant surfaced
    directly as a graph output.
    """
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 8])
    compute_out = helper.make_tensor_value_info(
        "compute_out", TensorProto.FLOAT, [1, 8]
    )
    const_out = helper.make_tensor_value_info("const_out", TensorProto.UINT8, [1])

    const_val = numpy_helper.from_array(
        np.array([204], dtype=np.uint8), name="const_val"
    )
    relu = helper.make_node("Relu", ["input"], ["compute_out"], name="relu")
    identity = helper.make_node("Identity", ["const_val"], ["const_out"], name="expose")

    graph = helper.make_graph(
        [relu, identity],
        "constant_graph_output",
        [inp],
        [compute_out, const_out],
        initializer=[const_val],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    onnx.save(model, str(path))


# Child-process worker: register the EP, open a session. Session creation drives
# the level-1 pass (fuse_graph + build_metadata_json); reaching
# "SESSION_CREATED_OK" proves the constant graph output did not abort the fuse or
# the metadata build.
_WORKER = textwrap.dedent(
    """
    import sys
    import onnxruntime as ort

    model_path, ep_dll = sys.argv[1], sys.argv[2]

    ort.register_execution_provider_library("AMDGPUExecutionProvider", ep_dll)
    from onnxruntime.capi._pybind_state import get_ep_devices
    devices = [d for d in get_ep_devices()
               if d.ep_name == "AMDGPUExecutionProvider"]
    if not devices:
        print("NO_EP_DEVICES", file=sys.stderr)
        sys.exit(3)

    so = ort.SessionOptions()
    so.add_provider_for_devices(devices, {"profile": "hip"})
    sess = ort.InferenceSession(model_path, sess_options=so)
    print("SESSION_CREATED_OK")
    """
)


def _run_worker(model_path: Path):
    env = dict(os.environ)
    # Leave strict off so a soft CPU fallback is not turned into an abort --
    # the only abort source we test for is a (regressed) compile-time CHECK.
    env["HIPDNN_EP_STRICT"] = "0"
    dist_bin = REPO_ROOT / "install" / "dist" / "bin"
    therock_bin = REPO_ROOT / "install" / "therock" / "bin"
    env["THEROCK_DIST"] = str(REPO_ROOT / "install" / "therock")
    env["PATH"] = f"{dist_bin};{therock_bin};" + env.get("PATH", "")
    return subprocess.run(
        [sys.executable, "-c", _WORKER, str(model_path), str(EP_DLL)],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )


@pytest.fixture(scope="module")
def model_path(tmp_path_factory):
    p = tmp_path_factory.mktemp("constant_output") / "constant_graph_output.onnx"
    _build_model(p)
    return p


@pytest.mark.skipif(not EP_DLL.exists(), reason="EP DLL not built — run build.py")
def test_compiles_constant_graph_output(model_path):
    """The compile must succeed for a graph with a constant graph output.

    fuse_graph and build_metadata_json exclude the producerless constant output,
    keeping the meta_def, artifact, and custom-op output map consistent, so
    session creation completes cleanly. A regression (FATAL on the missing
    producer / meta_def output mismatch, or a segfault building the fused return)
    would abort here and fail this test.
    """
    proc = _run_worker(model_path)
    combined = proc.stdout + "\n" + proc.stderr
    assert "SESSION_CREATED_OK" in proc.stdout, (
        f"session creation did not complete:\nrc={proc.returncode}\n{combined}"
    )
