#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""The compile must not hard-CHECK on a dynamic output dim whose symbolic name
appears on no input.

`build_metadata_json` (backend-mlir-compiler/level-1-pass/src/pass_main.cpp)
emits each output's shape verbatim. The output-allocator ABI (the only ABI at
the EP front-end) sizes dynamic outputs in-graph at runtime, so a data-dependent
output extent computed *inside* the graph (e.g. a vision model's feature count,
whose symbol is on no input) compiles fine -- no per-output-dim resolution and
no compile-time CHECK on the dynamic dim.

The surface under test is the compile-time metadata build, which runs during
session creation. The worker only creates the session; it does NOT run inference
-- NonZero's GPU kernel has a separate, pre-existing execution segfault, so
folding it in would conflate two bugs. Session creation is driven in a child
process so a glog CHECK abort (a regression) doesn't take down the pytest runner.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import onnx
import pytest
from onnx import TensorProto, helper

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
# AMDGPU umbrella EP (loads hip-backend.dll → hipgpu.dll underneath).
EP_DLL = REPO_ROOT / "install" / "dist" / "bin" / "amdgpu-ep.dll"

# Symbolic name carried by the output's data-dependent dim. Intentionally on no
# input -- that is the whole point of the repro.
OUTPUT_SYMBOL = "num_logical_patches"

# Signature of a regression that reintroduced a per-output-dim compile-time
# CHECK (the allocator ABI must size dynamic outputs in-graph instead).
_REGRESSION_SIGNATURE = (
    "static output shapes",
    "no graph input declares this symbolic name",
)


def _build_model(path: Path) -> None:
    """Graph whose output symbol is on no input and reaches the output through a
    downstream op (the realistic shape: a data-dependent extent propagated by a
    normal op, not the data-dependent op's own result).

    cond : bool[6]                              (static -> no input symbols)
    NonZero(cond)  -> nz  : int64[1, N]         (data-dependent extent N)
    Transpose(nz)  -> out : int64[N, 1]         (N carried to the output)
    """
    cond = helper.make_tensor_value_info("cond", TensorProto.BOOL, [6])
    out = helper.make_tensor_value_info("out", TensorProto.INT64, [OUTPUT_SYMBOL, 1])
    nz = helper.make_node("NonZero", ["cond"], ["nz"], name="nz")
    tr = helper.make_node("Transpose", ["nz"], ["out"], name="tr", perm=[1, 0])
    graph = helper.make_graph([nz, tr], "unresolved_output_symbol", [cond], [out])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    onnx.save(model, str(path))


# Child-process worker: register the EP, open a session. Session creation
# triggers the level-1 pass (build_metadata_json); reaching "SESSION_CREATED_OK"
# proves the metadata build did not abort on the data-dependent output symbol.
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
    so.add_provider_for_devices(devices, {"profile": "llm"})
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
    p = tmp_path_factory.mktemp("unresolved_dim") / "nonzero_transpose.onnx"
    _build_model(p)
    return p


@pytest.mark.skipif(not EP_DLL.exists(), reason="EP DLL not built — run build.py")
def test_compiles_unresolved_output_symbol(model_path):
    """The compile must succeed for a graph whose output symbol is on no input.

    The output-allocator ABI sizes dynamic outputs in-graph, so
    build_metadata_json emits the -1 shape verbatim with no compile-time CHECK;
    session creation completes cleanly. A regression that reintroduced a
    per-output-dim resolution CHECK would abort here and fail this test.
    """
    proc = _run_worker(model_path)
    combined = proc.stdout + "\n" + proc.stderr

    if any(sig in combined for sig in _REGRESSION_SIGNATURE):
        pytest.fail(
            "regression: compile aborted on a data-dependent output symbol that "
            f"is on no input.\nrc={proc.returncode}\n{combined}"
        )
    assert "SESSION_CREATED_OK" in proc.stdout, (
        f"session creation did not complete:\nrc={proc.returncode}\n{combined}"
    )
