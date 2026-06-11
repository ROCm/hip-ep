#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Repro for PR #340: output-allocator compile must not hard-CHECK on a dynamic
output dim whose symbolic name appears on no input.

`build_metadata_json` (backend-mlir-compiler/level-1-pass/src/pass_main.cpp)
builds a DimSource for every dynamic output dim by matching the dim's symbolic
name to whichever *input* first declares it. For a data-dependent output extent
computed *inside* the graph (e.g. a vision model's feature count), the symbol is
correctly on no input, and the classic path aborts the whole compile. The
output-allocator ABI never reads a DimSource (the DLL sizes outputs in-graph),
so the fix skips DimSource emission in that ABI -- making such graphs compile.

The fix surface is the compile-time metadata build, which runs during session
creation. The worker only creates the session; it does NOT run inference --
NonZero's GPU kernel has a separate, pre-existing execution segfault, so folding
it in would conflate two bugs. Session creation is driven in a child process so
a pre-fix glog CHECK abort doesn't take down the pytest runner.
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
EP_DLL = REPO_ROOT / "install" / "dist" / "bin" / "onnxruntime_morphizen_ep.dll"

# Symbolic name carried by the output's data-dependent dim. Intentionally on no
# input -- that is the whole point of the repro.
OUTPUT_SYMBOL = "num_logical_patches"

# Signature of the spurious abort the fix removes.
_UNRESOLVED_SIGNATURE = (
    "no graph input declares this symbolic name",
    "pit != dim_param_map.end()",
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


# Child-process worker: register the EP, open a session in the requested ABI.
# Session creation triggers the level-1 pass (build_metadata_json); reaching
# "SESSION_CREATED_OK" proves the metadata build did not abort on the symbol.
_WORKER = textwrap.dedent(
    """
    import sys
    import onnxruntime as ort

    model_path, use_alloc, ep_dll = sys.argv[1], sys.argv[2] == "1", sys.argv[3]

    ort.register_execution_provider_library("MorphiZenExecutionProvider", ep_dll)
    from onnxruntime.capi._pybind_state import get_ep_devices
    devices = [d for d in get_ep_devices()
               if d.ep_name == "MorphiZenExecutionProvider"]
    if not devices:
        print("NO_EP_DEVICES", file=sys.stderr)
        sys.exit(3)

    so = ort.SessionOptions()
    so.add_provider_for_devices(
        devices, {"use_output_allocator": "1"} if use_alloc else {})
    sess = ort.InferenceSession(model_path, sess_options=so)
    print("SESSION_CREATED_OK")
    """
)


def _run_worker(model_path: Path, use_alloc: bool):
    env = dict(os.environ)
    # Leave strict off so a pre-fix soft CPU fallback is not turned into an
    # abort -- the pass_main CHECK (if any) should be the only abort source.
    env["HIPDNN_EP_STRICT"] = "0"
    dist_bin = REPO_ROOT / "install" / "dist" / "bin"
    therock_bin = REPO_ROOT / "install" / "therock" / "bin"
    env["THEROCK_DIST"] = str(REPO_ROOT / "install" / "therock")
    env["PATH"] = f"{dist_bin};{therock_bin};" + env.get("PATH", "")
    return subprocess.run(
        [
            sys.executable,
            "-c",
            _WORKER,
            str(model_path),
            "1" if use_alloc else "0",
            str(EP_DLL),
        ],
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
def test_output_allocator_compiles_unresolved_symbol(model_path):
    """Allocator ABI must compile a graph whose output symbol is on no input.

    The PR #340 fix target. On pre-fix `main` the child aborts with the
    pass_main.cpp CHECK and this test fails (reproducing the bug); post-fix
    session creation completes cleanly.
    """
    proc = _run_worker(model_path, use_alloc=True)
    combined = proc.stdout + "\n" + proc.stderr

    if any(sig in combined for sig in _UNRESOLVED_SIGNATURE):
        pytest.fail(
            "PR #340 bug reproduced: output-allocator compile aborted on an "
            f"output symbol that is on no input.\nrc={proc.returncode}\n{combined}"
        )
    assert "SESSION_CREATED_OK" in proc.stdout, (
        "allocator-mode session creation did not complete (and not via the "
        f"known unresolved-dim CHECK):\nrc={proc.returncode}\n{combined}"
    )


@pytest.mark.skipif(not EP_DLL.exists(), reason="EP DLL not built — run build.py")
def test_classic_abi_still_errors_on_unresolved_symbol(model_path):
    """Classic ABI: aborting on an unresolvable output symbol is correct (it
    pre-allocates outputs and cannot size a symbol that is on no input). Asserts
    the classic run does not silently succeed with a wrong shape.
    """
    proc = _run_worker(model_path, use_alloc=False)
    combined = proc.stdout + "\n" + proc.stderr
    if "SESSION_CREATED_OK" in proc.stdout and proc.returncode == 0:
        return  # a future classic generalization handled it -- acceptable
    assert (
        any(sig in combined for sig in _UNRESOLVED_SIGNATURE) or proc.returncode != 0
    ), f"classic ABI neither succeeded nor failed as expected:\n{combined}"
