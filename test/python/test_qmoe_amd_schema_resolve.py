#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""ORT must be able to build a session for a single com.amd::QMoE node.

This is the minimal end-to-end check for the schema-only custom-op
registration module (backend-mlir-compiler/custom-op-schema/): a graph
containing one com.amd::QMoE node, with the 15 inputs and dtypes from the
registered schema (see
backend-mlir-compiler/custom-op-schema/ops/qmoe_schema.cpp), must resolve and
build a session against hipgpu.dll -- proving ORT recognizes the op and
can construct a kernel for it (the schema-only stub kernel) -- without
requiring the HIP dialect op / conversion / lowering / real kernel (that
work is intentionally out of scope here: the EP does not yet claim this
node in GetCapability, so it falls back to the CPU custom-op kernel path).

The worker only creates the session; it does NOT run inference. Running
would reach SchemaOnlyStubKernel::Compute(), which deliberately throws
ORT_NOT_IMPLEMENTED because a real per-op kernel does not exist yet.

Because the HIP dialect op / conversion / lowering for QMoE does
not exist yet, hip-ep still attempts to claim and compile the node (it
sees a registered com.amd::QMoE schema) and its MLIR pipeline fails with
"op was not bufferized" -- logged to stderr as an expected "MLIR
compilation failed, skipping" message, not a test failure. MorphiZenEP
then declines the node and ORT falls back to the CPU custom-op kernel
(the schema-only stub), which is what this test actually verifies works.
Session creation is driven in a child process so an EP-load failure
(e.g. a missing dependent DLL) doesn't take down the pytest runner.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# hip-ep's own EP dll (schema-only QMoE registration is linked directly
# into it, see backend-mlir-compiler/custom-op-schema/CMakeLists.txt).
# Overridable so this also works against a differently located install prefix.
EP_DLL = Path(
    os.environ.get("HIPEP_QMOE_TEST_DLL", r"C:\workspace\local\bin\hipgpu.dll")
)

# index -> (name, dtype). Must match backend-mlir-compiler/custom-op-schema/ops/qmoe_schema.cpp.
_QMOE_INPUTS = [
    ("hidden_states", TensorProto.FLOAT16),
    ("fc1_expert_weights", TensorProto.UINT8),
    ("fc1_expert_scales", TensorProto.FLOAT16),
    ("fc2_expert_weights", TensorProto.UINT8),
    ("fc2_expert_scales", TensorProto.FLOAT16),
    ("fc1_latent_weights", TensorProto.UINT8),
    ("fc1_latent_scales", TensorProto.FLOAT16),
    ("fc2_latent_weights", TensorProto.UINT8),
    ("fc2_latent_scales", TensorProto.FLOAT16),
    ("shared_fc1_weights", TensorProto.UINT8),
    ("shared_fc1_scales", TensorProto.FLOAT16),
    ("shared_fc2_weights", TensorProto.UINT8),
    ("shared_fc2_scales", TensorProto.FLOAT16),
    ("router_weight", TensorProto.FLOAT16),
    ("correction_bias", TensorProto.FLOAT16),
]

_HIDDEN_SHAPE = [1, 4, 8]


def _make_initializer(name: str, dtype: int) -> onnx.TensorProto:
    """Tiny placeholder tensor; shapes are not schema-constrained."""
    np_dtype = np.uint8 if dtype == TensorProto.UINT8 else np.float16
    data = np.zeros([2, 2], dtype=np_dtype)
    return helper.make_tensor(
        name, dtype, data.shape, data.flatten().tolist(), raw=False
    )


def _build_model(path: Path) -> None:
    hidden_states = helper.make_tensor_value_info(
        "hidden_states", TensorProto.FLOAT16, _HIDDEN_SHAPE
    )
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT16, _HIDDEN_SHAPE)

    initializers = [_make_initializer(name, dtype) for name, dtype in _QMOE_INPUTS[1:]]

    node = helper.make_node(
        "QMoE",
        [name for name, _ in _QMOE_INPUTS],
        ["output"],
        domain="com.amd",
        name="qmoe_amd_min",
    )

    graph = helper.make_graph(
        [node],
        "qmoe_amd_schema_resolve",
        [hidden_states],
        [output],
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 21),
            helper.make_opsetid("com.amd", 1),
        ],
    )
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


# Child-process worker: register hipgpu.dll, open a session against the
# minimal com.amd::QMoE graph. Reaching "SESSION_CREATED_OK" proves the
# schema is registered and a (stub) kernel could be built for the node.
_WORKER = textwrap.dedent(
    """
    import sys
    import onnxruntime as ort

    model_path, ep_dll = sys.argv[1], sys.argv[2]

    ort.register_execution_provider_library("MorphiZenExecutionProvider", ep_dll)
    from onnxruntime.capi._pybind_state import get_ep_devices
    devices = [d for d in get_ep_devices()
               if d.ep_name == "MorphiZenExecutionProvider"]
    if not devices:
        print("NO_EP_DEVICES", file=sys.stderr)
        sys.exit(3)

    so = ort.SessionOptions()
    so.add_provider_for_devices(devices, {})
    sess = ort.InferenceSession(model_path, sess_options=so)
    print("SESSION_CREATED_OK")
    """
)


def _run_worker(model_path: Path):
    env = dict(os.environ)
    # Leave strict off: this test only cares that the node resolves and a
    # kernel can be built, not about which provider ends up owning it. Note
    # CompilerDriver.cpp checks non-empty, not "1" -- pop it, don't set "0".
    env.pop("HIPDNN_EP_STRICT", None)
    therock_dist = env.get("THEROCK_DIST", r"C:\workspace\therock")
    env["THEROCK_DIST"] = therock_dist
    env["PATH"] = f"{EP_DLL.parent};{therock_dist}\\bin;" + env.get("PATH", "")
    return subprocess.run(
        [sys.executable, "-c", _WORKER, str(model_path), str(EP_DLL)],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )


@pytest.fixture(scope="module")
def model_path(tmp_path_factory):
    p = tmp_path_factory.mktemp("qmoe_amd_schema") / "qmoe_amd_min.onnx"
    _build_model(p)
    return p


@pytest.mark.skipif(
    not EP_DLL.exists(), reason="hipgpu.dll not built — run build-hip-ep.bat"
)
def test_qmoe_amd_schema_resolves(model_path):
    """A single com.amd::QMoE node must resolve and build a session.

    This proves the schema-only registration (register_custom_op.hpp,
    SchemaOnlyCustomOpBase, qmoe_schema.cpp, and the morphizen-core /
    ort-bridge wiring that surfaces it via GetCustomOpDomains) makes ORT
    recognize com.amd::QMoE end to end, ahead of the dialect op /
    conversion / lowering / real kernel work (out of scope here).
    """
    proc = _run_worker(model_path)
    combined = proc.stdout + "\n" + proc.stderr
    assert "SESSION_CREATED_OK" in proc.stdout, (
        f"session creation did not complete:\nrc={proc.returncode}\n{combined}"
    )
