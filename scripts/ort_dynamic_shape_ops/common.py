#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Shared harness for verifying ONNX Runtime output-tensor allocation.

Question under investigation
----------------------------
``Range``, ``ConstantOfShape`` and ``NonZero`` are *data-dependent shape*
ops: the OUTPUT shape is not knowable from the input *shapes* alone -- it
depends on the input *values* / *data* at runtime.

  * Range(start, limit, delta) -> 1-D, length = ceil((limit-start)/delta)
  * ConstantOfShape(shape)     -> shape == the int values inside `shape`
  * NonZero(x)                 -> [rank(x), N], N = #non-zero elements of x

So the natural question for anyone marshalling these ops (e.g. inside an
execution provider) is: *must the caller pre-allocate the output buffer,
or does ONNX Runtime allocate it?*

This module runs each op three ways against the **CPU EP** and reports
what happens:

  [1] session.run(None, feeds)
        The classic API. ORT owns allocation and hands back numpy arrays.

  [2] IOBinding + bind_output(name, "cpu")     (no shape/buffer given)
        Caller states only the device. ORT still allocates the output,
        sized to whatever the kernel produced at runtime.

  [3] IOBinding + bind_ortvalue_output(name, ov)   (caller pre-allocates)
        Caller binds its own, pre-sized OrtValue. We compare the bound
        buffer's data pointer against the data pointer of the OrtValue
        ORT returns, to see whether ORT *reused* the caller buffer or
        *replaced* it -- and what happens when the pre-sized buffer is
        the wrong size for the data-dependent result.
"""

import numpy as np
import onnxruntime as ort
from onnx import ModelProto, shape_inference

SEP = "=" * 74


# --------------------------------------------------------------------------- #
# Session + introspection helpers
# --------------------------------------------------------------------------- #
def make_cpu_session(model: ModelProto) -> ort.InferenceSession:
    """Build a CPU-EP-only InferenceSession from an in-memory model."""
    so = ort.SessionOptions()
    # The wrong-sized pre-bound-buffer cases in experiment [3] make a kernel
    # return a non-OK status, which ORT logs at ERROR level to stderr. We
    # already catch and print that failure cleanly, so silence ORT's own log
    # (4 == FATAL only) to keep the report readable.
    so.log_severity_level = 4
    return ort.InferenceSession(
        model.SerializeToString(),
        sess_options=so,
        providers=["CPUExecutionProvider"],
    )


def _fmt_dims(shape):
    # ORT NodeArg.shape is a list of int | str (symbolic) | None (unknown
    # dim); the whole shape is None when the rank itself is unknown.
    if shape is None:
        return "<unknown rank>"
    return [d if d is not None else "?" for d in shape]


def print_declared_io(session: ort.InferenceSession) -> None:
    print("  inputs:")
    for i in session.get_inputs():
        print(f"    {i.name:12s} {i.type:16s} shape={_fmt_dims(i.shape)}")
    print("  outputs (shape ORT resolved statically at session build):")
    for o in session.get_outputs():
        print(f"    {o.name:12s} {o.type:16s} shape={_fmt_dims(o.shape)}")


def show_static_shape_inference(model: ModelProto) -> None:
    """Print what onnx static shape inference can (not) determine for the
    graph outputs -- the '?'/symbolic dims are exactly the data-dependent
    sizes the caller cannot know ahead of execution."""
    inferred = shape_inference.infer_shapes(model)
    for vi in inferred.graph.output:
        tt = vi.type.tensor_type
        if not tt.HasField("shape"):
            print(f"    {vi.name}: <no shape inferred>")
            continue
        dims = []
        for d in tt.shape.dim:
            if d.HasField("dim_value"):
                dims.append(d.dim_value)
            elif d.HasField("dim_param"):
                dims.append(d.dim_param or "?")
            else:
                dims.append("?")
        print(f"    {vi.name}: {dims}")


# --------------------------------------------------------------------------- #
# The three calling conventions
# --------------------------------------------------------------------------- #
def run_via_run(session: ort.InferenceSession, feeds: dict) -> dict:
    names = [o.name for o in session.get_outputs()]
    results = session.run(names, feeds)
    return dict(zip(names, results))


def run_via_iobinding_device_alloc(
    session: ort.InferenceSession, feeds: dict, device: str = "cpu"
) -> dict:
    """Bind inputs, then bind each output to a *device only* (no shape, no
    buffer). ORT must allocate the data-dependent output itself."""
    io = session.io_binding()
    for name, arr in feeds.items():
        io.bind_cpu_input(name, arr)
    names = [o.name for o in session.get_outputs()]
    for name in names:
        io.bind_output(name, device)
    session.run_with_iobinding(io)
    return dict(zip(names, io.get_outputs()))


def _data_ptr(ov):
    try:
        return ov.data_ptr()
    except Exception:  # noqa: BLE001 - API not present on this build
        return None


def run_via_iobinding_prealloc(
    session: ort.InferenceSession,
    feeds: dict,
    output_name: str,
    prealloc: np.ndarray,
) -> dict:
    """Bind a caller-owned, pre-sized buffer as the output. Report whether
    ORT wrote into that buffer (pointer preserved) or replaced it, and
    surface any error ORT raises for a wrong-sized buffer."""
    io = session.io_binding()
    for name, arr in feeds.items():
        io.bind_cpu_input(name, arr)

    ov = ort.OrtValue.ortvalue_from_numpy(prealloc)
    before_ptr = _data_ptr(ov)

    bound_order = [output_name]
    io.bind_ortvalue_output(output_name, ov)
    for o in session.get_outputs():
        if o.name != output_name:
            io.bind_output(o.name, "cpu")
            bound_order.append(o.name)

    error = None
    out = None
    after_ptr = None
    try:
        session.run_with_iobinding(io)
        out = dict(zip(bound_order, io.get_outputs()))[output_name]
        after_ptr = _data_ptr(out)
    except Exception as e:  # noqa: BLE001 - we want to observe ORT's reaction
        error = e

    return {
        "before_ptr": before_ptr,
        "after_ptr": after_ptr,
        "out_ov": out,
        "error": error,
    }


# --------------------------------------------------------------------------- #
# Top-level battery: run all conventions for one op and print a report
# --------------------------------------------------------------------------- #
def _block(arr: np.ndarray) -> str:
    body = np.array2string(arr, threshold=64, max_line_width=70)
    return "        " + body.replace("\n", "\n        ")


def verify_op(
    title: str,
    description: str,
    model: ModelProto,
    feeds: dict,
    prealloc_cases: list,
) -> None:
    """Run the full battery for a single-output op.

    prealloc_cases: list of (label, numpy_array) buffers to bind as the
    output in experiment [3].
    """
    print(SEP)
    print(title)
    print(SEP)
    print(description.strip("\n"))

    print("\n-- static shape inference (what's known at graph-build time) --")
    show_static_shape_inference(model)

    sess = make_cpu_session(model)
    print("\n-- session I/O (CPU EP) --")
    print_declared_io(sess)

    out_name = sess.get_outputs()[0].name
    feed_desc = ", ".join(f"{k}={np.asarray(v).tolist()}" for k, v in feeds.items())
    print(f"\n-- feeds --\n    {feed_desc}")

    print("\n[1] session.run(None, feeds)  ->  ORT allocates the result")
    r = run_via_run(sess, feeds)[out_name]
    print(f"    result: shape={list(r.shape)} dtype={r.dtype}")
    print(_block(r))

    print("\n[2] IOBinding bind_output(name,'cpu')  ->  ORT allocates (no size given)")
    ov = run_via_iobinding_device_alloc(sess, feeds)[out_name]
    same_shape = list(ov.shape()) == list(r.shape)
    print(f"    result: shape={list(ov.shape())}  (matches run()? {same_shape})")

    for label, arr in prealloc_cases:
        print(
            f"\n[3] IOBinding bind_ortvalue_output(name, caller_buf)  "
            f"[{label}: shape={list(arr.shape)}]"
        )
        res = run_via_iobinding_prealloc(sess, feeds, out_name, arr)
        if res["error"] is not None:
            print(
                f"    -> ORT RAISED {type(res['error']).__name__}: "
                f"{str(res['error']).strip()[:200]}"
            )
            continue
        bp, ap = res["before_ptr"], res["after_ptr"]
        reused = bp is not None and ap is not None and bp == ap
        print(f"    -> ran OK; result shape={list(res['out_ov'].shape())}")
        if bp is not None and ap is not None:
            print(f"    -> caller_buf ptr=0x{bp:x}  result ptr=0x{ap:x}")
            print(f"    -> ORT reused the caller buffer? {reused}")
        else:
            print("    -> (OrtValue.data_ptr() unavailable on this build)")
    print()


def verify_dynamic_shapes(
    title: str,
    description: str,
    model: ModelProto,
    cases: list,
) -> None:
    """Drive ONE session with several different runtime input shapes.

    The model is built with symbolic ('?') input dims, so the concrete shape
    is bound per call -- no recompile, no per-shape session. For each case we
    run BOTH session.run and IOBinding(device-alloc) and confirm they agree
    on the data-dependent output shape ORT allocated.

    Then we re-test the pre-allocation question under shape change: a single
    caller buffer sized for the FIRST case is bound across every run, showing
    it is reused only for the run whose shape matches and is rejected for the
    others -- i.e. a fixed pre-allocated buffer cannot serve a data-dependent
    op once the shape moves.

    cases: list of (label, feeds_dict).
    """
    print(SEP)
    print(title)
    print(SEP)
    print(description.strip("\n"))

    print("\n-- static shape inference (what's known at graph-build time) --")
    show_static_shape_inference(model)

    sess = make_cpu_session(model)
    print("\n-- session I/O (CPU EP) --")
    print_declared_io(sess)

    out_name = sess.get_outputs()[0].name
    print("\n-- ONE session, multiple runtime input shapes (ORT allocates) --")
    first_out = None
    for label, feeds in cases:
        r = run_via_run(sess, feeds)[out_name]
        ov = run_via_iobinding_device_alloc(sess, feeds)[out_name]
        match = list(r.shape) == list(ov.shape())
        if first_out is None:
            first_out = r
        print(f"  [{label}]")
        for k, v in feeds.items():
            a = np.asarray(v)
            if a.size <= 12:
                print(f"      in : {k}={a.tolist()}  (shape {list(a.shape)})")
            else:
                print(f"      in : {k} shape {list(a.shape)}")
        print(
            f"      out: shape={list(r.shape)} dtype={r.dtype}  "
            f"(run()==IOBinding: {match}; ORT allocated both)"
        )
        if r.size <= 24:
            print(f"      out: values={r.tolist()}")

    # Pre-allocation under shape change: bind ONE caller buffer (sized for the
    # first case) to every run. It only fits the matching run.
    if first_out is not None:
        buf = np.zeros_like(first_out)
        first_label = cases[0][0]
        print(
            f"\n-- reuse ONE pre-bound caller buffer (shape {list(buf.shape)}, "
            f"sized for '{first_label}') across every run --"
        )
        for label, feeds in cases:
            res = run_via_iobinding_prealloc(sess, feeds, out_name, buf)
            if res["error"] is not None:
                print(
                    f"  [{label}] -> ORT RAISED: needed shape differs from the "
                    f"pre-bound {list(buf.shape)} (shape verification failed)"
                )
            else:
                bp, ap = res["before_ptr"], res["after_ptr"]
                reused = bp is not None and ap is not None and bp == ap
                print(f"  [{label}] -> ran OK; ORT reused the pre-bound buffer? {reused}")
    print()
