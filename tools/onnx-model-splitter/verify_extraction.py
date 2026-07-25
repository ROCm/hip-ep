#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""
Verify extracted ONNX models: check file sizes, external data offsets,
graph structure (nodes, inputs, outputs), initializer integrity, and for
non-dynamic variants (filenames not ending with _dynamic.onnx):
  (1) phase-1 static scan: no symbolic / invalid dims on declared types;
  (2) only if phase-1 passes: phase-2 ONNX shape_inference (strict when
      supported), onnx.checker when applicable, then a second fixed-dim scan.
      check_model is skipped (not failed) for extended ops onnx does not
      register (e.g. SimplifiedLayerNormalization); ORT may still run them.

Usage:
  python verify_extraction.py <output_dir>
  python verify_extraction.py D:/liuc/0-modelzoo/llm/.../space
  python verify_extraction.py <output_dir> --skip-fixed-shape-check
  python verify_extraction.py <output_dir> --skip-shape-inference-check
"""

import os
import sys
import argparse
from collections import defaultdict

import onnx


def _is_dynamic_variant_filename(filename):
    """extract_submodels names variants as <stem>_<variant>.onnx; dynamic is *_dynamic.onnx."""
    return filename.endswith("_dynamic.onnx")


def _tensor_shape_dim_issues(shape):
    """Return human-readable issues for one TensorShapeProto (tensor dims)."""
    if not shape.dim:
        return []
    out = []
    for i, d in enumerate(shape.dim):
        if d.dim_param:
            out.append(f"dim[{i}] symbolic={d.dim_param!r}")
        elif d.dim_value < 0:
            out.append(f"dim[{i}] dim_value={d.dim_value}")
        elif d.dim_value == 0:
            out.append(f"dim[{i}] dim_value=0 (unspecified)")
    return out


def _vi_tensor_shape_issues(vi):
    if not vi.type.HasField("tensor_type"):
        return []
    tt = vi.type.tensor_type
    if not tt.HasField("shape"):
        return ["missing tensor_type.shape"]
    return _tensor_shape_dim_issues(tt.shape)


def _build_vi_map(graph):
    """Same merge order as extract_submodels.get_value_info_map."""
    vi = {}
    for v in graph.value_info:
        vi[v.name] = v
    for v in graph.input:
        vi[v.name] = v
    for v in graph.output:
        vi[v.name] = v
    return vi


def _collect_fixed_variant_shape_issues(model):
    """For fixed-shape ONNX files: list tensors / nodes that still have unfixed dims."""
    g = model.graph
    tensor_issues = defaultdict(list)
    init_by_name = {init.name: init for init in g.initializer}
    vi_map = _build_vi_map(g)

    def _add(name, issues):
        if not issues or not name:
            return
        for x in issues:
            if x not in tensor_issues[name]:
                tensor_issues[name].append(x)

    for vi in g.input:
        _add(vi.name, _vi_tensor_shape_issues(vi))
    for vi in g.output:
        _add(vi.name, _vi_tensor_shape_issues(vi))
    for vi in g.value_info:
        _add(vi.name, _vi_tensor_shape_issues(vi))

    for name, init in init_by_name.items():
        idims = []
        for i, d in enumerate(init.dims):
            if d <= 0:
                idims.append(f"dims[{i}]={d}")
        _add(name, idims)

    for node in g.node:
        for inp in node.input:
            if not inp or inp in init_by_name:
                continue
            vi = vi_map.get(inp)
            if vi is None:
                continue
            _add(inp, _vi_tensor_shape_issues(vi))

    lines = []
    for name in sorted(tensor_issues):
        issues = tensor_issues[name]
        if issues:
            lines.append(f"      tensor {name!r}: {', '.join(issues)}")

    bad_nodes = []
    for node in g.node:
        if any(inp and inp in tensor_issues for inp in node.input):
            bad_nodes.append(f"{node.name}({node.op_type})")
    if bad_nodes:
        head = ", ".join(bad_nodes[:24])
        if len(bad_nodes) > 24:
            head += f", ... (+{len(bad_nodes) - 24} more)"
        lines.append(
            f"      nodes consuming unfixed tensors ({len(bad_nodes)}): {head}"
        )

    return lines


def _infer_shapes_for_check(model):
    """Return inferred model with graceful fallback across ONNX APIs.

    Newer ONNX builds accept ``strict_mode=True`` and are useful to catch
    hard mismatches. Some real-world exports still carry stale ValueInfo
    annotations (for example Constant rank metadata) that only fail in strict
    mode while regular infer_shapes succeeds. For verification we therefore
    progressively relax inference before giving up.
    """
    infer = onnx.shape_inference.infer_shapes
    first_error = None

    try:
        return infer(model, check_type=True, strict_mode=True)
    except TypeError:
        # Older ONNX doesn't expose strict_mode.
        pass
    except Exception as e:
        first_error = e

    try:
        return infer(model, check_type=True)
    except TypeError:
        # Older ONNX doesn't expose check_type.
        pass
    except Exception as e:
        if first_error is None:
            first_error = e

    try:
        return infer(model)
    except Exception as e:
        if first_error is not None:
            raise first_error from e
        raise


def _graph_initializers_use_external_data(graph):
    return any(bool(init.external_data) for init in graph.initializer)


def _detect_ext_data_name_from_graph(graph):
    """Read the external data filename from the first external initializer."""
    for init in graph.initializer:
        if init.external_data:
            for entry in init.external_data:
                if entry.key == "location":
                    return entry.value
    return None


def _check_model_failure_is_extended_op_only(exc):
    """onnx.checker only knows built-in ops; MS ORT ops fail with 'No Op registered'."""
    msg = str(exc).lower()
    return (
        "no op registered" in msg
        or "no schema registered" in msg
        or "is not a registered function" in msg
    )


def _phase2_shape_normality_issues(model):
    """After phase-1 passed: ONNX shape inference + checker + residual unfixed dims."""
    lines = []
    try:
        inferred = _infer_shapes_for_check(model)
    except Exception as e:
        lines.append(f"      [phase-2] shape_inference failed: {e}")
        return lines

    if not _graph_initializers_use_external_data(inferred.graph):
        try:
            onnx.checker.check_model(inferred, full_check=False)
        except Exception as e:
            if not _check_model_failure_is_extended_op_only(e):
                lines.append(f"      [phase-2] check_model failed: {e}")
                return lines

    residual = _collect_fixed_variant_shape_issues(inferred)
    if residual:
        lines.append("      [phase-2] after inference, still unfixed or invalid dims:")
        lines.extend(residual)
    return lines


def _run_two_phase_shape_checks(model, run_phase2=True):
    """Phase 1: fixed dims on disk graph. Phase 2: only if phase 1 is clean."""
    phase1 = _collect_fixed_variant_shape_issues(model)
    if phase1:
        return False, [("phase-1", x) for x in phase1]
    if not run_phase2:
        return True, []
    phase2 = _phase2_shape_normality_issues(model)
    if phase2:
        return False, [("phase-2", x) for x in phase2]
    return True, []


def verify_single_op(d, check_fixed_shapes=True, check_shape_inference=True):
    """Verify single_op directory structure."""
    if not os.path.exists(d):
        print("  [NOT FOUND]")
        return True

    folders = sorted([f for f in os.listdir(d) if os.path.isdir(os.path.join(d, f))])
    print(f"  {len(folders)} folders:")
    for f in folders:
        cnt = len([x for x in os.listdir(os.path.join(d, f)) if x.endswith(".onnx")])
        print(f"    {f:50s} ({cnt} onnx files)")

    all_ok = True
    for folder in folders:
        folder_path = os.path.join(d, folder)
        for f in sorted(os.listdir(folder_path)):
            if not f.endswith(".onnx"):
                continue
            fpath = os.path.join(folder_path, f)
            try:
                m = onnx.load(fpath, load_external_data=False)
                n_nodes = len(m.graph.node)
                if n_nodes == 0:
                    print(f"    WARNING: {folder}/{f} has 0 nodes!")
                    all_ok = False
                if check_fixed_shapes and not _is_dynamic_variant_filename(f):
                    ok_shape, entries = _run_two_phase_shape_checks(
                        m, run_phase2=check_shape_inference
                    )
                    if not ok_shape:
                        print(f"    SHAPE (fixed variant): {folder}/{f}")
                        for _, line in entries:
                            print(line)
                        all_ok = False
            except Exception as e:
                print(f"    ERROR loading {folder}/{f}: {e}")
                all_ok = False

    return all_ok


def verify_external_data_dir(
    d, label, check_fixed_shapes=True, check_shape_inference=True
):
    """Verify a directory with external weights (single_layer or full_model)."""
    if not os.path.exists(d):
        print("  [NOT FOUND]")
        return True

    onnx_files = sorted([f for f in os.listdir(d) if f.endswith(".onnx")])

    weights_name = None
    for f in onnx_files:
        try:
            m = onnx.load(os.path.join(d, f), load_external_data=False)
            weights_name = _detect_ext_data_name_from_graph(m.graph)
            if weights_name:
                break
        except Exception:
            continue

    if not weights_name:
        print("  [No external data referenced by any .onnx file]")
        for f in onnx_files:
            sz = os.path.getsize(os.path.join(d, f))
            print(f"    {f:45s} {sz / (1024 * 1024):.1f} MB")
        return True

    weights_path = os.path.join(d, weights_name)
    if not os.path.exists(weights_path):
        print(f"  ERROR: external data file {weights_name!r} not found!")
        return False

    weights_size = os.path.getsize(weights_path)
    if weights_size > 1024**3:
        print(f"  {weights_name}: {weights_size / (1024**3):.2f} GB")
    else:
        print(f"  {weights_name}: {weights_size / (1024**2):.1f} MB")

    all_ok = True
    global_max_end = 0

    for f in onnx_files:
        fpath = os.path.join(d, f)
        try:
            m = onnx.load(fpath, load_external_data=False)
        except Exception as e:
            print(f"    {f:45s} ERROR loading: {e}")
            all_ok = False
            continue

        n_nodes = len(m.graph.node)
        n_inputs = len(m.graph.input)
        n_outputs = len(m.graph.output)

        ext_count = 0
        emb_count = 0
        max_end = 0
        bad_count = 0

        for init in m.graph.initializer:
            ext = {e.key: e.value for e in init.external_data}
            if ext:
                ext_count += 1
                offset = int(ext.get("offset", 0))
                length = int(ext.get("length", 0))
                end = offset + length
                if end > max_end:
                    max_end = end
                if end > weights_size:
                    bad_count += 1
            else:
                emb_count += 1

        if bad_count > 0:
            status = f"ERROR: {bad_count} tensors out of bounds!"
            all_ok = False
        else:
            status = "OK"

        print(
            f"    {f:40s} nodes={n_nodes:>4d}  in={n_inputs:>3d}  "
            f"out={n_outputs:>3d}  ext={ext_count}  emb={emb_count}  "
            f"max_end={max_end:>15d}  {status}"
        )
        if max_end > global_max_end:
            global_max_end = max_end

        if check_fixed_shapes and not _is_dynamic_variant_filename(f):
            ok_shape, entries = _run_two_phase_shape_checks(
                m, run_phase2=check_shape_inference
            )
            if not ok_shape:
                all_ok = False
                print(f"    SHAPE (fixed variant): {f}")
                for _, line in entries:
                    print(line)

    if all_ok:
        print(
            f"  >> ALL OFFSETS VALID (max_end = {weights_name} size: {global_max_end == weights_size})"
        )
    else:
        print("  >> ERRORS FOUND!")

    return all_ok


def main():
    parser = argparse.ArgumentParser(
        description="Verify extracted ONNX models (single_op, single_layer, full_model)"
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Base output directory containing single_op/, single_layer/, full_model/",
    )
    parser.add_argument(
        "--skip-fixed-shape-check",
        action="store_true",
        help="Do not verify that non-_dynamic.onnx files have fully fixed tensor shapes",
    )
    parser.add_argument(
        "--skip-shape-inference-check",
        action="store_true",
        help="After phase-1 passes, skip ONNX shape_inference + check_model (phase-2)",
    )
    args = parser.parse_args()
    check_shapes = not args.skip_fixed_shape_check
    check_infer = not args.skip_shape_inference_check

    base = args.output_dir
    if not os.path.isdir(base):
        print(f"ERROR: {base} is not a directory")
        sys.exit(1)

    overall_ok = True

    print("=" * 70)
    print("single_op")
    print("=" * 70)
    ok = verify_single_op(
        os.path.join(base, "single_op"),
        check_fixed_shapes=check_shapes,
        check_shape_inference=check_infer if check_shapes else False,
    )
    if not ok:
        overall_ok = False
    print()

    print("=" * 70)
    print("single_layer")
    print("=" * 70)
    ok = verify_external_data_dir(
        os.path.join(base, "single_layer"),
        "single_layer",
        check_fixed_shapes=check_shapes,
        check_shape_inference=check_infer if check_shapes else False,
    )
    if not ok:
        overall_ok = False
    print()

    print("=" * 70)
    print("full_model")
    print("=" * 70)
    ok = verify_external_data_dir(
        os.path.join(base, "full_model"),
        "full_model",
        check_fixed_shapes=check_shapes,
        check_shape_inference=check_infer if check_shapes else False,
    )
    if not ok:
        overall_ok = False
    print()

    print("=" * 70)
    if overall_ok:
        print("RESULT: ALL CHECKS PASSED")
    else:
        print("RESULT: SOME CHECKS FAILED")
    print("=" * 70)

    sys.exit(0 if overall_ok else 1)


if __name__ == "__main__":
    main()
