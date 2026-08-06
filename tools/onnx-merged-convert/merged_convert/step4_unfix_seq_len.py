#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper, shape_inference

SEQ_LEN_PARAM = "seq_len"
DEFAULT_STATIC_SEQ_LENS = (1, 128)


def _dim_to_obj(dim) -> int | str | None:
    if dim.dim_param:
        return dim.dim_param
    if dim.dim_value:
        return int(dim.dim_value)
    return None


def _shape_tuple(vi: onnx.ValueInfoProto) -> tuple | None:
    tt = vi.type.tensor_type
    if not tt.HasField("shape"):
        return None
    return tuple((_dim_to_obj(d) for d in tt.shape.dim))


def _load_model(path: Path, *, load_external: bool) -> onnx.ModelProto:
    return onnx.load(str(path), load_external_data=load_external)


def _maybe_infer_shapes(model: onnx.ModelProto) -> onnx.ModelProto:
    try:
        return shape_inference.infer_shapes(model)
    except Exception as exc:
        print(f"WARNING: shape inference failed: {exc}", file=sys.stderr)
        return model


def _collect_shapes(model: onnx.ModelProto) -> dict[str, tuple]:
    shapes: dict[str, tuple] = {}
    graph = model.graph
    for vi in list(graph.input) + list(graph.output) + list(graph.value_info):
        shape = _shape_tuple(vi)
        if shape is not None:
            shapes[vi.name] = shape
    return shapes


def _initializer_map(
    model: onnx.ModelProto, *, load_values: bool
) -> dict[str, np.ndarray | None]:
    out: dict[str, np.ndarray | None] = {}
    for init in model.graph.initializer:
        if load_values:
            out[init.name] = numpy_helper.to_array(init)
        else:
            out[init.name] = None
    return out


@dataclass
class SeqLenAxis:
    tensor: str
    axis: int
    value_a: int
    value_b: int


@dataclass
class CompareReport:
    model_a: str
    model_b: str
    node_count_a: int
    node_count_b: int
    graph_structure_match: bool
    initializer_value_diffs: int
    tensor_shape_diffs: list[tuple[str, tuple, tuple]] = field(default_factory=list)
    seq_len_axes: list[SeqLenAxis] = field(default_factory=list)
    node_io_diffs: list[dict] = field(default_factory=list)
    axis_pattern_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "model_a": self.model_a,
            "model_b": self.model_b,
            "node_count_a": self.node_count_a,
            "node_count_b": self.node_count_b,
            "graph_structure_match": self.graph_structure_match,
            "initializer_value_diffs": self.initializer_value_diffs,
            "tensor_shape_diff_count": len(self.tensor_shape_diffs),
            "tensor_shape_diffs": [
                {"tensor": n, "shape_a": list(a), "shape_b": list(b)}
                for n, a, b in self.tensor_shape_diffs
            ],
            "seq_len_axes": [
                {
                    "tensor": s.tensor,
                    "axis": s.axis,
                    "value_a": s.value_a,
                    "value_b": s.value_b,
                }
                for s in self.seq_len_axes
            ],
            "node_io_diff_count": len(self.node_io_diffs),
            "node_io_diffs": self.node_io_diffs,
            "axis_pattern_counts": self.axis_pattern_counts,
        }


def _graph_structure_match(model_a: onnx.ModelProto, model_b: onnx.ModelProto) -> bool:
    nodes_a = model_a.graph.node
    nodes_b = model_b.graph.node
    if len(nodes_a) != len(nodes_b):
        return False
    for na, nb in zip(nodes_a, nodes_b):
        if (
            na.op_type != nb.op_type
            or na.name != nb.name
            or na.input != nb.input
            or (na.output != nb.output)
        ):
            return False
    return True


def _detect_seq_len_axes(
    shape_a: tuple, shape_b: tuple, tensor_name: str, static_seq_lens: tuple[int, ...]
) -> list[SeqLenAxis]:
    hits: list[SeqLenAxis] = []
    if len(shape_a) != len(shape_b):
        return hits
    values = set(static_seq_lens)
    for axis, (va, vb) in enumerate(zip(shape_a, shape_b)):
        if (
            isinstance(va, int)
            and isinstance(vb, int)
            and (va in values)
            and (vb in values)
            and (va != vb)
        ):
            hits.append(
                SeqLenAxis(tensor=tensor_name, axis=axis, value_a=va, value_b=vb)
            )
    return hits


def compare_models(
    path_a: Path,
    path_b: Path,
    *,
    infer_shapes: bool = True,
    static_seq_lens: tuple[int, ...] = DEFAULT_STATIC_SEQ_LENS,
    load_external: bool = True,
) -> CompareReport:
    model_a = _load_model(path_a, load_external=load_external)
    model_b = _load_model(path_b, load_external=load_external)
    if infer_shapes:
        model_a = _maybe_infer_shapes(model_a)
        model_b = _maybe_infer_shapes(model_b)
    shapes_a = _collect_shapes(model_a)
    shapes_b = _collect_shapes(model_b)
    tensor_diffs: list[tuple[str, tuple, tuple]] = []
    seq_axes: list[SeqLenAxis] = []
    axis_counter: Counter[tuple[int, int, int]] = Counter()
    for name in sorted(set(shapes_a) | set(shapes_b)):
        sa, sb = (shapes_a.get(name), shapes_b.get(name))
        if sa == sb:
            continue
        if sa is None or sb is None:
            tensor_diffs.append((name, sa or (), sb or ()))
            continue
        tensor_diffs.append((name, sa, sb))
        for hit in _detect_seq_len_axes(sa, sb, name, static_seq_lens):
            seq_axes.append(hit)
            axis_counter[hit.axis, hit.value_a, hit.value_b] += 1
    node_io_diffs: list[dict] = []
    nodes_a = list(model_a.graph.node)
    nodes_b = list(model_b.graph.node)
    for idx, (na, nb) in enumerate(zip(nodes_a, nodes_b)):
        if na.op_type != nb.op_type or na.name != nb.name:
            node_io_diffs.append(
                {
                    "node_index": idx,
                    "kind": "structure",
                    "node_a": na.name,
                    "node_b": nb.name,
                    "op_a": na.op_type,
                    "op_b": nb.op_type,
                }
            )
            continue
        for io_kind, names in (("input", na.input), ("output", na.output)):
            for tensor in names:
                if not tensor:
                    continue
                sa, sb = (shapes_a.get(tensor), shapes_b.get(tensor))
                if sa != sb:
                    node_io_diffs.append(
                        {
                            "node_index": idx,
                            "kind": io_kind,
                            "node": na.name,
                            "op_type": na.op_type,
                            "tensor": tensor,
                            "shape_a": list(sa) if sa else None,
                            "shape_b": list(sb) if sb else None,
                        }
                    )
    init_a = _initializer_map(model_a, load_values=load_external)
    init_b = _initializer_map(model_b, load_values=load_external)
    init_diffs = 0
    if load_external:
        for name in sorted(set(init_a) & set(init_b)):
            a, b = (init_a[name], init_b[name])
            if a is None or b is None:
                continue
            if a.shape != b.shape or a.dtype != b.dtype or (not np.array_equal(a, b)):
                init_diffs += 1
    return CompareReport(
        model_a=str(path_a),
        model_b=str(path_b),
        node_count_a=len(nodes_a),
        node_count_b=len(nodes_b),
        graph_structure_match=_graph_structure_match(model_a, model_b),
        initializer_value_diffs=init_diffs,
        tensor_shape_diffs=tensor_diffs,
        seq_len_axes=seq_axes,
        node_io_diffs=node_io_diffs,
        axis_pattern_counts={
            f"axis={axis} ({va} vs {vb})": count
            for (axis, va, vb), count in axis_counter.most_common()
        },
    )


def _set_dim_param(vi: onnx.ValueInfoProto, axis: int, param: str) -> None:
    dim = vi.type.tensor_type.shape.dim[axis]
    dim.ClearField("dim_value")
    dim.dim_param = param


def _apply_seq_len_bindings(
    model: onnx.ModelProto,
    bindings: dict[str, set[int]],
    *,
    seq_len_param: str,
    clear_value_info: bool,
) -> int:
    """Replace marked axes with ``seq_len_param`` on graph inputs/outputs."""
    changed = 0
    graph = model.graph
    if clear_value_info:
        del graph.value_info[:]
    for vi in list(graph.input) + list(graph.output):
        shape = _shape_tuple(vi)
        if shape is None:
            continue
        axes = bindings.get(vi.name)
        if not axes:
            continue
        for axis in sorted(axes):
            if axis >= len(shape):
                continue
            _set_dim_param(vi, axis, seq_len_param)
            changed += 1
    return changed


def _external_data_location(model: onnx.ModelProto) -> str | None:
    locations = {
        entry.value
        for init in model.graph.initializer
        for entry in init.external_data
        if entry.key == "location"
    }
    if len(locations) == 1:
        return locations.pop()
    return None


def _save_unfixed_model(model: onnx.ModelProto, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    ext_name = _external_data_location(model)
    if ext_name is not None:
        data_path = out_path.parent / ext_name
        if not data_path.exists():
            raise FileNotFoundError(
                f"External weights not found for {out_path.name}: {data_path}"
            )
        onnx.save_model(
            model,
            str(out_path),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=ext_name,
            size_threshold=65536,
        )
        return
    onnx.save(model, str(out_path))


def unfix_seq_len(
    src_path: Path,
    ref_path_a: Path,
    ref_path_b: Path,
    out_path: Path,
    *,
    static_seq_lens: tuple[int, ...] = DEFAULT_STATIC_SEQ_LENS,
    seq_len_param: str = SEQ_LEN_PARAM,
    clear_value_info: bool = True,
    reinfer_shapes: bool = False,
) -> CompareReport:
    report = compare_models(
        ref_path_a,
        ref_path_b,
        infer_shapes=True,
        static_seq_lens=static_seq_lens,
        load_external=False,
    )
    if not report.graph_structure_match:
        raise ValueError(
            f"Graph structure mismatch between {ref_path_a.name} and {ref_path_b.name}"
        )
    if report.initializer_value_diffs:
        raise ValueError(
            f"Expected identical initializers, found {report.initializer_value_diffs} diffs"
        )
    if not report.seq_len_axes:
        raise ValueError("No seq_len candidate axes detected between reference models")
    bindings: dict[str, set[int]] = defaultdict(set)
    for hit in report.seq_len_axes:
        bindings[hit.tensor].add(hit.axis)
    io_names = {
        vi.name for vi in _load_model(src_path, load_external=False).graph.input
    }
    io_names.update(
        (vi.name for vi in _load_model(src_path, load_external=False).graph.output)
    )
    io_bindings = {name: axes for name, axes in bindings.items() if name in io_names}
    if not io_bindings:
        raise ValueError(
            "Seq_len axes were found only on intermediates; missing graph I/O bindings"
        )
    model = _load_model(src_path, load_external=False)
    changed = _apply_seq_len_bindings(
        model,
        io_bindings,
        seq_len_param=seq_len_param,
        clear_value_info=clear_value_info,
    )
    if reinfer_shapes:
        model = _maybe_infer_shapes(model)
    _save_unfixed_model(model, out_path)
    print(f"Wrote dynamic decoder: {out_path}")
    print(f"  rebound {changed} input/output dimension(s) to '{seq_len_param}'")
    for name in sorted(io_bindings):
        axes = sorted(io_bindings[name])
        print(f"  {name}: axis {axes} -> {seq_len_param}")
    if clear_value_info:
        print("  cleared stale graph.value_info")
    return report


def _print_compare_report(report: CompareReport, *, json_out: Path | None) -> None:
    print(f"Model A: {report.model_a}")
    print(f"Model B: {report.model_b}")
    print(f"Nodes: {report.node_count_a} / {report.node_count_b}")
    print(f"Graph structure match: {report.graph_structure_match}")
    print(f"Initializer value diffs: {report.initializer_value_diffs}")
    print(f"Tensor shape diffs: {len(report.tensor_shape_diffs)}")
    print(f"Seq_len candidate axes: {len(report.seq_len_axes)}")
    for hit in report.seq_len_axes:
        print(
            f"  {hit.tensor}[axis={hit.axis}]: {hit.value_a} vs {hit.value_b} -> {SEQ_LEN_PARAM}"
        )
    if report.axis_pattern_counts:
        print("Axis patterns (1 vs 128):")
        for key, count in report.axis_pattern_counts.items():
            print(f"  {key}: {count} tensor(s)")
    print(f"Node I/O shape diffs: {len(report.node_io_diffs)}")
    for row in report.node_io_diffs[:40]:
        print(f"  {row}")
    if len(report.node_io_diffs) > 40:
        print(f"  ... and {len(report.node_io_diffs) - 40} more")
    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        print(f"JSON report: {json_out}")
