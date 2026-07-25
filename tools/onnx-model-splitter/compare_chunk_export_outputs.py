#!/usr/bin/env python3
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""
Compare prefill / decode ONNX pairs and genai_config_*.json under an ``export_chunk_model.py`` output directory.

For each stem ``TAG`` (from ``prefill_TAG.onnx``), expects:
  - ``decode_TAG.onnx``
  - ``genai_config_TAG.json`` (not ``*_dml.json``)

Checks (prefill vs decode graphs):
  - Same node count; same node names and ``op_type`` per name.
  - Per node: same ordered input names and output names.
  - **Default:** for each edge (same tensor name on both sides), only **element types** must
    match; **shapes are not compared** on intermediates (prefill uses prompt length ``S``,
    decode uses ``1`` - differences like ``[...,512,...]`` vs ``[...,1,...]`` are expected).
  - ``--compare-intermediate-shapes``: also require identical shapes on every edge (verbose).
  - ``GroupQueryAttention``: same wiring for the **total-sequence** input (0-based index **6**
    when present); initializer payloads compared when relevant.
  - Graph-level inputs/outputs: same names and order; **types** must match; **shapes** must match
    except ``input_ids`` / ``inputs_embeds`` / ``position_ids`` (prefill prompt length vs decode
    ``1``) and ``logits`` dim1 (other dims must match).
  - ``past_key_values.*`` / ``present.*``: same shapes between prefill and decode.

Also validates ``genai_config_TAG.json``:
  - ``decoder.pipeline.prefill.filename`` / ``decode.filename`` match ``prefill_TAG.onnx`` /
    ``decode_TAG.onnx``.
  - If ``pipeline.*`` lists ``inputs`` / ``outputs``: compare the **prefix** (all names before the
    first ``past_key_values.*`` or ``present.*``) to ONNX. If order differs but **Counter** matches,
    emit a non-fatal note only (e.g. Gemma ``inputs_embeds`` vs ``attention_mask`` order vs template);
    if the multiset differs, FAIL. The **past/present** tail is compared the same way (order-only
    mismatch is a note).

Usage:
  python compare_chunk_export_outputs.py /path/to/export_chunk_model_output
  python compare_chunk_export_outputs.py /path/to/out --quiet
  python compare_chunk_export_outputs.py /path/to/out --compare-intermediate-shapes
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper

PAST_KV_IO_RE = re.compile(r"^past_key_values\.\d+\.(?:key|value)$")
PRESENT_IO_RE = re.compile(r"^present\.\d+\.(?:key|value)$")

# Prefill vs decode: these graph I/O tensors intentionally differ on the prompt / step axis.
_GRAPH_IO_SEQ_AXIS_EXCEPTION_INPUTS = frozenset(
    {"input_ids", "inputs_embeds", "position_ids"}
)
_GRAPH_IO_SEQ_AXIS_EXCEPTION_OUTPUTS = frozenset({"logits"})


def _graph_io_tensor_compatible(
    name: str,
    ta: str,
    sa: tuple,
    tb: str,
    sb: tuple,
    *,
    kind: str,
) -> tuple[bool, str | None]:
    """Return (ok, reason_if_bad). ``kind`` is ``'in'`` or ``'out'``."""
    if ta != tb:
        return False, f"dtype mismatch: {ta!r} vs {tb!r}"
    if kind == "in" and name in _GRAPH_IO_SEQ_AXIS_EXCEPTION_INPUTS:
        if len(sa) != len(sb):
            return False, f"rank mismatch: {len(sa)} vs {len(sb)}"
        if len(sa) >= 1 and sa[0] != sb[0]:
            return False, f"batch dim shape[0] mismatch: {sa} vs {sb}"
        return True, None
    if kind == "out" and name in _GRAPH_IO_SEQ_AXIS_EXCEPTION_OUTPUTS:
        if len(sa) == 3 and len(sb) == 3:
            if sa[0] != sb[0] or sa[2] != sb[2]:
                return (
                    False,
                    f"logits: only dim1 may differ; got prefill {sa} vs decode {sb}",
                )
            return True, None
        return (
            False,
            f"logits: expected rank-3; got prefill rank {len(sa)} vs decode {len(sb)}",
        )
    if (ta, sa) != (tb, sb):
        return False, f"prefill ({ta}, {sa}) vs decode ({tb}, {sb})"
    return True, None


def _dim_to_tuple(dim: onnx.TensorShapeProto.Dimension) -> tuple[str, int | str]:
    if dim.dim_param:
        return ("param", dim.dim_param)
    return ("value", int(dim.dim_value))


def _tensor_type_str(tt: onnx.TypeProto.Tensor) -> str:
    return TensorProto.DataType.Name(tt.elem_type)


def _tensor_info_from_type_proto(
    tt: onnx.TypeProto.Tensor,
) -> tuple[str, tuple[tuple[str, int | str], ...]]:
    if not tt.HasField("elem_type"):
        return ("UNKNOWN", ())
    et = _tensor_type_str(tt)
    shape = tuple(_dim_to_tuple(d) for d in tt.shape.dim)
    return (et, shape)


def tensor_meta_for_name(
    graph: onnx.GraphProto, name: str
) -> tuple[str, tuple[tuple[str, int | str], ...]] | None:
    """Return (elem_type_name, shape_dims) or None if unresolved."""
    for vi in graph.input:
        if vi.name == name and vi.type.HasField("tensor_type"):
            return _tensor_info_from_type_proto(vi.type.tensor_type)
    for vi in graph.output:
        if vi.name == name and vi.type.HasField("tensor_type"):
            return _tensor_info_from_type_proto(vi.type.tensor_type)
    for vi in graph.value_info:
        if vi.name == name and vi.type.HasField("tensor_type"):
            return _tensor_info_from_type_proto(vi.type.tensor_type)
    for init in graph.initializer:
        if init.name == name:
            et = TensorProto.DataType.Name(init.data_type)
            shape = tuple(("value", int(d)) for d in init.dims)
            return (et, shape)
    return None


def _init_numpy(init: onnx.TensorProto) -> np.ndarray | None:
    try:
        return numpy_helper.to_array(init)
    except Exception:
        return None


def initializer_payload_equal(
    ga: onnx.GraphProto,
    gb: onnx.GraphProto,
    name: str,
) -> tuple[bool, str]:
    """True if both graphs have initializer ``name`` with identical dtype/shape/bytes."""
    a = next((i for i in ga.initializer if i.name == name), None)
    b = next((i for i in gb.initializer if i.name == name), None)
    if a is None and b is None:
        return True, "both absent"
    if a is None or b is None:
        return False, f"only one side has initializer {name!r}"
    if a.data_type != b.data_type or list(a.dims) != list(b.dims):
        return False, f"dtype/dims differ for {name!r}"
    arr_a = _init_numpy(a)
    arr_b = _init_numpy(b)
    if arr_a is None or arr_b is None:
        same = a.raw_data == b.raw_data
        return same, "raw_data ok" if same else "raw_data differ"
    if np.array_equal(arr_a, arr_b):
        return True, "array_equal"
    return False, f"values differ: {arr_a.ravel()[:4]} vs {arr_b.ravel()[:4]} ..."


def gqa_total_input_compare(
    na: onnx.NodeProto,
    nb: onnx.NodeProto,
    ga: onnx.GraphProto,
    gb: onnx.GraphProto,
) -> list[str]:
    """Compare GroupQueryAttention wiring for the **total sequence length** input slot.

    This is **not** a generic GQA port detector: it follows the wiring contract produced by
    ``export_chunk_model`` / chunk export in this repo. There, ``GroupQueryAttention`` exposes the
    total-sequence tensor at **0-based input index 6** when the node has more than six inputs;
    otherwise the same role is taken to be index **5**. If a graph uses a different op signature
    (same input count but different semantics), this check may compare the wrong slot.
    """
    errs: list[str] = []
    # export_chunk_model chunk graphs: total-seq at index 6 when len(inputs) > 6, else 5.
    idx = 6 if len(na.input) > 6 and len(nb.input) > 6 else 5
    if len(na.input) <= idx or len(nb.input) <= idx:
        return [
            f"GQA {na.name!r}: need input index {idx}, got len "
            f"{len(na.input)} vs {len(nb.input)}"
        ]
    ia, ib = na.input[idx], nb.input[idx]
    if ia != ib:
        errs.append(f"GQA {na.name!r} input[{idx}] name: {ia!r} vs {ib!r}")
        init_a = next((x for x in ga.initializer if x.name == ia), None)
        init_b = next((x for x in gb.initializer if x.name == ib), None)
        if init_a is not None and init_b is not None:
            va, vb = _init_numpy(init_a), _init_numpy(init_b)
            if va is not None and vb is not None and np.array_equal(va, vb):
                errs[-1] += " (initializer values equal)"
            elif va is not None and vb is not None:
                errs.append(f"  initializer values differ for input[{idx}]")
        return errs
    if ia:
        init_a = next((x for x in ga.initializer if x.name == ia), None)
        init_b = next((x for x in gb.initializer if x.name == ib), None)
        if init_a is not None and init_b is not None:
            eq, detail = initializer_payload_equal(ga, gb, ia)
            if not eq:
                errs.append(f"GQA {na.name!r} input[{idx}]={ia!r}: {detail}")
    return errs


ISSUE_SECTION_ORDER = (
    "load",
    "graph_io",
    "node_struct",
    "edge_dtype",
    "edge_shape",
    "gqa",
    "kv_cache",
    "genai",
    "misc",
)
ISSUE_SECTION_TITLE = {
    "load": "load / read",
    "graph_io": "graph I/O",
    "node_struct": "node structure (name / op_type / wiring)",
    "edge_dtype": "intermediate edge dtypes",
    "edge_shape": "intermediate edge shapes (--compare-intermediate-shapes only)",
    "gqa": "GroupQueryAttention total",
    "kv_cache": "past_key_values / present",
    "genai": "genai_config.json",
    "misc": "other",
}


@dataclass
class TripletReport:
    stem: str
    ok: bool = True
    sections: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))

    def add(self, section: str, msg: str, *, fatal: bool = True) -> None:
        if section not in ISSUE_SECTION_ORDER:
            section = "misc"
        self.sections[section].append(msg)
        if fatal:
            self.ok = False


def _nodes_by_name(graph: onnx.GraphProto) -> dict[str, onnx.NodeProto]:
    out: dict[str, onnx.NodeProto] = {}
    for i, n in enumerate(graph.node):
        key = (n.name or "").strip()
        if not key:
            key = f"__unnamed_op{i}_{n.op_type}"
        elif key in out:
            key = f"{key}__idx{i}"
        out[key] = n
    return out


def compare_graphs(
    prefill: onnx.ModelProto,
    decode: onnx.ModelProto,
    rep: TripletReport,
    *,
    compare_intermediate_shapes: bool,
) -> None:
    ga, gb = prefill.graph, decode.graph

    # Graph I/O
    def io_list(g: onnx.GraphProto, kind: str) -> list[tuple[str, str, tuple]]:
        seq = g.input if kind == "in" else g.output
        out = []
        for vi in seq:
            if not vi.type.HasField("tensor_type"):
                out.append((vi.name, "NON_TENSOR", ()))
                continue
            et, sh = _tensor_info_from_type_proto(vi.type.tensor_type)
            out.append((vi.name, et, sh))
        return out

    ins_a, ins_b = io_list(ga, "in"), io_list(gb, "in")
    outs_a, outs_b = io_list(ga, "out"), io_list(gb, "out")
    if [x[0] for x in ins_a] != [x[0] for x in ins_b]:
        rep.add(
            "graph_io",
            "graph input name order differs\n"
            f"  prefill: {[x[0] for x in ins_a]}\n"
            f"  decode:  {[x[0] for x in ins_b]}",
        )
    for (na, ta, sa), (nb, tb, sb) in zip(ins_a, ins_b):
        if na != nb:
            continue
        ok_io, why = _graph_io_tensor_compatible(na, ta, sa, tb, sb, kind="in")
        if not ok_io:
            rep.add("graph_io", f"graph input {na!r}: {why}")

    if [x[0] for x in outs_a] != [x[0] for x in outs_b]:
        rep.add(
            "graph_io",
            "graph output name order differs\n"
            f"  prefill: {[x[0] for x in outs_a]}\n"
            f"  decode:  {[x[0] for x in outs_b]}",
        )
    for (na, ta, sa), (nb, tb, sb) in zip(outs_a, outs_b):
        if na != nb:
            continue
        ok_io, why = _graph_io_tensor_compatible(na, ta, sa, tb, sb, kind="out")
        if not ok_io:
            rep.add("graph_io", f"graph output {na!r}: {why}")

    if len(ga.node) != len(gb.node):
        rep.add(
            "node_struct",
            f"node count mismatch: prefill {len(ga.node)} vs decode {len(gb.node)}",
        )

    ma, mb = _nodes_by_name(ga), _nodes_by_name(gb)
    names_a, names_b = set(ma), set(mb)
    only_a, only_b = sorted(names_a - names_b), sorted(names_b - names_a)
    if only_a:
        rep.add(
            "node_struct",
            f"nodes only in prefill ({len(only_a)}): "
            f"{only_a[:20]}{'...' if len(only_a) > 20 else ''}",
        )
    if only_b:
        rep.add(
            "node_struct",
            f"nodes only in decode ({len(only_b)}): "
            f"{only_b[:20]}{'...' if len(only_b) > 20 else ''}",
        )

    for name in sorted(names_a & names_b):
        na, nb = ma[name], mb[name]
        if na.op_type != nb.op_type:
            rep.add(
                "node_struct",
                f"node {name!r}: op_type {na.op_type!r} vs {nb.op_type!r}",
            )
            continue
        if na.op_type == "GroupQueryAttention":
            for msg in gqa_total_input_compare(na, nb, ga, gb):
                rep.add("gqa", msg)
        if list(na.input) != list(nb.input):
            rep.add(
                "node_struct",
                f"node {name!r}: input list differs\n"
                f"  prefill: {list(na.input)}\n"
                f"  decode:  {list(nb.input)}",
            )
        else:
            for ia, ib in zip(na.input, nb.input):
                if not ia and not ib:
                    continue
                if ia != ib or not ia:
                    continue
                meta_a = tensor_meta_for_name(ga, ia)
                meta_b = tensor_meta_for_name(gb, ib)
                if compare_intermediate_shapes:
                    if meta_a != meta_b:
                        rep.add(
                            "edge_shape",
                            f"node {name!r} input {ia!r}: prefill {meta_a} vs decode {meta_b}",
                        )
                else:
                    et_a = meta_a[0] if meta_a else None
                    et_b = meta_b[0] if meta_b else None
                    if et_a != et_b:
                        rep.add(
                            "edge_dtype",
                            f"node {name!r} input {ia!r}: dtype prefill {et_a} vs decode {et_b}",
                        )
                    elif meta_a is None or meta_b is None:
                        rep.add(
                            "misc",
                            f"node {name!r} input {ia!r}: missing type info on one side "
                            f"(prefill meta={meta_a}, decode meta={meta_b})",
                            fatal=False,
                        )
        if list(na.output) != list(nb.output):
            rep.add(
                "node_struct",
                f"node {name!r}: output list differs\n"
                f"  prefill: {list(na.output)}\n"
                f"  decode:  {list(nb.output)}",
            )
        else:
            for oa, ob in zip(na.output, nb.output):
                if not oa and not ob:
                    continue
                if oa != ob or not oa:
                    continue
                meta_a = tensor_meta_for_name(ga, oa)
                meta_b = tensor_meta_for_name(gb, ob)
                if compare_intermediate_shapes:
                    if meta_a != meta_b:
                        rep.add(
                            "edge_shape",
                            f"node {name!r} output {oa!r}: prefill {meta_a} vs decode {meta_b}",
                        )
                else:
                    et_a = meta_a[0] if meta_a else None
                    et_b = meta_b[0] if meta_b else None
                    if et_a != et_b:
                        rep.add(
                            "edge_dtype",
                            f"node {name!r} output {oa!r}: dtype prefill {et_a} vs decode {et_b}",
                        )
                    elif meta_a is None or meta_b is None:
                        rep.add(
                            "misc",
                            f"node {name!r} output {oa!r}: missing type info on one side "
                            f"(prefill meta={meta_a}, decode meta={meta_b})",
                            fatal=False,
                        )

    # past / present shapes by name
    past_names = sorted(
        {vi.name for vi in ga.input if PAST_KV_IO_RE.match(vi.name)}
        & {vi.name for vi in gb.input if PAST_KV_IO_RE.match(vi.name)}
    )
    for nm in past_names:
        ma_ = tensor_meta_for_name(ga, nm)
        mb_ = tensor_meta_for_name(gb, nm)
        if ma_ != mb_:
            rep.add(
                "kv_cache",
                f"past {nm!r}: prefill {ma_} vs decode {mb_}",
            )

    pres_names = sorted(
        {vi.name for vi in ga.output if PRESENT_IO_RE.match(vi.name)}
        & {vi.name for vi in gb.output if PRESENT_IO_RE.match(vi.name)}
    )
    for nm in pres_names:
        ma_ = tensor_meta_for_name(ga, nm)
        mb_ = tensor_meta_for_name(gb, nm)
        if ma_ != mb_:
            rep.add(
                "kv_cache",
                f"present {nm!r}: prefill {ma_} vs decode {mb_}",
            )


def _describe_io_list_mismatch(
    json_list: list[str],
    onnx_list: list[str],
) -> str:
    """Human-readable diff for pipeline inputs/outputs vs graph I/O names."""
    lj, lo = len(json_list), len(onnx_list)
    lines: list[str] = [f"count json={lj} onnx={lo}"]
    m = min(lj, lo)
    if m == 0:
        lines.append(f"json full: {json_list[:32]!r}")
        lines.append(f"onnx full: {onnx_list[:32]!r}")
        return "\n".join(lines)

    mismatch_at: int | None = None
    for i in range(m):
        if json_list[i] != onnx_list[i]:
            mismatch_at = i
            break

    if mismatch_at is None:
        if lj == lo:
            return "\n".join(
                lines + ["(internal: lists should differ but are identical per index)"]
            )
        if json_list[:m] == onnx_list[:m]:
            if lj > lo:
                tail = json_list[m : m + 16]
                lines.append(
                    f"first {m} names match; json has {lj - lo} extra entries "
                    f"(often template layer count > ONNX); extra head: {tail!r}"
                )
            else:
                tail = onnx_list[m : m + 16]
                lines.append(
                    f"first {m} names match; onnx has {lo - lj} extra entries; extra head: {tail!r}"
                )
        return "\n".join(lines)

    i = mismatch_at
    lines.append(
        f"first mismatch index i={i}: json={json_list[i]!r} onnx={onnx_list[i]!r}"
    )
    lo_c = max(0, i - 2)
    hi_c = min(m, i + 5)
    lines.append(f"  json slice [{lo_c}:{hi_c}]: {json_list[lo_c:hi_c]!r}")
    lines.append(f"  onnx slice [{lo_c}:{hi_c}]: {onnx_list[lo_c:hi_c]!r}")

    set_j, set_o = set(json_list), set(onnx_list)
    only_j = sorted(set_j - set_o)
    only_o = sorted(set_o - set_j)
    if only_j or only_o:
        lines.append(
            "  set diff (ignore order/multiplicity): "
            f"only in json {len(only_j)}, only in onnx {len(only_o)}"
        )
        if only_j[:12]:
            lines.append(
                f"    only json: {only_j[:12]!r}{'...' if len(only_j) > 12 else ''}"
            )
        if only_o[:12]:
            lines.append(
                f"    only onnx: {only_o[:12]!r}{'...' if len(only_o) > 12 else ''}"
            )

    return "\n".join(lines)


def _split_io_prefix_and_kv(
    names: list[str], *, is_input: bool
) -> tuple[list[str], list[str]]:
    """Split into (prefix before KV tensors, past_key_values.* or present.* tail)."""
    if is_input:
        for i, n in enumerate(names):
            if n.startswith("past_key_values."):
                return names[:i], names[i:]
        return names, []
    for i, n in enumerate(names):
        if n.startswith("present."):
            return names[:i], names[i:]
    return names, []


def validate_genai_json(
    cfg: dict[str, Any],
    stem: str,
    rep: TripletReport,
    *,
    prefill_graph: onnx.GraphProto,
    decode_graph: onnx.GraphProto,
) -> None:
    dec = cfg.get("model", {}).get("decoder", {})
    pipe = dec.get("pipeline") or {}
    pre = pipe.get("prefill") or {}
    decp = pipe.get("decode") or {}
    exp_pre, exp_dec = f"prefill_{stem}.onnx", f"decode_{stem}.onnx"
    if pre.get("filename") != exp_pre:
        rep.add(
            "genai",
            f"prefill.filename: expected {exp_pre!r}, got {pre.get('filename')!r}",
        )
    if decp.get("filename") != exp_dec:
        rep.add(
            "genai",
            f"decode.filename: expected {exp_dec!r}, got {decp.get('filename')!r}",
        )

    def check_io_lists(
        side: str, graph: onnx.GraphProto, block: dict[str, Any], kind: str
    ) -> None:
        key = "inputs" if kind == "in" else "outputs"
        if key not in block or not isinstance(block[key], list):
            return
        listed = list(block[key])
        actual = [x.name for x in (graph.input if kind == "in" else graph.output)]
        is_in = kind == "in"
        pre_j, kv_j = _split_io_prefix_and_kv(listed, is_input=is_in)
        pre_o, kv_o = _split_io_prefix_and_kv(actual, is_input=is_in)
        if pre_j != pre_o:
            if Counter(pre_j) == Counter(pre_o) and len(pre_j) == len(pre_o):
                rep.add(
                    "genai",
                    f"pipeline.{side}.{key}: prefix (before first past/present) **multiset** matches ONNX; "
                    f"**order** only differs (e.g. ``inputs_embeds`` vs ``attention_mask``; safe if binding "
                    f"by name). json: {pre_j!r} | onnx: {pre_o!r}",
                    fatal=False,
                )
            else:
                detail = _describe_io_list_mismatch(pre_j, pre_o)
                rep.add(
                    "genai",
                    f"pipeline.{side}.{key} prefix vs ONNX: name multiset or length mismatch.\n{detail}\n"
                    f"  hint: often ``input_ids`` / ``inputs_embeds`` / ``attention_mask`` / "
                    f"``position_ids`` vs template.",
                )
                return
        if Counter(kv_j) != Counter(kv_o):
            detail = _describe_io_list_mismatch(kv_j, kv_o)
            rep.add(
                "genai",
                f"pipeline.{side}.{key}: past_key_values.* / present.* name multiset vs ONNX mismatch.\n"
                f"{detail}\n"
                f"  hint: count/set diff often means template num_hidden_layers vs model layer count.",
            )
            return
        if kv_j != kv_o:
            rep.add(
                "genai",
                f"pipeline.{side}.{key}: past/present multiset matches ONNX; **list order** only differs "
                f"(runtime usually binds by name). json first 8: {kv_j[:8]!r} | onnx first 8: {kv_o[:8]!r}",
                fatal=False,
            )

    check_io_lists("prefill", prefill_graph, pre, "in")
    check_io_lists("prefill", prefill_graph, pre, "out")
    check_io_lists("decode", decode_graph, decp, "in")
    check_io_lists("decode", decode_graph, decp, "out")


def discover_triplets(root: str) -> list[tuple[str, str, str, str]]:
    """Return list of (stem, prefill_path, decode_path, genai_json_path)."""
    root = os.path.abspath(root)
    out: list[tuple[str, str, str, str]] = []
    if not os.path.isdir(root):
        return out
    for fn in sorted(os.listdir(root)):
        if not fn.startswith("prefill_") or not fn.endswith(".onnx"):
            continue
        stem = fn[len("prefill_") : -len(".onnx")]
        p_pre = os.path.join(root, fn)
        p_dec = os.path.join(root, f"decode_{stem}.onnx")
        p_js = os.path.join(root, f"genai_config_{stem}.json")
        if stem.endswith("_dml"):
            continue
        if os.path.isfile(p_dec) and os.path.isfile(p_js):
            out.append((stem, p_pre, p_dec, p_js))
    return out


def print_triplet_report(rep: TripletReport, *, per_section_cap: int) -> None:
    status = "OK" if rep.ok else "FAIL"
    print(f"[{status}] stem={rep.stem!r}")
    total = sum(len(rep.sections[s]) for s in ISSUE_SECTION_ORDER)
    if total == 0:
        print("    (no issues)")
        return
    for sec in ISSUE_SECTION_ORDER:
        lines = rep.sections.get(sec, [])
        if not lines:
            continue
        title = ISSUE_SECTION_TITLE.get(sec, sec)
        print(f"  --- {title} ({len(lines)}) ---")
        shown = 0
        for line in lines:
            if shown >= per_section_cap:
                rest = len(lines) - shown
                if rest > 0:
                    print(f"    ... {rest} more (raise --per-section-cap to show more)")
                break
            if len(line) > 220:
                line = line[:217] + "..."
            print(f"    - {line}")
            shown += 1


def run_one(
    stem: str,
    p_pre: str,
    p_dec: str,
    p_js: str,
    *,
    quiet: bool,
    compare_intermediate_shapes: bool,
    per_section_cap: int,
) -> TripletReport:
    rep = TripletReport(stem=stem)
    try:
        m_pre = onnx.load(p_pre, load_external_data=False)
        m_dec = onnx.load(p_dec, load_external_data=False)
    except Exception as e:
        rep.add("load", f"failed to load ONNX: {e}")
        return rep

    compare_graphs(
        m_pre,
        m_dec,
        rep,
        compare_intermediate_shapes=compare_intermediate_shapes,
    )

    try:
        with open(p_js, encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception as e:
        rep.add("load", f"failed to read genai JSON: {e}")
        return rep

    validate_genai_json(
        cfg,
        stem,
        rep,
        prefill_graph=m_pre.graph,
        decode_graph=m_dec.graph,
    )

    if not quiet:
        print_triplet_report(rep, per_section_cap=per_section_cap)
    return rep


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Compare prefill/decode ONNX and genai_config under export_chunk_model output."
    )
    ap.add_argument(
        "output_dir",
        help="Directory written by export_chunk_model.py (-o / --output).",
    )
    ap.add_argument("--quiet", action="store_true", help="Only print summary line.")
    ap.add_argument(
        "--compare-intermediate-shapes",
        action="store_true",
        help="Also compare full shapes on every intermediate edge (default: dtype only; "
        "prefill/decode seq dims often differ).",
    )
    ap.add_argument(
        "--per-section-cap",
        type=int,
        default=25,
        metavar="N",
        help="Max detail lines per issue section (default: 25).",
    )
    args = ap.parse_args(argv)

    root = os.path.abspath(args.output_dir)
    if not os.path.isdir(root):
        print(f"Not a directory: {root}", file=sys.stderr)
        return 2

    triplets = discover_triplets(root)
    if not triplets:
        print(
            f"No complete triplets (prefill_*.onnx + decode_*.onnx + genai_config_*.json) in {root}"
        )
        return 1

    covered_prefill = {f"prefill_{s}.onnx" for s, _, _, _ in triplets}
    all_prefill = [
        f for f in os.listdir(root) if f.startswith("prefill_") and f.endswith(".onnx")
    ]
    orphan_prefill = sorted(set(all_prefill) - covered_prefill)
    if orphan_prefill and not args.quiet:
        print(
            "Note: prefill ONNX without matching decode+genai_config (skipped):",
            ", ".join(orphan_prefill[:20])
            + (" ..." if len(orphan_prefill) > 20 else ""),
            file=sys.stderr,
        )

    reports: list[TripletReport] = []
    for stem, p_pre, p_dec, p_js in triplets:
        reports.append(
            run_one(
                stem,
                p_pre,
                p_dec,
                p_js,
                quiet=args.quiet,
                compare_intermediate_shapes=args.compare_intermediate_shapes,
                per_section_cap=max(1, args.per_section_cap),
            )
        )

    n_ok = sum(1 for r in reports if r.ok)
    n_fail = len(reports) - n_ok
    failed_stems = [r.stem for r in reports if not r.ok]
    fail_hint = f" failed: {', '.join(failed_stems[:16])}" + (
        " ..." if len(failed_stems) > 16 else ""
    )
    print(
        f"Summary: {n_ok}/{len(reports)} triplets passed, {n_fail} failed (dir={root})"
        + (fail_hint if failed_stems else "")
    )
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
