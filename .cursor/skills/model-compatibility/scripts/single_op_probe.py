#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Probe operator support without a whole-graph dump (evidence level C).

When the EP cannot import a model at all, there is no `ep_input.mlir` to
slice and the report would otherwise fall back to reading documentation --
which cannot tell an implementation that exists from one that accepts this
model. This builds a one-node ONNX model per operator instead, dumps each
through the same MorphiZen import path, and hands the result to the same
probe the whole-graph path uses.

The construction rules mirror the MLIR slicer's, for the same reasons:

- Weights and values captured from an enclosing scope become graph inputs.
  Their contents cannot change a converter's decision, and copying them is
  not an option when a single embedding table runs to a gigabyte.
- Small inline Constants are carried in, because converters read them.

Results are emitted in the same shape as `probe.py`'s, so everything
downstream is unchanged.

Usage:
  python single_op_probe.py <model.onnx> <out_dir> --package <gpu-test-package>
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import onnx
from onnx import (
    ModelProto,
    NodeProto,
    ValueInfoProto,
    helper,
    shape_inference,
)

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import probe  # noqa: E402
from onnx_graph_walk import NodeContext, iter_typed_nodes  # noqa: E402

# A Constant small enough that its value is more useful than its size. Above
# this it is a weight, and weights become inputs.
INLINE_CONSTANT_LIMIT = 1024

DUMP_TIMEOUT_SEC = 300


def _is_typed(v: ValueInfoProto) -> bool:
    """Does this value carry enough type for a model signature?

    An element type is enough. A missing shape means unknown rank, which
    `onnx.checker` rejects in a signature but the importer accepts, turning
    it into an unranked MLIR tensor the probe already knows how to handle.
    """
    t = v.type
    if t.HasField("tensor_type"):
        return bool(t.tensor_type.elem_type)
    # Sequence and optional types have no shape of their own but are still
    # expressible; let the importer be the one to reject them.
    return t.HasField("sequence_type") or t.HasField("optional_type")


def _subgraph_captures(node: NodeProto) -> list[str]:
    """Values a node's subgraphs read from the scope around them.

    A branch or loop body is a graph that can reference names it does not
    define. Lifting the node out leaves those references dangling, so they
    have to be rebuilt as inputs of the standalone model.
    """
    found: list[str] = []

    def walk(graph) -> None:
        defined = {v.name for v in graph.input}
        defined |= {i.name for i in graph.initializer}
        defined |= {o for n in graph.node for o in n.output if o}
        for n in graph.node:
            for name in n.input:
                if name and name not in defined:
                    found.append(name)
            for attr in n.attribute:
                if attr.g.ByteSize():
                    walk(attr.g)
                for sub in attr.graphs:
                    walk(sub)

    for attr in node.attribute:
        if attr.g.ByteSize():
            walk(attr.g)
        for sub in attr.graphs:
            walk(sub)
    return found


def _constant_size(node: NodeProto) -> int:
    for attr in node.attribute:
        if attr.name == "value" and attr.t.ByteSize():
            return len(attr.t.raw_data) or attr.t.ByteSize()
    return 0


def build_single_op_model(
    ctx: NodeContext, opset: list, ir_version: int
) -> tuple[ModelProto | None, str]:
    """Wrap one node in a model of its own.

    Returns (model, reason); the model is None when the node cannot be
    isolated, and the reason is a finding rather than a bare failure.
    """
    node = ctx.node
    body: list[NodeProto] = []
    inputs: list[ValueInfoProto] = []
    seen: set[str] = set()

    for name in list(node.input) + _subgraph_captures(node):
        if not name:
            continue
        const = ctx.constants.get(name)
        if const is not None and _constant_size(const) <= INLINE_CONSTANT_LIMIT:
            if name not in seen:
                body.append(const)
                seen.add(name)
            continue
        vi = ctx.types.get(name)
        if vi is None or not _is_typed(vi):
            return None, f"no type for operand {name}"
        if name not in seen:
            inputs.append(vi)
            seen.add(name)

    outputs = []
    for name in node.output:
        if not name:
            continue
        vi = ctx.types.get(name)
        if vi is None or not _is_typed(vi):
            return None, f"no type for result {name}"
        outputs.append(vi)
    if not outputs:
        return None, "node produces no typed result"

    body.append(node)
    graph = helper.make_graph(body, f"probe_{node.op_type}", inputs, outputs)
    model = helper.make_model(graph, opset_imports=opset)
    model.ir_version = ir_version
    # The checker advises, it does not decide. It rejects a signature of
    # unknown rank that the importer accepts and turns into an unranked
    # tensor, which the probe handles. Keep its complaint to explain an
    # import failure, should one follow.
    try:
        onnx.checker.check_model(model)
    except Exception as exc:  # noqa: BLE001 - the message is the finding
        return model, str(exc).splitlines()[0][:160]
    return model, ""


def dump_model(
    runner: Path, config: Path, model_path: Path, out_dir: Path
) -> tuple[Path | None, str]:
    """Import one model through the EP and return the MLIR it produced."""
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        proc = subprocess.run(
            [
                str(runner),
                "-m",
                str(model_path),
                "--no-run",
                "--allow-cpu-fallback",
                "--provider-options",
                f"config_file={config}",
                "--provider-options",
                f"pass.init.directory={out_dir.as_posix()}",
            ],
            capture_output=True,
            text=True,
            timeout=DUMP_TIMEOUT_SEC,
        )
    except subprocess.TimeoutExpired:
        return None, f"import timed out after {DUMP_TIMEOUT_SEC}s"

    mlir = out_dir / "ep_input.mlir"
    if mlir.exists():
        return mlir, ""

    log = (proc.stderr or "") + (proc.stdout or "")
    for line in log.splitlines():
        if "Error in ORT API" in line or "error:" in line:
            # Trim glog's timestamp and source-location preamble.
            _, _, tail = line.partition("] ")
            return None, (tail or line).strip()[:200]
    return None, f"importer exited with code {proc.returncode}"


def _find_op_line(mlir_lines: list[str], op_type: str) -> int | None:
    """Where the operator of interest sits in its own dump."""
    for i, line in enumerate(mlir_lines):
        if f'"onnx.{op_type}"' in line or f'function_name = "{op_type}"' in line:
            return i + 1
    return None


def probe_operator(
    opt: Path, mlir: Path, op_type: str, work: Path
) -> tuple[dict, dict]:
    """Run the whole-graph probe's two stages against one imported operator.

    Returns (stage1_entry, stage2_entry) in `probe.py`'s own shape, so the
    assembler cannot tell which path produced them.
    """
    src = mlir.read_text(encoding="utf-8").splitlines()
    def_index = probe.build_def_index(src)
    line_no = _find_op_line(src, op_type)
    if line_no is None:
        reason = "operator absent from its own import"
        return (
            {"inconclusive": 1, "reason": reason},
            {"status": "not_sliceable", "reason": reason},
        )

    # Same slicer as the whole-graph path, candidate shapes included: an
    # import of a subgraph operator can come back unranked just as a slice
    # of one can.
    candidates, line_no, why, inferred = probe._slices_for(
        src, def_index, [{"line_no": line_no}]
    )
    if not candidates:
        return (
            {"inconclusive": 1, "reason": why},
            {"status": "not_sliceable", "reason": why},
        )

    work.mkdir(parents=True, exist_ok=True)
    out = work / f"{op_type}.hip.mlir"
    converted = False
    for module in candidates:
        converted = probe._converts_cleanly(opt, module, work / f"{op_type}.mlir", out)
        if converted:
            break
    targets: dict[str, int] = {}
    if converted and out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            for name in probe.ops_on_line(line):
                if not name.startswith("onnx.") and name not in probe.DPS_HELPER_OPS:
                    targets[name] = targets.get(name, 0) + 1

    s1 = {"targets": targets, "unconverted_lines": [] if converted else [line_no]}
    if not converted:
        if inferred:
            why = "unranked types; a slice with an inferred shape did not convert"
            return {"inconclusive": 1, "reason": why}, {
                "status": "not_sliceable",
                "reason": why,
            }
        return s1, {"status": "skipped", "reason": "unconverted in stage1"}

    s2_path = work / f"{op_type}.s2.mlir"
    llvm = work / f"{op_type}.llvm.mlir"
    for module in candidates:
        s2_path.write_text(module, encoding="utf-8")
        code, log = probe.run_opt(opt, s2_path, probe.STAGE2_PASSES, llvm, debug=False)
        if code == 0 and llvm.exists():
            break
    if code != 0 or not llvm.exists():
        return s1, {
            # A guessed shape is as likely a suspect as the lowering.
            "status": "not_sliceable" if inferred else "lowering_broken",
            "source_line": line_no,
            "error": probe._failure_reason(code, log)[:300],
            **(
                {
                    "reason": "unranked types; a slice with an inferred shape did not lower"
                }
                if inferred
                else {}
            ),
        }

    symbols = sorted(
        {
            s
            for s in probe._WRAP_SYM.findall(llvm.read_text(encoding="utf-8"))
            if not probe.INFRA_SYMBOLS.match(s)
        }
    )
    return s1, {
        "status": "ok",
        "source_line": line_no,
        "runtime_funcs": symbols,
        "compile_time": not symbols,
        "attributes": [],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--package", type=Path, required=True)
    ap.add_argument(
        "--config",
        type=Path,
        default=_HERE / "morphizen_init_config.json",
        help="MorphiZen plugin config used for the per-operator imports.",
    )
    args = ap.parse_args()

    runner = args.package / "bin" / "hip-onnx-runner.exe"
    if not runner.exists():
        runner = args.package / "bin" / "hip-onnx-runner"
    opt = args.package / "bin" / "hip-mlir-opt.exe"
    if not opt.exists():
        opt = args.package / "bin" / "hip-mlir-opt"
    for tool in (runner, opt):
        if not tool.exists():
            sys.exit(f"not found in the package: {tool}")

    model = onnx.load(args.model)
    try:
        inferred = shape_inference.infer_shapes(model, strict_mode=False)
    except Exception:
        # Better to probe what is already typed than to probe nothing.
        inferred = model
    opset = list(model.opset_import)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = args.out_dir / "models"
    dumps_dir = args.out_dir / "dumps"
    work_dir = args.out_dir / "slices"

    stage1: dict[str, dict] = {}
    stage2: dict[str, dict] = {}
    counts: dict[str, int] = {}
    domains: dict[str, str] = {}

    # One representative per operator type: the probe answers per type, and
    # a second instance of the same operator costs an import to say the same
    # thing. Instances are counted separately, from the whole model.
    chosen: dict[str, NodeContext] = {}
    for ctx in iter_typed_nodes(inferred.graph):
        op = ctx.node.op_type
        if op == "Constant":
            continue
        counts[op] = counts.get(op, 0) + 1
        domains.setdefault(op, ctx.node.domain or "onnx")
        chosen.setdefault(op, ctx)

    for op, ctx in chosen.items():
        base = {
            "domain": domains[op],
            "count": counts[op],
            "targets": {},
            "helpers": {},
            "folded": 0,
            "dropped_attrs": [],
            "unconverted_lines": [],
            "converted": 0,
            "unconverted": 0,
        }
        print(f"  single-op {op} ...", flush=True)

        built, complaint = build_single_op_model(ctx, opset, model.ir_version)
        if built is None:
            stage1[op] = {**base, "inconclusive": counts[op], "reason": complaint}
            stage2[op] = {"status": "not_sliceable", "reason": complaint}
            continue

        models_dir.mkdir(parents=True, exist_ok=True)
        model_path = models_dir / f"{op}.onnx"
        onnx.save(built, model_path)

        mlir, err = dump_model(runner, args.config, model_path, dumps_dir / op)
        if mlir is not None:
            complaint = ""
        if mlir is None:
            if complaint:
                err = f"{err} (the model we built is also irregular: {complaint})"
            # The EP rejected the operator before any conversion ran. That is
            # a front-end limit, not a missing HIP operator, and the two call
            # for work in different places.
            stage1[op] = {
                **base,
                "unconverted": counts[op],
                "import_failed": True,
                "reason": err,
            }
            stage2[op] = {"status": "import_failed", "error": err}
            continue

        s1, s2 = probe_operator(opt, mlir, op, work_dir)
        merged = {**base, **s1}
        if s2.get("status") == "skipped":
            merged["unconverted"] = counts[op]
        elif "inconclusive" in s1:
            merged["inconclusive"] = counts[op]
        else:
            merged["converted"] = counts[op]
        stage1[op] = merged
        stage2[op] = s2

    result = {
        "stage1": {
            "meta": {
                "mode": "single-op",
                "source": str(args.model),
                "operators_probed": len(chosen),
            },
            "operators": stage1,
        },
        "stage2": {"operators": stage2},
    }
    out = args.out_dir / "probe_result.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    imported = sum(1 for v in stage2.values() if v.get("status") != "import_failed")
    print(f"[OK] {out}  ({imported}/{len(chosen)} operator types imported)")


if __name__ == "__main__":
    main()
