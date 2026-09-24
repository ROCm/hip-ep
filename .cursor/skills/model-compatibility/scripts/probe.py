#!/usr/bin/env python3
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Decide operator support by compiling, not by reading source.

Two stages, because the compiler front end and back end fail in different
ways and the distinction drives different work:

  stage1  whole graph, through convert-onnx-to-hip
          Does a converter exist at all? Operators still named onnx.* after
          the pass had no converter (or their matcher bailed out). An
          unconverted operator does not fail the pass -- applyPatternsGreedily
          simply leaves it alone -- so one probe covers the whole model.

  stage2  one operator at a time, through hip-to-llvm-pipeline
          Is the lowering chain complete, and which runtime symbol does it
          reach? Whole-graph lowering is not an option: constant
          externalization needs an injected FileSystem that hip-mlir-opt does
          not have.

Support is read from the IR diff rather than from the
"[convert-onnx-to-hip] N unconverted ..." line, so a change to that
diagnostic's wording cannot silently break the verdict. The line is still
captured as a cross-check.

Usage:
  python probe.py <ep_input.mlir> <ep_input_ops.json> <out_dir> --package <pkg>
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

from mlir_slice import (
    build_def_index,
    build_single_op_module,
    element_type,
    parse_op_line,
    retype,
    slices_for,
)

# --onnx-to-hip-pipeline's first six passes. The rest of that pipeline
# bufferizes, and would fail on the very operators this probe reports on.
# --mlir-print-debuginfo makes each produced operation carry its source line,
# which is how results pair back to inputs.
STAGE1_PASSES = [
    "--simplify-onnx",
    "--hip-add-context-arg",
    "--onnx-loop-outline",
    "--onnx-if-outline",
    "--hip-infer-loop-body-shapes",
    "--convert-onnx-to-hip",
    "--mlir-print-debuginfo",
]

# Usually DPS init construction and shape math rather than an operator's own
# lowering target, so they are reported separately -- but not always: Shape
# lowers to tensor.dim + tensor.from_elements. An operator with nothing else
# falls back to reporting these.
DPS_HELPER_OPS = frozenset(
    {
        "tensor.dim",
        "tensor.empty",
        "tensor.from_elements",
        "arith.index_cast",
        "arith.constant",
    }
)

STAGE2_PASSES = ["--onnx-to-hip-pipeline", "--hip-to-llvm-pipeline"]

# Runtime entry points every lowered function calls regardless of which
# operator it holds; not evidence of what this operator maps to.
INFRA_SYMBOLS = re.compile(
    r"^wrap_hip(Malloc|Free|Memcpy\w*|StreamSynchronize|Stream\w*|Device\w*)$"
)

_LOC_DEF = re.compile(r'^#(loc\d+)\s*=\s*loc\(".*":(\d+):\d+\)')
_LOC_USE = re.compile(r"loc\(#(loc\d+)\)")
_QUOTED_OP = re.compile(r'"([a-z][a-z0-9_]*\.[A-Za-z0-9_]+)"\s*\(')
_BARE_OP = re.compile(r"=\s*([a-z][a-z0-9_]*\.[a-z0-9_.]+)")
_UNCONVERTED_LINE = re.compile(r"^\s{2}(onnx\.\S+)\s+x(\d+)\s*$")
_WRAP_SYM = re.compile(r"\bwrap_\w+")


def _tool(package_root: Path, name: str) -> Path:
    exe = package_root / "bin" / name
    if not exe.exists():
        raise SystemExit(f"{name} not found under {package_root / 'bin'}")
    return exe


# A run near this bound is not slow, it is stuck: a perturbed attribute can
# leave the greedy rewrite without a fixed point. Real runs are two orders of
# magnitude below it.
OPT_TIMEOUT_SEC = 5

TIMEOUT_EXIT = -9


def run_opt(
    opt: Path,
    src: Path,
    passes: list[str],
    out: Path,
    debug: bool = True,
    timeout: int = OPT_TIMEOUT_SEC,
) -> tuple[int, str]:
    """Run hip-mlir-opt; return (exit code, combined output).

    A timeout reports TIMEOUT_EXIT rather than raising: one operator that
    will not converge should not abort the whole probe.
    """
    env = dict(os.environ)
    env["HIPDNN_EP_DEBUG"] = "1" if debug else "0"
    try:
        proc = subprocess.run(
            [str(opt), str(src), *passes, "-o", str(out)],
            capture_output=True,
            text=True,
            env=env,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return TIMEOUT_EXIT, f"timed out after {timeout}s"
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def _failure_reason(code: int, log: str) -> str:
    """Say why a run failed, in whatever form the failure took.

    A diagnostic is only one of the ways hip-mlir-opt can fail: it also
    crashes, and then there is no `error:` line to find. Reporting "unknown"
    for a segfault hides the most actionable failure of the three.
    """
    if code == TIMEOUT_EXIT:
        return log.strip()
    for line in log.splitlines():
        if "error:" in line:
            return line.strip()
    crash = re.search(r"exception 0x[0-9a-fA-F]+", log)
    if crash:
        top = re.search(r"^#0\s+0x\S+ in\s+(.+)$", log, re.MULTILINE)
        where = (
            f" at {top.group(1).strip()}" if top and "??" not in top.group(1) else ""
        )
        return f"hip-mlir-opt crashed ({crash.group(0)}){where}"
    if code and code & 0xC0000000 == 0xC0000000:
        return f"hip-mlir-opt terminated abnormally (0x{code & 0xFFFFFFFF:08X})"
    return f"hip-mlir-opt exited with code {code}"


def parse_loc_table(lines: list[str]) -> dict[str, int]:
    """Map loc identifier -> source line number."""
    table: dict[str, int] = {}
    for line in lines:
        m = _LOC_DEF.match(line.strip())
        if m:
            table[m.group(1)] = int(m.group(2))
    return table


def ops_on_line(line: str) -> list[str]:
    """Operation names produced on this line, in both MLIR syntaxes."""
    names = _QUOTED_OP.findall(line)
    names += [n for n in _BARE_OP.findall(line) if "." in n]
    return names


def read_conversion_result(after_path: Path) -> dict[int, dict]:
    """Group the conversion output by the source line each result came from.

    Returns {source_line: {"targets": [...], "helpers": [...],
    "surviving_onnx": [...]}}.
    """
    lines = after_path.read_text(encoding="utf-8").splitlines()
    loc_table = parse_loc_table(lines)

    by_src: dict[int, dict] = {}
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#loc"):
            continue
        use = _LOC_USE.search(line)
        if not use:
            continue
        src_line = loc_table.get(use.group(1))
        if src_line is None:
            continue
        entry = by_src.setdefault(
            src_line,
            {"targets": [], "helpers": [], "surviving_onnx": [], "result_text": []},
        )
        entry["result_text"].append(line.strip())
        for name in ops_on_line(line):
            if name.startswith("onnx."):
                entry["surviving_onnx"].append(name)
            elif name in DPS_HELPER_OPS:
                entry["helpers"].append(name)
            else:
                entry["targets"].append(name)
    return by_src


def parse_unconverted_diagnostic(log: str) -> dict[str, int]:
    """Parse the compiler's own unconverted-operator summary, for cross-check."""
    found: dict[str, int] = {}
    seen_header = False
    for line in log.splitlines():
        if "unconverted onnx op type(s) still live" in line:
            seen_header = True
            continue
        if not seen_header:
            continue
        m = _UNCONVERTED_LINE.match(line.rstrip())
        if not m:
            if line.strip():
                seen_header = False
            continue
        found[m.group(1)] = int(m.group(2))
    return found


def convert_candidates(
    opt: Path, candidates: list[str], work: Path, op_type: str
) -> tuple[str | None, dict[str, int], str]:
    """Convert the first candidate that converts, and name what it became.

    Returns (the module that converted, target op counts, the IR it
    produced). Trying each in turn is what makes a guessed shape usable:
    only a candidate that succeeds is evidence.
    """
    out = work / f"{op_type}.hip.mlir"
    work.mkdir(parents=True, exist_ok=True)
    for module in candidates:
        if _converts_cleanly(opt, module, work / f"{op_type}.mlir", out):
            targets: dict[str, int] = {}
            produced = out.read_text(encoding="utf-8") if out.exists() else ""
            for line in produced.splitlines():
                for name in ops_on_line(line):
                    if not name.startswith("onnx.") and name not in DPS_HELPER_OPS:
                        targets[name] = targets.get(name, 0) + 1
            return module, targets, produced
    return None, {}, ""


def lower_candidates(
    opt: Path, candidates: list[str], work: Path, op_type: str, inferred: bool
) -> dict:
    """Lower the first candidate that lowers, and name the symbols it reached.

    Returns a stage2 entry. A slice built on a guessed shape cannot convict
    the lowering -- the shape is as likely a suspect -- so its failure is
    recorded as unverified rather than broken.
    """
    work.mkdir(parents=True, exist_ok=True)
    src = work / f"{op_type}.s2.mlir"
    out = work / f"{op_type}.llvm.mlir"
    code, log = 1, "no candidates"
    for module in candidates:
        src.write_text(module, encoding="utf-8")
        code, log = run_opt(opt, src, STAGE2_PASSES, out, debug=False)
        if code == 0 and out.exists():
            break

    if code == TIMEOUT_EXIT:
        return {"status": "timeout", "error": log}
    if code != 0 or not out.exists():
        entry = {
            "status": "not_sliceable" if inferred else "lowering_broken",
            "error": _failure_reason(code, log)[:300],
        }
        if inferred:
            entry["reason"] = (
                "unranked types; a slice with an inferred shape did not lower"
            )
        return entry

    symbols = sorted(
        {
            s
            for s in _WRAP_SYM.findall(out.read_text(encoding="utf-8"))
            if not INFRA_SYMBOLS.match(s)
        }
    )
    return {"status": "ok", "runtime_funcs": symbols, "compile_time": not symbols}


# Element types worth trying when looking for the operand that blocks a
# conversion. Converters most often accept a narrower set than the model
# supplies -- MatMulNBits takes i8 or f16 zero_points but not f32.
_DTYPE_ALTERNATIVES = {
    "f32": ["f16", "i8"],
    "f16": ["f32"],
    "f64": ["f32"],
    "bf16": ["f16", "f32"],
    "i8": ["f16"],
    "ui8": ["i8", "f16"],
    "i64": ["i32"],
    "i32": ["i64"],
}


def _converts_cleanly(opt: Path, module: str, path: Path, out: Path) -> bool:
    """True when nothing named onnx.* survives conversion of this module."""
    path.write_text(module, encoding="utf-8")
    code, _ = run_opt(opt, path, STAGE1_PASSES[:-1], out, debug=False)
    if code != 0 or not out.exists():
        return False
    return not re.search(r'"onnx\.[A-Za-z0-9_]+"\s*\(', out.read_text(encoding="utf-8"))


def attribution_probe(
    opt: Path,
    src_lines: list[str],
    def_index: dict[str, int],
    line_no: int,
    op_type: str,
    work_dir: Path,
) -> dict:
    """For an operator that did not convert, find which operand blocks it.

    Change one operand's element type at a time and retry. If the conversion
    then succeeds, that operand is the reason -- a single-variable result
    that names the fix, rather than just reporting that something failed.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    parsed = parse_op_line(src_lines[line_no - 1])
    if not parsed:
        return {"found": False, "reason": "could not parse the operation"}

    for idx, ty in enumerate(parsed["in_types"]):
        elem = element_type(ty)
        if elem is None:
            continue
        for alt in _DTYPE_ALTERNATIVES.get(elem, []):
            new_type = retype(ty, alt)
            if not new_type:
                continue
            module, _why = build_single_op_module(
                src_lines, def_index, line_no, type_override=(idx, new_type)
            )
            if module is None:
                continue
            tag = f"{op_type}.operand{idx}_{alt}"
            if _converts_cleanly(
                opt, module, work_dir / f"{tag}.mlir", work_dir / f"{tag}.out.mlir"
            ):
                return {
                    "found": True,
                    "operand_index": idx,
                    "operand_type": ty,
                    "accepted_type": new_type,
                    "summary": (
                        f"operand {idx} is {ty}; converting it to {new_type} "
                        f"makes the conversion succeed"
                    ),
                }
    return {
        "found": False,
        "reason": "no single operand element type change made it convert",
    }


def perturb_value(raw: str | None) -> str | None:
    """A different but still well-formed value, or None if we cannot make one.

    Only the value matters, not what it means: the test is whether changing it
    changes the conversion output at all.
    """
    if raw is None:
        return None
    s = raw.strip()
    if s.startswith('"') and s.endswith('"'):
        return '"hipep_probe_sentinel"'
    if s in ("true", "false"):
        return "false" if s == "true" else "true"
    if re.fullmatch(r"-?\d+", s):
        return str(int(s) + 1)
    if re.fullmatch(r"-?\d*\.?\d+([eE][-+]?\d+)?", s):
        try:
            return repr(float(s) * 2.0 + 1.0)
        except ValueError:
            return None
    # Arrays, dense<> blobs, type literals: no safe generic edit.
    return None


def _rewrite_attr(module: str, attr: str, new_value: str) -> str:
    """Replace an attribute's value, keeping its MLIR type annotation.

    `axis = 0 : si64` must become `axis = 1 : si64`, not `axis = 1`: bare
    integer literals are i64, and swapping the signedness makes the converter
    take paths it never takes on real input -- one such rewrite sent the
    greedy rewrite into a loop that never terminated.
    """

    def repl(m: re.Match) -> str:
        value = m.group(2)
        typed = re.match(r"^(.*?)(\s*:\s*[a-z]?[iuf]\d+)\s*$", value)
        suffix = typed.group(2) if typed else ""
        return m.group(1) + new_value + suffix

    return re.sub(rf"(\b{re.escape(attr)}\s*=\s*)([^,}}]+)", repl, module, count=1)


def _normalize_ir(text: str) -> str:
    """Drop source-location noise so two runs compare on content alone."""
    out = []
    for line in text.splitlines():
        if line.strip().startswith("#loc"):
            continue
        out.append(re.sub(r"\s*loc\(#loc\d+\)", "", line).rstrip())
    return "\n".join(out)


def attr_present(ir_text: str, attr: str) -> bool:
    return re.search(rf"\b{re.escape(attr)}\s*=", ir_text) is not None


def merged_attributes(instances: list[dict]) -> dict:
    """Every attribute any instance of the operator sets.

    Instances need not agree: one Shape can carry `start` and the next not.
    Looking at only the first would miss whatever the others set.
    """
    merged: dict = {}
    for inst in instances:
        for name, value in (inst.get("attributes") or {}).items():
            merged.setdefault(name, value)
    return merged


def attribute_probe(
    opt: Path,
    base_module: str,
    attrs: dict,
    candidates: list[str],
    op_type: str,
    work_dir: Path,
) -> list[dict]:
    """Did the converter look at each attribute missing from its output?

    The diff that produced `candidates` cannot answer that: a converter that
    read an attribute and omitted it because it held the default value looks
    exactly like one that never read it.

    Perturbation settles it. Change the value and convert again; the verdict
    is one of three:

      handled       the output changed, the converter rejected the new
                    value, or conversion stopped terminating. All three are
                    behaviour differences, and a converter that ignores an
                    attribute has none.
      ignored       output byte-for-byte identical. It never looked, so the
                    attribute is silently dropped -- the only case worth
                    reporting.
      inconclusive  no well-formed different value could be built for this
                    value form.
    """
    if not candidates:
        return []
    work_dir.mkdir(parents=True, exist_ok=True)
    base_path = work_dir / f"{op_type}.attr_base.mlir"
    base_out = work_dir / f"{op_type}.attr_base.out.mlir"
    base_path.write_text(base_module, encoding="utf-8")
    code, _ = run_opt(opt, base_path, STAGE1_PASSES[:-1], base_out, debug=False)
    if code != 0 or not base_out.exists():
        return [
            {
                "attribute": "*",
                "verdict": "inconclusive",
                "detail": "baseline conversion of the slice failed",
            }
        ]

    base_ir = _normalize_ir(base_out.read_text(encoding="utf-8"))

    findings: list[dict] = []
    for attr in candidates:
        print(f"    attr {op_type}.{attr} ...", flush=True)
        value = attrs[attr]
        new_value = perturb_value(value)
        if new_value is None:
            findings.append(
                {
                    "attribute": attr,
                    "value": value,
                    "verdict": "inconclusive",
                    "detail": "no safe perturbation for this value form",
                }
            )
            continue

        perturbed = _rewrite_attr(base_module, attr, new_value)
        if perturbed == base_module:
            findings.append(
                {
                    "attribute": attr,
                    "value": value,
                    "verdict": "inconclusive",
                    "detail": "could not rewrite the attribute",
                }
            )
            continue

        p_path = work_dir / f"{op_type}.attr_{attr}.mlir"
        p_out = work_dir / f"{op_type}.attr_{attr}.out.mlir"
        p_path.write_text(perturbed, encoding="utf-8")
        p_code, _ = run_opt(opt, p_path, STAGE1_PASSES[:-1], p_out, debug=False)

        # Only one question: did the converter look at this attribute? A
        # different output, a rejected value and a conversion that stops
        # terminating all answer yes -- one ignoring it behaves identically
        # whatever the value.
        if p_code == TIMEOUT_EXIT:
            verdict = "handled"
            detail = "conversion stops terminating on the changed value"
        elif p_code != 0:
            verdict = "handled"
            detail = "converter rejects the changed value"
        elif _normalize_ir(p_out.read_text(encoding="utf-8")) == base_ir:
            verdict = "ignored"
            detail = "output identical after changing the value"
        else:
            verdict = "handled"
            detail = "output changed with the value"

        findings.append(
            {
                "attribute": attr,
                "value": value,
                "perturbed_to": new_value,
                "verdict": verdict,
                "detail": detail,
            }
        )
    return findings


def stage1_per_operator(
    opt: Path, mlir_path: Path, ops: dict, out_dir: Path
) -> dict[str, dict]:
    """Fallback for when the whole-graph probe fails.

    One operator failing the pass takes the whole run down with it -- an
    onnx.If whose output shapes are not static, say -- and the other
    operators' verdicts go with it. Slicing each one out isolates them, at
    the cost of losing context: fusions that need neighbouring operators
    cannot fire, so support here is a lower bound.
    """
    src_lines = mlir_path.read_text(encoding="utf-8").splitlines()
    def_index = build_def_index(src_lines)
    work = out_dir / "stage1_slices"
    work.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    for op_type, info in ops.items():
        if op_type.startswith("_"):
            continue
        instances = info.get("instances") or []
        if not instances:
            continue
        candidates, line_no, why, inferred = slices_for(src_lines, def_index, instances)
        base = {
            "domain": (info.get("domain") or ["onnx"])[0],
            "count": info["count"],
            "targets": {},
            "helpers": {},
            "folded": 0,
            "dropped_attrs": [],
            "unconverted_lines": [],
        }
        if not candidates:
            results[op_type] = {
                **base,
                "converted": 0,
                "unconverted": 0,
                "inconclusive": info["count"],
                "reason": why,
            }
            continue

        print(f"  stage1-slice {op_type} ...", flush=True)
        converted, targets, produced = convert_candidates(
            opt, candidates, work, op_type
        )
        converts = converted is not None
        if not converts and inferred:
            # The slice only exists because we invented a shape for it, so
            # its failure may be ours rather than the converter's.
            results[op_type] = {
                **base,
                "converted": 0,
                "unconverted": 0,
                "inconclusive": info["count"],
                "reason": "unranked types; a slice with an inferred shape did not convert",
            }
            continue
        # The slice is one instance, so only its own attributes can be
        # compared against what it produced -- unlike the whole-graph path,
        # which sees every instance.
        sliced = next((i for i in instances if i["line_no"] == line_no), instances[0])
        attrs = sliced.get("attributes") or {}
        results[op_type] = {
            **base,
            "converted": info["count"] if converts else 0,
            "unconverted": 0 if converts else info["count"],
            "targets": targets,
            "unconverted_lines": [] if converts else [line_no],
            "attributes_seen": attrs,
            "dropped_attrs": [a for a in attrs if not attr_present(produced, a)]
            if converts
            else [],
        }
    return results


def stage2(
    opt: Path, mlir_path: Path, ops: dict, stage1_results: dict, out_dir: Path
) -> dict[str, dict]:
    """Per-operator lowering probe: is the chain complete, and to what symbol?"""
    src_lines = mlir_path.read_text(encoding="utf-8").splitlines()
    def_index = build_def_index(src_lines)
    slice_dir = out_dir / "slices"
    slice_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    for op_type, info in ops.items():
        if op_type.startswith("_"):
            continue
        s1 = stage1_results.get(op_type, {})
        instances = info["instances"]
        if not instances:
            results[op_type] = {"status": "skipped", "reason": "no instances"}
            continue

        if s1.get("unconverted"):
            # Nothing converted, so there is no lowering to test. Ask instead
            # which operand the converter objects to -- that is the finding
            # the report needs for a blocked operator.
            bad_line = (s1.get("unconverted_lines") or [instances[0]["line_no"]])[0]
            print(f"  attribution {op_type} ...", flush=True)
            results[op_type] = {
                "status": "skipped",
                "reason": "unconverted in stage1",
                "attribution": attribution_probe(
                    opt,
                    src_lines,
                    def_index,
                    bad_line,
                    op_type,
                    out_dir / "attribution",
                ),
            }
            continue

        candidates, line_no, why, inferred = slices_for(src_lines, def_index, instances)
        if not candidates:
            # Not a lowering failure: stage1 showed it converts, and only the
            # rest of the chain goes unchecked.
            results[op_type] = {
                "status": "not_sliceable",
                "source_line": line_no,
                "reason": f"{why} (no usable instance among {len(instances)})",
            }
            continue

        print(f"  stage2 {op_type} ...", flush=True)
        entry = lower_candidates(opt, candidates, slice_dir, op_type, inferred)
        if entry["status"] != "ok":
            results[op_type] = {**entry, "source_line": line_no}
            continue

        # Only operators stage1 saw lose an attribute need this; the same
        # slice serves, so nothing extra is built.
        results[op_type] = {
            **entry,
            "source_line": line_no,
            # Whichever set stage1 compared against: the union over all
            # instances on the whole-graph path, one instance's on the
            # sliced one. Mixing them would offer an attribute for
            # perturbation that is not in this slice.
            "attributes": attribute_probe(
                opt,
                candidates[0],
                s1.get("attributes_seen") or merged_attributes(instances),
                s1.get("dropped_attrs", []),
                op_type,
                out_dir / "attrs",
            ),
        }
    return results


def stage1(
    opt: Path, mlir_path: Path, ops: dict, out_dir: Path
) -> tuple[dict[str, dict], dict]:
    """Whole-graph probe. Returns (per-operator result, run metadata)."""
    after = out_dir / "stage1_after.mlir"
    code, log = run_opt(opt, mlir_path, STAGE1_PASSES, after)
    (out_dir / "stage1.log").write_text(log, encoding="utf-8")

    meta = {
        "exit_code": code,
        "passes": STAGE1_PASSES,
        "output": str(after),
        "diagnostic": parse_unconverted_diagnostic(log),
    }
    if code != 0 or not after.exists():
        meta["failed"] = True
        meta["error"] = _failure_reason(code, log)
        return {}, meta

    by_src = read_conversion_result(after)

    results: dict[str, dict] = {}
    for op_type, info in ops.items():
        if op_type.startswith("_"):
            continue
        converted = 0
        unconverted = 0
        folded = 0
        targets: dict[str, int] = {}
        helpers: dict[str, int] = {}
        unconverted_lines: list[int] = []
        for inst in info["instances"]:
            entry = by_src.get(inst["line_no"])
            if entry is None:
                # No output carries this source line: the operation was
                # folded, or fused into a neighbour that now owns the result.
                # Either way it converted, leaving nothing behind.
                converted += 1
                folded += 1
                continue
            if entry["surviving_onnx"]:
                unconverted += 1
                unconverted_lines.append(inst["line_no"])
            else:
                converted += 1
            for t in entry["targets"]:
                targets[t] = targets.get(t, 0) + 1
            for h in entry["helpers"]:
                helpers[h] = helpers.get(h, 0) + 1
        # Shape and ConstantOfShape lower entirely into what is normally DPS
        # scaffolding; with nothing else to report, report that.
        if not targets and helpers:
            targets = helpers
            helpers = {}

        # Attributes that did not survive conversion. Free to collect -- the
        # whole-graph output is already in hand -- and only these operators
        # go on to the perturbation test.
        # Every instance, not just the first: they need not set the same
        # attributes, and each is compared against the IR it produced.
        dropped: set[str] = set()
        for inst in info["instances"]:
            entry = by_src.get(inst["line_no"])
            produced = " ".join(entry["result_text"]) if entry else ""
            if not produced:
                continue
            dropped |= {
                a
                for a in (inst.get("attributes") or {})
                if not attr_present(produced, a)
            }
        dropped_attrs = sorted(dropped)
        results[op_type] = {
            "domain": info["domain"][0] if info["domain"] else "onnx",
            "count": info["count"],
            "converted": converted,
            "unconverted": unconverted,
            "folded": folded,
            "targets": dict(sorted(targets.items(), key=lambda kv: -kv[1])),
            "helpers": dict(sorted(helpers.items(), key=lambda kv: -kv[1])),
            "unconverted_lines": unconverted_lines[:10],
            "dropped_attrs": dropped_attrs,
        }
    return results, meta


def main() -> None:
    ap = argparse.ArgumentParser(description="Compile-based operator support probe")
    ap.add_argument("mlir_path", type=Path)
    ap.add_argument("ops_json", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--package", type=Path, required=True, help="gpu-test-package root")
    ap.add_argument(
        "--stage1-only",
        action="store_true",
        help="Skip the per-operator lowering probe",
    )
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    opt = _tool(args.package, "hip-mlir-opt.exe")
    ops = json.loads(args.ops_json.read_text(encoding="utf-8"))

    results, meta = stage1(opt, args.mlir_path, ops, args.out_dir)

    print(f"stage1 exit={meta['exit_code']}")
    if meta.get("failed"):
        print(f"  FAILED: {meta.get('error', '')}")
        print("  falling back to per-operator slices")
        results = stage1_per_operator(opt, args.mlir_path, ops, args.out_dir)
        meta["fallback"] = "per-operator slices"
        meta["fallback_note"] = (
            "whole-graph conversion failed; each operator was converted in "
            "isolation, so fusions needing surrounding context could not fire "
            "and support is a lower bound"
        )

    print(f"{'operator':36s} {'conv':>5s} {'unconv':>7s} {'folded':>7s}  targets")
    print("-" * 104)
    for op, r in sorted(results.items(), key=lambda kv: -kv[1]["count"]):
        tgt = ", ".join(f"{k}x{v}" for k, v in list(r["targets"].items())[:3])
        if not tgt and r["folded"]:
            tgt = "(folded at compile time)"
        print(
            f"{op:36s} {r['converted']:5d} {r['unconverted']:7d} "
            f"{r['folded']:7d}  {tgt}"
        )
    print()
    print(f"compiler diagnostic (cross-check): {meta['diagnostic']}")

    s2: dict[str, dict] = {}
    if not args.stage1_only:
        print()
        s2 = stage2(opt, args.mlir_path, ops, results, args.out_dir)
        print(f"{'operator':36s} {'stage2':16s}  runtime / detail")
        print("-" * 104)
        for op, r in sorted(s2.items(), key=lambda kv: -ops[kv[0]]["count"]):
            if r["status"] == "ok":
                detail = (
                    ", ".join(r["runtime_funcs"])
                    if r["runtime_funcs"]
                    else "(compile-time only)"
                )
            else:
                detail = r.get("error") or r.get("reason", "")
            print(f"{op:36s} {r['status']:16s}  {detail[:60]}")

        flagged = [
            (op, f)
            for op, r in s2.items()
            for f in r.get("attributes", [])
            if f["verdict"] != "handled"
        ]
        if flagged:
            print()
            print("attributes the converter does not read:")
            for op, f in flagged:
                note = "" if f["verdict"] == "ignored" else f"  ({f['detail']})"
                print(
                    f"  [{f['verdict']:12s}] {op}.{f['attribute']} = "
                    f"{f.get('value', '')}{note}"
                )

    out = {
        "stage1": {"meta": meta, "operators": results},
        "stage2": {"passes": STAGE2_PASSES, "operators": s2},
    }
    (args.out_dir / "probe_result.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print()
    print(f"[OK] {args.out_dir / 'probe_result.json'}")


if __name__ == "__main__":
    main()
