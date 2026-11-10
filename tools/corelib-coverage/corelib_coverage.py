#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Which ops of an ONNX model hip-ep can run on the NPU through RyzenAI CoreLib.

Reads the graph only (no weights, no GPU, no NPU), lists every op hip-ep routes
to CoreLib -- MatMulNBits, SimplifiedLayerNormalization / RMSNormalization and
SkipSimplifiedLayerNormalization -- with its shape, applies hip-ep's own
weight-format rules (lib/Runtime/real/ryzenaicorelib.cpp), and, given a kernel
store, the PDIs that ship a kernel at each shape and a RYZENAICORELIB_PDI that
reaches them. See docs/design/ryzenai-corelib-backend.md.

  python corelib_coverage.py model.onnx [model2.onnx ...]
      [--kernels <transaction/mds dir | name list file | DynamicDispatch repo>]
      [--ref <git ref, with a repo>] [--json out.json] [--write-pdi]

--write-pdi writes the suggested setting beside each model as
<model>.corelib_pdi.txt (one line, RYZENAICORELIB_PDI=<prefill>;<decode>), which
run_model.bat reads when no PDI is given, so the setting travels with the model.

The store's kernels are read from the names CoreLib resolves:
  mladfmatmulbias/mladf_aie4_p<pdi>_abfp16wbfp16acc16f_<m>_<k>_<n>_<group>[_bias]
  mladfrmsnorm/rmsnorm_aie4_p<pdi>_a16_<m>_<k>
  addrmsnorm/addrmsnorm_aie4_p<pdi>_a16a16_<m>_<k>
A name list holds one store-relative path (or file name) per line, as
`git ls-tree -r --name-only <ref> -- transaction/mds` prints them.
"""

import argparse
import collections
import json
import os
import re
import subprocess
import sys

import onnx
from onnx import TensorProto, helper

MATMUL = re.compile(
    r"(?:^|/)mladfmatmulbias/mladf_aie4_p(\d+)_abfp16wbfp16acc16f_"
    r"(\d+)_(\d+)_(\d+)_(\d+)(_bias)?\.elf$"
)
NORM = re.compile(r"(?:^|/)mladfrmsnorm/rmsnorm_aie4_p(\d+)_a16_(\d+)_(\d+)\.elf$")
ADD_NORM = re.compile(
    r"(?:^|/)addrmsnorm/addrmsnorm_aie4_p(\d+)_a16a16_(\d+)_(\d+)\.elf$"
)
# Plain file names (no directory) match only when the list carries no paths.
MATMUL_BARE = re.compile(MATMUL.pattern.replace(r"(?:^|/)mladfmatmulbias/", "^"))
NORM_BARE = re.compile(NORM.pattern.replace(r"(?:^|/)mladfrmsnorm/", "^"))
ADD_NORM_BARE = re.compile(ADD_NORM.pattern.replace(r"(?:^|/)addrmsnorm/", "^"))

NORM_OPS = {"SimplifiedLayerNormalization", "RMSNormalization"}
SKIP_NORM_OPS = {"SkipSimplifiedLayerNormalization"}


def kernel_group(block):
    """CoreLib derives the kernel group: divisible by 64 takes g64, else g32."""
    return 64 if block % 64 == 0 else 32


# ---------------------------------------------------------------------------
# Kernel store
# ---------------------------------------------------------------------------


class Kernels:
    """Row counts per PDI: matmul[(k, n, group, bias)], norm[k], add_norm[k]."""

    def __init__(self, names):
        self.matmul = collections.defaultdict(lambda: collections.defaultdict(set))
        self.norm = collections.defaultdict(lambda: collections.defaultdict(set))
        self.add_norm = collections.defaultdict(lambda: collections.defaultdict(set))
        self.count = 0
        for name in names:
            name = name.strip().replace("\\", "/")
            bare = "/" not in name
            m = (MATMUL_BARE if bare else MATMUL).search(name)
            if m:
                pdi, rows, k, n, group = map(int, m.groups()[:5])
                self.matmul[(k, n, group, bool(m.group(6)))][pdi].add(rows)
                self.count += 1
                continue
            for pattern, table in (
                (NORM_BARE if bare else NORM, self.norm),
                (ADD_NORM_BARE if bare else ADD_NORM, self.add_norm),
            ):
                m = pattern.search(name)
                if m:
                    pdi, rows, k = map(int, m.groups())
                    table[k][pdi].add(rows)
                    self.count += 1
                    break

    def for_matmul(self, k, n, block, bias):
        """PDI -> rows, either spelling of the group, as hip-ep accepts."""
        found = collections.defaultdict(set)
        for group in {block, kernel_group(block)}:
            for pdi, rows in self.matmul.get((k, n, group, bias), {}).items():
                found[pdi] |= rows
        return found


def read_kernel_names(path, ref):
    if ref:
        out = subprocess.run(
            [
                "git",
                "-C",
                path,
                "ls-tree",
                "-r",
                "--name-only",
                ref,
                "--",
                "transaction/mds",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        return out.splitlines()
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            return f.read().splitlines()
    names = []
    for root, _, files in os.walk(path):
        rel = os.path.relpath(root, path).replace("\\", "/")
        names += [f if rel == "." else f"{rel}/{f}" for f in files]
    return names


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def all_nodes(graph):
    for node in graph.node:
        yield node
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                yield from all_nodes(attr.g)
            elif attr.type == onnx.AttributeProto.GRAPHS:
                for g in attr.graphs:
                    yield from all_nodes(g)


def tensor_info(model):
    """name -> (dims, elem_type) for initializers and typed values."""
    info = {}
    for init in model.graph.initializer:
        info[init.name] = (list(init.dims), init.data_type)
    for vi in list(model.graph.value_info) + list(model.graph.input):
        t = vi.type.tensor_type
        if t.HasField("shape"):
            dims = [
                d.dim_value if d.HasField("dim_value") else None for d in t.shape.dim
            ]
            info.setdefault(vi.name, (dims, t.elem_type))
    return info


def present(node, i):
    return len(node.input) > i and node.input[i] != ""


def matmul_row(node, info):
    a = {x.name: helper.get_attribute_value(x) for x in node.attribute}
    k, n, block, bits = a["K"], a["N"], a.get("block_size", 0), a.get("bits", 4)
    zp, g_idx, bias = present(node, 3), present(node, 4), present(node, 5)
    shape = f"K={k} N={n} block={block}" + (f" bits={bits}" if bits != 4 else "")
    shape += (" zp" if zp else "") + (" bias" if bias else "")
    why = None
    if bits != 4:
        why = "corelib takes 4-bit weights only"
    elif g_idx:
        why = "g_idx is not supported"
    elif block % 32 or k % block or k % (2 * kernel_group(block)):
        why = "block size or K does not fit corelib's 4-bit layout"
    elif (
        zp
        and info.get(node.input[3], (None, TensorProto.UINT8))[1] != TensorProto.UINT8
    ):
        why = "unsupported zero point type"
    elif info.get(node.input[2], (None, TensorProto.FLOAT16))[1] not in (
        TensorProto.FLOAT16,
        TensorProto.FLOAT,
    ):
        why = "unsupported scale type"
    return ("MatMulNBits", shape, (k, n, block, bias), why)


def norm_row(node, info, skip):
    scale = node.input[2] if skip else node.input[1]
    dims = info.get(scale, ([None], 0))[0]
    k = dims[-1] if dims else None
    bias = skip and present(node, 3)
    op = "SkipRmsNorm" if skip else "RmsNorm"
    shape = f"k={k if k else '?'}" + (" bias" if bias else "")
    why = None
    if bias:
        why = "corelib's add_rmsnorm takes no bias"
    elif not k:
        why = "width unknown statically (scale is not an initializer)"
    return (op, shape, k, why)


def analyze(path):
    model = onnx.load(path, load_external_data=False)
    info = tensor_info(model)
    rows = collections.OrderedDict()
    for node in all_nodes(model.graph):
        if node.op_type == "MatMulNBits":
            row = matmul_row(node, info)
        elif node.op_type in NORM_OPS:
            row = norm_row(node, info, skip=False)
        elif node.op_type in SKIP_NORM_OPS:
            row = norm_row(node, info, skip=True)
        else:
            continue
        key = row[:2]
        if key in rows:
            rows[key][1] += 1
        else:
            rows[key] = [row, 1]
    return [(row, count) for row, count in rows.values()]


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def describe(pdis):
    """'p1 (64..4096), p16 (1)'"""
    parts = []
    for pdi in sorted(pdis):
        rows = sorted(pdis[pdi])
        span = str(rows[0]) if len(rows) == 1 else f"{rows[0]}..{rows[-1]}"
        parts.append(f"p{pdi} ({span})")
    return ", ".join(parts)


def choose_pdis(needs):
    """Greedy cover: the PDIs reaching the most op instances, in that order."""
    chosen, remaining = [], list(needs)
    while remaining:
        score = collections.Counter()
        for pdis, weight in remaining:
            for pdi in pdis:
                score[pdi] += weight
        if not score:
            break
        best = max(score, key=lambda p: (score[p], -p))
        chosen.append(best)
        remaining = [(p, w) for p, w in remaining if best not in p]
    return chosen


def report(path, kernels):
    """Prints the model's table; returns it as a dict for --json."""
    rows = analyze(path)
    result = {"model": path, "ops": [], "instances": 0, "npu_instances": 0, "pdi": None}
    print(f"\n{path}")
    if not rows:
        print(
            "  no MatMulNBits, SimplifiedLayerNormalization, RMSNormalization or "
            "SkipSimplifiedLayerNormalization: nothing is routed to corelib"
        )
        return result
    head = f"  {'op':<12} {'shape':<34} {'inst':>4}  "
    print(head + ("NPU kernels (PDI (rows))" if kernels else "routable"))
    prefill_needs, decode_needs = [], []
    total = npu = 0
    for (op, shape, key, why), count in rows:
        total += count
        entry = {"op": op, "shape": shape, "instances": count}
        result["ops"].append(entry)
        if why:
            entry["gpu_because"] = why
            print(f"  {op:<12} {shape:<34} {count:>4}  GPU: {why}")
            continue
        if not kernels:
            npu += count
            print(f"  {op:<12} {shape:<34} {count:>4}  yes")
            continue
        if op == "MatMulNBits":
            pdis = kernels.for_matmul(*key)
        else:
            pdis = (kernels.add_norm if op == "SkipRmsNorm" else kernels.norm).get(
                key, {}
            )
        if not pdis:
            entry["gpu_because"] = "no kernel at this shape"
            print(f"  {op:<12} {shape:<34} {count:>4}  GPU: no kernel at this shape")
            continue
        npu += count
        entry["pdis"] = {str(p): sorted(r) for p, r in sorted(pdis.items())}
        print(f"  {op:<12} {shape:<34} {count:>4}  {describe(pdis)}")
        # Prefill takes the PDIs shipping several rows; decode those built for
        # one row, else any shipping one (a dispatch tries its own phase's
        # list, then the other's).
        prefill = {p for p, r in pdis.items() if max(r) > 1}
        decode = {p for p, r in pdis.items() if r == {1}} or {
            p for p, r in pdis.items() if 1 in r
        }
        if prefill:
            prefill_needs.append((prefill, count))
        if decode:
            decode_needs.append((decode, count))
    verb = "with kernels" if kernels else "routable (weight format allows it)"
    print(f"  total: {npu} of {total} op instances {verb}")
    result["instances"], result["npu_instances"] = total, npu
    if kernels and (prefill_needs or decode_needs):
        pre, dec = choose_pdis(prefill_needs), choose_pdis(decode_needs)
        result["pdi"] = (
            f"{','.join(map(str, pre)) or '1'};{','.join(map(str, dec)) or '16'}"
        )
        print(f"  RYZENAICORELIB_PDI={result['pdi']}")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("models", nargs="+", help="ONNX model files (graph only is read)")
    ap.add_argument(
        "--kernels",
        help="transaction/mds directory, a file listing "
        "kernel names, or a DynamicDispatch repository (with --ref)",
    )
    ap.add_argument(
        "--ref",
        help="git ref of the --kernels repository, e.g. corelib/kernels/20261029.0",
    )
    ap.add_argument("--json", help="also write every model's table to this file")
    ap.add_argument(
        "--write-pdi",
        action="store_true",
        help="write <model>.corelib_pdi.txt beside each model (needs --kernels)",
    )
    args = ap.parse_args()
    if args.write_pdi and not args.kernels:
        ap.error("--write-pdi needs --kernels")

    kernels = None
    if args.kernels:
        kernels = Kernels(read_kernel_names(args.kernels, args.ref))
        print(
            f"kernel store: {args.kernels}{' @ ' + args.ref if args.ref else ''}"
            f" ({kernels.count} matmul and norm kernels)"
        )
        if not kernels.count:
            print(
                "  no kernels found: check the path (a sparse checkout needs "
                "mladfmatmulbias, mladfrmsnorm and addrmsnorm)"
            )
    results = []
    for path in args.models:
        result = report(path, kernels)
        results.append(result)
        if args.write_pdi and result["pdi"]:
            sidecar = path + ".corelib_pdi.txt"
            with open(sidecar, "w", encoding="utf-8") as f:
                f.write(f"RYZENAICORELIB_PDI={result['pdi']}\n")
            print(f"  wrote {sidecar}")
    if args.json:
        store = {"kernels": args.kernels, "ref": args.ref}
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"kernel_store": store, "models": results}, f, indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
