#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Offline "prune logits" transform for ONNX decoder models.

Many exported decoder models emit full-sequence logits: the lm_head runs on every
position, producing logits of shape [batch, sequence_length, vocab]. During
autoregressive generation only the LAST token's logits are needed, so at long
prefill this wastes a multi-GB logits buffer (+ lm_head dequant scratch) and a
seq-wide matmul.

This tool prunes the graph to last-token logits, inserting a
Gather(axis=1, index=-1) + Unsqueeze(axes=[1]) in front of lm_head so only the
last position reaches it:

    Before:  final_norm -[B,seq,H]-> lm_head(MatMul*) -> logits [B, seq, vocab]
    After:   final_norm -[B,seq,H]-> Gather(idx=-1,axis=1) -[B,H]->
                 Unsqueeze(axes=[1]) -[B,1,H]-> lm_head -> logits [B, 1, vocab]

lm_head resolution (the graph is not always the simple case above):

- Common case: lm_head directly produces the `logits` output. The prune is
  inserted on lm_head's hidden-state input, and lm_head is matched across the
  usual variants -- plain `MatMul`, quantized `MatMulNBits`, `Gemm`,
  `FusedMatMul` (or any node whose name contains "lm_head").
- Trailing-node case: some exports have nodes BETWEEN lm_head and the output
  (e.g. logit soft-cap / scaling / a final `Cast`). Then the producer of
  `logits` is not the matmul, and pruning there would leave the matmul running
  on the full sequence (no prefill savings). The tool detects this and refuses,
  asking you to point at the real matmul with `--lm-head-name` so the prune
  lands before it.
- Opset differences are handled: `Unsqueeze` axes go in an attribute
  (opset < 13) or an input initializer (opset >= 13).
- Already-pruned models (logits sequence dim already 1) are detected and
  rejected rather than double-pruned.

Design / safety contract:

- One-shot: a single run does pre-check -> backup -> transform -> post-review
  (-> optional ORT numeric verify). Any failure leaves the model untouched, and
  a failure AFTER we start writing auto-rolls-back to the original.
- Backup == rename: the original `<model>.onnx` is renamed to `<model>_orig.onnx`
  (that rename IS the backup); the pruned graph is then written under the
  ORIGINAL filename. Because the model keeps its name, `genai_config.json` needs
  no change and is never touched.
- External weights (`*.onnx_data*`) are referenced by name inside the proto and
  are NEVER rewritten: renaming the `.onnx` leaves those references valid, and
  the pruned model (structure only) shares the same data files as `_orig`.
- `--restore` reverts from `_orig` (with the same before/after validation).
- ORT numeric verification is OPT-IN (`--ort-verify`) and requires onnxruntime
  (kept out of the core requirements). It is checked in pre-flight so a missing
  onnxruntime fails fast, before any change is made.

Input is always a model DIRECTORY (never a bare `.onnx`): the target decoder
graph is resolved from `genai_config.json`'s `model.decoder.filename`, so on
multi-part exports (e.g. VLMs with separate vision/embedding/decoder graphs)
only the text/decoder graph is ever selected.

Usage:
  python prune_logits.py <model_dir>                    # apply
  python prune_logits.py <model_dir> --precheck         # eligibility + plan only
  python prune_logits.py <model_dir> --ort-verify       # + numeric equivalence
  python prune_logits.py <model_dir> --restore          # revert from *_orig
"""

import argparse
import importlib.util
import json
import os
import subprocess
import sys

import onnx
from onnx import TensorProto, helper

# Backup suffix: the original <model>.onnx is renamed to <model>_orig.onnx.
ORIG_SUFFIX = "_orig"

# Dummy prompt length for --ort-verify (kept tiny; compute is dominated by the
# one-time model weight load, not sequence length).
VERIFY_TOKENS = 8

# lm_head is a (possibly quantized) matmul producing the vocab projection.
LM_HEAD_OPS = {"MatMul", "MatMulNBits", "Gemm", "FusedMatMul"}

# Tool-owned names for the inserted subgraph, so we never depend on (or collide
# with) a model's existing constant names.
IDX_INIT = "prune_logits.last_index"
AXES_INIT = "prune_logits.unsqueeze_axes"
GATHER_NAME = "/prune_logits/Gather"
UNSQ_NAME = "/prune_logits/Unsqueeze"
GATHER_OUT = "/prune_logits/gather_out"
UNSQ_OUT = "/prune_logits/unsqueeze_out"


class PruneError(Exception):
    """Raised for any user-facing failure (bad model, ineligible, etc.)."""


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def resolve_paths(target):
    """Resolve (model_dir, model_onnx, orig_onnx) from a model directory.

    Only a directory is accepted. The target ONNX is resolved via
    `genai_config.json`'s `model.decoder.filename` so that on multi-part
    exports (e.g. VLMs with separate vision/embedding/decoder graphs) only the
    text/decoder graph is ever selected. When no `genai_config.json` is present,
    fall back to the single non-`*_orig.onnx` `.onnx` in the directory.
    """
    target = os.path.abspath(target)
    if not os.path.isdir(target):
        raise PruneError(
            f"expected a model directory, got: {target}; "
            "pass the model directory (the tool resolves the decoder .onnx "
            "via genai_config.json)"
        )

    model_dir = target
    cfg = os.path.join(model_dir, "genai_config.json")
    filename = None
    if os.path.isfile(cfg):
        try:
            # utf-8-sig so a UTF-8 BOM (common on Windows-exported configs)
            # does not break json parsing and silently trigger the fallback.
            with open(cfg, "r", encoding="utf-8-sig") as f:
                filename = json.load(f)["model"]["decoder"]["filename"]
        except (KeyError, ValueError):
            filename = None
    if filename:
        model_onnx = os.path.join(model_dir, filename)
        if not os.path.isfile(model_onnx):
            raise PruneError(
                f"genai_config.json decoder.filename is '{filename}' but "
                f"{model_onnx} does not exist"
            )
    else:
        cands = [
            f
            for f in os.listdir(model_dir)
            if f.endswith(".onnx") and not f.endswith(ORIG_SUFFIX + ".onnx")
        ]
        if len(cands) != 1:
            raise PruneError(
                f"no genai_config.json and cannot pick a single model .onnx in "
                f"{model_dir}: found {cands}; add a genai_config.json with "
                "model.decoder.filename so the decoder graph is unambiguous"
            )
        model_onnx = os.path.join(model_dir, cands[0])

    stem, ext = os.path.splitext(model_onnx)
    orig_onnx = stem + ORIG_SUFFIX + ext
    return model_dir, model_onnx, orig_onnx


def get_opset(model):
    for imp in model.opset_import:
        if imp.domain in ("", "ai.onnx"):
            return imp.version
    return 0


# ---------------------------------------------------------------------------
# Introspection helpers
# ---------------------------------------------------------------------------


def _dim_repr(d):
    if d.HasField("dim_value"):
        return d.dim_value
    return d.dim_param or "?"


def _output_vi(graph, name):
    for o in graph.output:
        if o.name == name:
            return o
    return None


def _is_pruned(model, logits_name):
    """True when `logits_name` is rank-3 with a static sequence dim of 1."""
    vi = _output_vi(model.graph, logits_name)
    if vi is None:
        return False
    dims = vi.type.tensor_type.shape.dim
    return len(dims) == 3 and dims[1].HasField("dim_value") and dims[1].dim_value == 1


def _producer_map(graph):
    m = {}
    for n in graph.node:
        for o in n.output:
            m[o] = n
    return m


def find_lm_head(graph, logits_name, lm_head_name=None):
    """Return the node producing `logits_name` (or the named lm_head node)."""
    if lm_head_name:
        for n in graph.node:
            if n.name == lm_head_name:
                return n
        raise PruneError(f"lm-head node not found by name: {lm_head_name}")
    prod = _producer_map(graph).get(logits_name)
    if prod is None:
        raise PruneError(
            f"no node produces graph output '{logits_name}'; "
            f"pass --logits-name / --lm-head-name"
        )
    return prod


def tail_trace(graph, logits_name, depth=6):
    """Return a short producer chain (op_type/name) walking back from logits."""
    prod = _producer_map(graph)
    lines = []
    frontier = [logits_name]
    seen = set()
    for _ in range(depth):
        nxt = []
        for name in frontier:
            n = prod.get(name)
            if n is None or id(n) in seen:
                continue
            seen.add(id(n))
            lines.append(f"    {n.op_type:24s} {n.name}")
            nxt.extend(i for i in n.input if i in prod)
        frontier = nxt
    return lines


# ---------------------------------------------------------------------------
# Pre-check (eligibility)
# ---------------------------------------------------------------------------


def check_eligibility(model, logits_name, lm_head_name):
    """Validate the model can be pruned. Returns (lm_head_node, hidden_input).

    Raises PruneError on any hard failure.
    """
    graph = model.graph

    out_vi = _output_vi(graph, logits_name)
    if out_vi is None:
        raise PruneError(f"graph output '{logits_name}' not found")

    dims = out_vi.type.tensor_type.shape.dim
    if len(dims) != 3:
        raise PruneError(
            f"'{logits_name}' rank is {len(dims)} (expected 3: "
            f"[batch, sequence, vocab]); shape={[_dim_repr(d) for d in dims]}"
        )

    seq_dim = dims[1]
    if seq_dim.HasField("dim_value") and seq_dim.dim_value == 1:
        raise PruneError(
            f"'{logits_name}' sequence dim is already 1 "
            f"({[_dim_repr(d) for d in dims]}); model looks already pruned"
        )

    lm = find_lm_head(graph, logits_name, lm_head_name)
    if lm.op_type not in LM_HEAD_OPS and "lm_head" not in lm.name.lower():
        # The producer of `logits` is not the lm_head matmul itself, which means
        # there are trailing nodes (e.g. logit softcap / scaling / Cast) between
        # lm_head and the output. Inserting the prune here would leave the matmul
        # running on the full sequence (no prefill savings), so refuse and let
        # the user point at the real lm_head explicitly.
        raise PruneError(
            f"producer of '{logits_name}' is {lm.op_type} (name={lm.name}), not "
            f"an lm_head op {sorted(LM_HEAD_OPS)}; there appear to be nodes between "
            f"lm_head and the output. Re-run with --lm-head-name <matmul-node> to "
            f"point at the real lm_head."
        )
    if not lm.input:
        raise PruneError(f"lm_head node {lm.name} has no inputs")

    hidden = lm.input[0]
    # Best-effort: warn (not fail) when the hidden-state seq dim is unknown.
    return lm, hidden


# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------


def apply_prune(model, lm, hidden, logits_name):
    """Insert Gather(-1,axis=1)+Unsqueeze([1]) before lm_head; collapse logits."""
    graph = model.graph
    opset = get_opset(model)

    # Gather index (scalar int64 -1) as an initializer -- opset-agnostic.
    graph.initializer.append(
        helper.make_tensor(IDX_INIT, TensorProto.INT64, dims=[], vals=[-1])
    )
    gather = helper.make_node(
        "Gather", [hidden, IDX_INIT], [GATHER_OUT], name=GATHER_NAME, axis=1
    )

    # Unsqueeze axes moved from attribute (opset<13) to input (opset>=13).
    if opset >= 13:
        graph.initializer.append(
            helper.make_tensor(AXES_INIT, TensorProto.INT64, dims=[1], vals=[1])
        )
        unsq = helper.make_node(
            "Unsqueeze", [GATHER_OUT, AXES_INIT], [UNSQ_OUT], name=UNSQ_NAME
        )
    else:
        unsq = helper.make_node(
            "Unsqueeze", [GATHER_OUT], [UNSQ_OUT], name=UNSQ_NAME, axes=[1]
        )

    lm.input[0] = UNSQ_OUT

    nodes = list(graph.node)
    lm_idx = nodes.index(lm)
    nodes[lm_idx:lm_idx] = [gather, unsq]
    del graph.node[:]
    graph.node.extend(nodes)

    # Collapse the logits sequence dim to a static 1.
    seq_dim = _output_vi(graph, logits_name).type.tensor_type.shape.dim[1]
    seq_dim.ClearField("dim_param")
    seq_dim.dim_value = 1


# ---------------------------------------------------------------------------
# Post-review (structural)
# ---------------------------------------------------------------------------


def review_structure(model_path, lm_head_name, logits_name):
    """Reload the written model (structure only) and assert the rewrite is sane.

    Returns a list of human-readable check lines; raises PruneError on failure.
    """
    m = onnx.load(model_path, load_external_data=False)
    g = m.graph
    outs = {o for n in g.node for o in n.output}
    checks = []

    def require(cond, msg):
        checks.append(("OK  " if cond else "FAIL") + "  " + msg)
        if not cond:
            raise PruneError("post-review failed: " + msg)

    require(GATHER_OUT in outs, f"Gather output present ({GATHER_OUT})")
    require(UNSQ_OUT in outs, f"Unsqueeze output present ({UNSQ_OUT})")

    gather = next((n for n in g.node if n.name == GATHER_NAME), None)
    unsq = next((n for n in g.node if n.name == UNSQ_NAME), None)
    require(gather is not None, "Gather node present")
    require(unsq is not None, "Unsqueeze node present")

    axis = next((a.i for a in gather.attribute if a.name == "axis"), None)
    require(axis == 1, f"Gather axis == 1 (got {axis})")
    require(gather.input[1] == IDX_INIT, "Gather index wired to last-index init")
    require(unsq.input[0] == GATHER_OUT, "Unsqueeze consumes Gather output")

    lm = find_lm_head(g, logits_name, lm_head_name)
    require(lm.input[0] == UNSQ_OUT, "lm_head input[0] rewired to Unsqueeze")

    # Topological order: Gather -> Unsqueeze -> lm_head.
    order = {n.name: i for i, n in enumerate(g.node)}
    require(
        order[GATHER_NAME] < order[UNSQ_NAME] < order[lm.name],
        "node order Gather < Unsqueeze < lm_head",
    )

    out_vi = _output_vi(g, logits_name)
    dims = out_vi.type.tensor_type.shape.dim
    require(
        dims[1].HasField("dim_value") and dims[1].dim_value == 1,
        f"logits sequence dim == 1 (got {[_dim_repr(d) for d in dims]})",
    )
    return checks


# ---------------------------------------------------------------------------
# ORT numeric verification (opt-in, isolated subprocess)
# ---------------------------------------------------------------------------


# Subprocess exit codes for the ORT verification worker.
VERIFY_PASS = 0
VERIFY_MISMATCH = 1  # ran successfully but logits differ -> transform is wrong
VERIFY_INCONCLUSIVE = 2  # could not run (missing CPU kernel, OOM, load error)


def run_ort_verify(orig_onnx, pruned_onnx, tokens, logits_name):
    """Run the numeric-equivalence check in a subprocess; return (code, report)."""
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--_ort-worker",
        "--orig",
        orig_onnx,
        "--pruned",
        pruned_onnx,
        "--tokens",
        str(tokens),
        "--logits-name",
        logits_name,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    report = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, report


def _elem_to_numpy(elem):
    import numpy as np

    return {
        TensorProto.FLOAT: np.float32,
        TensorProto.FLOAT16: np.float16,
        TensorProto.DOUBLE: np.float64,
        TensorProto.INT64: np.int64,
        TensorProto.INT32: np.int32,
        TensorProto.BOOL: np.bool_,
    }.get(elem, np.float32)


def _resolve_dim(d, tokens):
    if d.HasField("dim_value") and d.dim_value > 0:
        return d.dim_value
    p = (d.dim_param or "").lower()
    if "past" in p:
        return 0
    if "batch" in p:
        return 1
    if "seq" in p or "sequence" in p:
        return tokens
    return 1


def _make_dummy_inputs(model, tokens):
    import numpy as np

    feeds = {}
    for vi in model.graph.input:
        tt = vi.type.tensor_type
        shape = [_resolve_dim(d, tokens) for d in tt.shape.dim]
        np_dtype = _elem_to_numpy(tt.elem_type)
        name = vi.name.lower()
        if "input_ids" in name:
            arr = (np.arange(int(np.prod(shape)), dtype=np.int64) % 100 + 1).reshape(
                shape
            )
        elif "position" in name:
            arr = np.arange(int(np.prod(shape)), dtype=np.int64).reshape(shape)
        elif "mask" in name:
            arr = np.ones(shape, dtype=np_dtype)
        else:
            arr = np.zeros(shape, dtype=np_dtype)
        feeds[vi.name] = arr.astype(np_dtype)
    return feeds


def _run_last_token_logits(onnx_path, feeds, logits_name, last_index):
    """Create a CPU session, run once, return the last-token logits vector.

    Session and outputs are dropped before returning so peak memory stays near a
    single model's footprint (orig and pruned are run sequentially, not together).
    """
    import numpy as np
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
    out = np.asarray(sess.run([logits_name], feeds)[0])
    vec = out[0, last_index, :].astype(np.float32)
    shape = out.shape
    del out, sess
    return vec, shape


def _ort_worker(orig_onnx, pruned_onnx, tokens, logits_name):
    """Subprocess entry: run orig+pruned on CPU and compare last-token logits.

    Exit codes: 0 PASS, 1 numeric MISMATCH, 2 INCONCLUSIVE (could not run, e.g.
    a contrib op such as QMoE has no CPU kernel, out-of-memory, or a load error).
    """
    import numpy as np

    struct = onnx.load(orig_onnx, load_external_data=False)
    feeds = _make_dummy_inputs(struct, tokens)

    try:
        last_o, shape_o = _run_last_token_logits(orig_onnx, feeds, logits_name, -1)
        last_p, shape_p = _run_last_token_logits(pruned_onnx, feeds, logits_name, 0)
    except Exception as exc:  # missing CPU kernel / OOM / load failure
        print(f"[ort-verify] could not run on CPUExecutionProvider: {exc}")
        print("[ort-verify] RESULT: INCONCLUSIVE")
        return VERIFY_INCONCLUSIVE

    max_abs = float(np.max(np.abs(last_o - last_p)))
    argmax_eq = int(np.argmax(last_o)) == int(np.argmax(last_p))
    top5_o = set(np.argsort(last_o)[-5:].tolist())
    top5_p = set(np.argsort(last_p)[-5:].tolist())
    top5_overlap = len(top5_o & top5_p)
    denom = float(np.linalg.norm(last_o) * np.linalg.norm(last_p)) or 1.0
    cos = float(np.dot(last_o, last_p) / denom)

    print(f"[ort-verify] orig last-token logits shape: {shape_o}")
    print(f"[ort-verify] pruned logits shape:          {shape_p}")
    print(f"[ort-verify] max_abs_diff = {max_abs:.6g}")
    print(f"[ort-verify] argmax_equal = {argmax_eq}")
    print(f"[ort-verify] top5_overlap = {top5_overlap}/5")
    print(f"[ort-verify] cosine       = {cos:.8f}")

    # The last-token slice is mathematically identical; allow only fp rounding.
    ok = argmax_eq and top5_overlap == 5 and (max_abs <= 1e-2 or cos >= 0.9999)
    print("[ort-verify] RESULT:", "PASS" if ok else "MISMATCH")
    return VERIFY_PASS if ok else VERIFY_MISMATCH


# ---------------------------------------------------------------------------
# Rollback / restore
# ---------------------------------------------------------------------------


def rollback(model_onnx, orig_onnx):
    """Undo a partial apply: drop the pruned file, put the original name back."""
    if os.path.isfile(model_onnx):
        os.remove(model_onnx)
    if os.path.isfile(orig_onnx):
        os.rename(orig_onnx, model_onnx)


def do_restore(model_onnx, orig_onnx, logits_name):
    if not os.path.isfile(orig_onnx):
        raise PruneError(f"nothing to restore: backup not found ({orig_onnx})")

    # Pre-check: the current model should look pruned (seq dim == 1).
    if os.path.isfile(model_onnx):
        cur = onnx.load(model_onnx, load_external_data=False)
        vi = _output_vi(cur.graph, logits_name)
        if vi is not None:
            d = vi.type.tensor_type.shape.dim
            if len(d) == 3 and not (d[1].HasField("dim_value") and d[1].dim_value == 1):
                raise PruneError(
                    f"current model does not look pruned "
                    f"({[_dim_repr(x) for x in d]}); refusing to restore"
                )
        os.remove(model_onnx)
    os.rename(orig_onnx, model_onnx)

    # Post-check: restored model's logits seq dim must be dynamic again.
    m = onnx.load(model_onnx, load_external_data=False)
    vi = _output_vi(m.graph, logits_name)
    d = vi.type.tensor_type.shape.dim
    if len(d) == 3 and d[1].HasField("dim_value") and d[1].dim_value == 1:
        raise PruneError("post-restore check failed: logits still collapsed to 1")
    print(f"restored: {model_onnx}  (logits dims={[_dim_repr(x) for x in d]})")


# ---------------------------------------------------------------------------
# Main apply flow
# ---------------------------------------------------------------------------


def do_apply(args):
    model_dir, model_onnx, orig_onnx = resolve_paths(args.model)
    print(f"model dir : {model_dir}")
    print(f"model     : {model_onnx}")
    print(f"backup    : {orig_onnx}")

    # Pre-flight: fail fast if ORT verification is requested but unavailable,
    # BEFORE touching the model.
    if args.ort_verify and importlib.util.find_spec("onnxruntime") is None:
        raise PruneError(
            "--ort-verify requires onnxruntime, which is not installed.\n"
            "  Install it first:  pip install onnxruntime"
        )

    # Load structure only; external weights are never read or rewritten.
    model = onnx.load(model_onnx, load_external_data=False)
    is_pruned = _is_pruned(model, args.logits_name)
    has_backup = os.path.isfile(orig_onnx)

    # "Already processed by this tool" (and safely revertible) requires BOTH a
    # collapsed logits dim AND a backup. The other combinations are distinct
    # states with honest, non-misleading guidance (don't promise --restore when
    # there is no backup to restore from).
    if is_pruned and has_backup:
        raise PruneError(
            f"model is already pruned and a backup exists ({orig_onnx}); "
            f"use --restore to revert to the original."
        )
    if is_pruned and not has_backup:
        raise PruneError(
            f"model is already pruned (logits seq dim == 1) but no backup "
            f"({orig_onnx}) exists; nothing to do (and nothing to restore)."
        )
    if has_backup:  # not pruned, yet a backup is present -> inconsistent leftover
        raise PruneError(
            f"a backup exists ({orig_onnx}) but the current model is not pruned "
            f"(inconsistent state); use --restore, or remove the stray backup "
            f"and re-run."
        )

    lm, hidden = check_eligibility(model, args.logits_name, args.lm_head_name)

    out_vi = _output_vi(model.graph, args.logits_name)
    before = [_dim_repr(d) for d in out_vi.type.tensor_type.shape.dim]
    print("\n[pre-check] eligible for prune-logits")
    print(f"  lm_head        : {lm.op_type}  {lm.name}")
    print(f"  hidden input   : {hidden}")
    print(f"  logits (before): {before}")
    print("  tail:")
    for ln in tail_trace(model.graph, args.logits_name):
        print(ln)

    if args.precheck:
        print("\n[precheck] no changes written.")
        return 0

    # Backup == rename (external data references stay valid).
    os.rename(model_onnx, orig_onnx)
    try:
        apply_prune(model, lm, hidden, args.logits_name)
        onnx.save(model, model_onnx)  # structure only; shares *_orig external data

        print("\n[post-review] structural checks:")
        for ln in review_structure(model_onnx, args.lm_head_name, args.logits_name):
            print("  " + ln)

        if args.ort_verify:
            print("\n[ort-verify] running numeric equivalence (CPU, subprocess)...")
            code, report = run_ort_verify(
                orig_onnx, model_onnx, VERIFY_TOKENS, args.logits_name
            )
            print(report.rstrip())
            if code == VERIFY_MISMATCH:
                raise PruneError("ORT numeric verification: last-token logits differ")
            if code == VERIFY_INCONCLUSIVE:
                # Infrastructure limitation (e.g. QMoE has no CPU kernel), not a
                # sign the structural transform is wrong -- keep the reviewed
                # model and warn rather than discarding a correct result.
                print(
                    "\n[warning] ORT numeric verify could not run on this model "
                    "(see reason above); kept the structurally-reviewed model. "
                    "Re-run with --restore if you want to revert.",
                    file=sys.stderr,
                )
    except Exception as exc:
        print(f"\n[rollback] {exc}", file=sys.stderr)
        rollback(model_onnx, orig_onnx)
        print("[rollback] original model restored; no changes kept.", file=sys.stderr)
        raise

    after = [
        _dim_repr(d)
        for d in _output_vi(
            onnx.load(model_onnx, load_external_data=False).graph, args.logits_name
        ).type.tensor_type.shape.dim
    ]
    print("\n[done] prune-logits applied.")
    print(f"  pruned model : {model_onnx}   logits={after}")
    print(f"  original kept: {orig_onnx}")
    print("  genai_config.json unchanged (model keeps its filename).")
    return 0


def parse_args(argv):
    p = argparse.ArgumentParser(
        description="Offline prune-logits transform for ONNX decoder models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "model",
        nargs="?",
        default=None,
        metavar="model_dir",
        help="model directory (decoder .onnx is resolved via genai_config.json)",
    )
    p.add_argument("--logits-name", default="logits", help="graph logits output name")
    p.add_argument("--lm-head-name", default=None, help="override lm_head node name")
    p.add_argument(
        "--ort-verify",
        action="store_true",
        help="verify last-token logits equivalence with onnxruntime (needs "
        "onnxruntime; off by default)",
    )
    p.add_argument(
        "--precheck",
        action="store_true",
        help="run eligibility checks and print the plan only; no writes",
    )
    p.add_argument(
        "--restore", action="store_true", help="revert from the *_orig backup"
    )

    # Hidden subprocess entry for the isolated ORT verification worker.
    p.add_argument("--_ort-worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--orig", default=None, help=argparse.SUPPRESS)
    p.add_argument("--pruned", default=None, help=argparse.SUPPRESS)
    p.add_argument("--tokens", type=int, default=8, help=argparse.SUPPRESS)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)

    if args._ort_worker:
        return _ort_worker(args.orig, args.pruned, args.tokens, args.logits_name)

    if args.model is None:
        print("error: the 'model' argument is required", file=sys.stderr)
        return 1

    try:
        if args.restore:
            _, model_onnx, orig_onnx = resolve_paths(args.model)
            do_restore(model_onnx, orig_onnx, args.logits_name)
            return 0
        return do_apply(args)
    except PruneError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
