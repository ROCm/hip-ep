#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Convert a pre-folded Mistral-Small-4 OGA model to its latent-cache (MLA) form.

Three independent graph rewrites of decoder/model.onnx, applied in this order:

1. Residual scaling (--residual-scale, default 8). The BOS row's layer-32 MLP
   output (~1.5e5) overflows fp16, and the resulting inf becomes NaN logits
   for every prompt that starts with BOS. RMSNorm is scale invariant, so the
   residual stream is carried divided by a power of two s: inputs_embeds is
   multiplied by 1/s, every residual contribution is scaled by 1/s through
   its weight scales (o_proj, shared-expert up_proj, the up rows of the QMoE
   fc1 scales), and every skip-norm epsilon becomes eps / s^2. All touched
   scales stay fp16-normal, so the rescale is exact.

2. int8 lm_head (--lm-head-bits, default 8). The fp16 lm_head MatMul becomes
   a symmetric RTN MatMulNBits (block 128), halving the largest decode GEMV.

3. Static latent KV cache (on unless --no-mla). Every GroupQueryAttention
   layer is rewritten so the cache holds the 320-wide latent row
   [kv_a_layernorm(c) (256), roped k_pe (64)] instead of 32 heads of expanded
   K/V. The cache is a fixed [B, 1, max_length, 320] buffer shared between
   past and present (genai past_present_share_buffer):
     nonpad      = ReduceSum(attention_mask)
     present.key = TensorScatter(past.key, latent, nonpad - S, axis 2)
     K, V        = expand(present.key) with the k_nope / v rows of kv_b_proj
     out         = Attention(q, K, V, nonpad_kv_seqlen = nonpad), causal
   hip-ep's hip-fusion-transform recognizes this expansion and runs it as one
   hip.mla_attention op that decodes over the latent cache directly. genai
   pairs every key cache with a value cache of the same shape, so the value
   cache is a copy of the key cache that the graph never reads.

--no-mla keeps GroupQueryAttention, which gives the matching baseline (same
residual scaling and lm_head) for A/B comparisons.

The destination symlinks every top-level entry except decoder/ and the
genai_config*.json files, hard-links the decoder's external data (ORT rejects
external data that resolves outside the model directory) and writes the new
tensors to decoder/res.data, decoder/lmhead_q<bits>.data and
decoder/mla_latent.data.

  convert_mistral_mla.py <src model dir> <dst model dir>
      [--residual-scale 8] [--lm-head-bits 8] [--no-mla] [--chunk-size N]
      [--copy-data]
"""

import argparse
import errno
import json
import os
import shutil
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

# Mistral-Small-4 attention geometry.
H = 32  # query heads
D_NOPE = 64  # qk_nope_head_dim
D_ROPE = 64  # qk_rope_head_dim
D_V = 128  # v_head_dim
D_LAT = 256  # kv_lora_rank
D_QK = D_LAT + D_ROPE  # latent cache row
N_LAYERS = 36

F16 = TensorProto.FLOAT16
LM_HEAD_BLOCK = 128


def fail(msg):
    sys.exit(f"error: {msg}")


def ext_info(t):
    return {e.key: e.value for e in t.external_data}


def set_external(t, location, offset, length):
    del t.external_data[:]
    t.data_location = TensorProto.EXTERNAL
    for k, v in (
        ("location", location),
        ("offset", str(offset)),
        ("length", str(length)),
    ):
        e = t.external_data.add()
        e.key, e.value = k, v


def external_tensor(name, dims, dtype, location, offset, length):
    t = TensorProto(name=name, dims=dims, data_type=dtype)
    set_external(t, location, offset, length)
    return t


def new_data_file(path):
    # The decoder's data files are hard links into the source model: replace
    # the directory entry instead of truncating a shared inode.
    if os.path.lexists(path):
        os.unlink(path)
    return open(path, "wb")


def read_tensor(t, ddir):
    dtype = {TensorProto.UINT8: np.uint8, TensorProto.FLOAT16: np.float16}[t.data_type]
    ext = ext_info(t)
    if not ext:
        return numpy_helper.to_array(t)
    return np.fromfile(
        os.path.join(ddir, ext["location"]),
        dtype=dtype,
        count=int(np.prod(t.dims)),
        offset=int(ext["offset"]),
    ).reshape(t.dims)


# ---------------------------------------------------------------------------
# Destination layout
# ---------------------------------------------------------------------------


def link_or_copy(src, dst, copy):
    if copy:
        shutil.copyfile(src, dst)
        return
    try:
        os.link(src, dst)
    except OSError as e:
        if e.errno == errno.EXDEV:
            fail(
                f"cannot hard-link {src} into {os.path.dirname(dst)}: different "
                "filesystems. Put the destination on the source's filesystem, or "
                "pass --copy-data to copy the decoder data (~100 GB)."
            )
        raise


def prepare_dst(src, dst, copy):
    if os.path.exists(os.path.join(dst, "decoder", "model.onnx")):
        fail(f"{dst} already holds a converted model; choose a new directory")
    os.makedirs(os.path.join(dst, "decoder"), exist_ok=True)
    for name in os.listdir(src):
        if name == "decoder" or name.startswith("genai_config"):
            continue
        if not os.path.lexists(os.path.join(dst, name)):
            os.symlink(
                os.path.realpath(os.path.join(src, name)), os.path.join(dst, name)
            )
    sdec = os.path.join(src, "decoder")
    for name in os.listdir(sdec):
        if name != "model.onnx" and not os.path.exists(
            os.path.join(dst, "decoder", name)
        ):
            link_or_copy(
                os.path.realpath(os.path.join(sdec, name)),
                os.path.join(dst, "decoder", name),
                copy,
            )


def write_genai_configs(src, dst, mla, chunk_size):
    names = [
        n
        for n in os.listdir(src)
        if n.startswith("genai_config") and n.endswith(".json")
    ]
    if "genai_config.json" not in names:
        fail(f"{src} has no genai_config.json")
    for name in names:
        with open(os.path.join(src, name)) as f:
            cfg = json.load(f)
        if mla:
            cfg["model"]["decoder"]["num_key_value_heads"] = 1
            cfg["model"]["decoder"]["head_size"] = D_QK
            cfg["search"]["past_present_share_buffer"] = True
        if chunk_size:
            cfg["search"]["chunk_size"] = chunk_size
        with open(os.path.join(dst, name), "w") as f:
            json.dump(cfg, f, indent=4)


# ---------------------------------------------------------------------------
# 1. Residual scaling
# ---------------------------------------------------------------------------


def scale_residual(g, sdec, ddir, s):
    init = {t.name: t for t in g.initializer}
    cons = {}
    for n in g.node:
        for i in n.input:
            cons.setdefault(i, []).append(n)

    norms = [n for n in g.node if n.op_type == "SkipSimplifiedLayerNormalization"]
    if len(norms) != 2 * N_LAYERS or norms[0].input[0] != "inputs_embeds":
        fail(
            f"expected {2 * N_LAYERS} skip norms with inputs_embeds first; is this "
            "the Mistral-Small-4 decoder, and not already residual-scaled?"
        )

    out = new_data_file(os.path.join(ddir, "res.data"))

    def rewrite(name, fn):
        t = init.get(name)
        if t is None:
            fail(
                f"{name} is not an initializer. The QMoE fc1 scales must be "
                "pre-folded (offline Reshape/Transpose folding of the QMoE "
                "weights); this tool expects the pre-folded model."
            )
        if len(cons[name]) != 1:
            fail(f"{name} is shared by several nodes")
        if t.data_type != F16:
            fail(f"{name} is not fp16")
        ext = ext_info(t)
        v = np.memmap(
            os.path.join(sdec, ext["location"]),
            dtype=np.float16,
            mode="r",
            offset=int(ext["offset"]),
            shape=tuple(t.dims),
        )
        v = fn(np.asarray(v, dtype=np.float32))
        q = v.astype(np.float16)
        if not np.array_equal(q.astype(np.float32), v):
            fail(f"{name}: rescale by 1/{s:g} is not exact in fp16")
        off = out.tell()
        pad = (-off) % 4096
        out.write(b"\0" * pad)
        off += pad
        out.write(q.tobytes())
        set_external(t, "res.data", off, q.nbytes)

    def up_rows(v):
        v = v.copy()
        v[:, 1::2] /= s  # swiglu_fusion=1 interleaves gate/up rows
        return v

    n_scaled = 0
    for n in g.node:
        if n.op_type == "QMoE":
            rewrite(n.input[3], up_rows)
            n_scaled += 1
        elif n.op_type == "MatMulNBits" and (
            "/o_proj/" in n.name or "/shared_experts/up_proj/" in n.name
        ):
            rewrite(n.input[2], lambda v: v / s)
            n_scaled += 1
    out.close()
    if n_scaled != 3 * N_LAYERS:
        fail(f"rescaled {n_scaled} scale tensors, expected {3 * N_LAYERS}")

    for n in norms:
        for at in n.attribute:
            if at.name == "epsilon":
                at.f = at.f / (s * s)
    g.initializer.append(
        numpy_helper.from_array(np.array(1.0 / s, np.float16), "res16/inv_scale")
    )
    mul = helper.make_node(
        "Mul",
        ["inputs_embeds", "res16/inv_scale"],
        ["inputs_embeds__res"],
        name="res16/embeds_scale",
    )
    norms[0].input[0] = "inputs_embeds__res"
    nodes = list(g.node)
    nodes.insert(nodes.index(norms[0]), mul)
    del g.node[:]
    g.node.extend(nodes)
    print(
        f"residual scale 1/{s:g}: {n_scaled} scale tensors rescaled, "
        f"{len(norms)} norm epsilons divided by {s * s:g}"
    )


# ---------------------------------------------------------------------------
# 2. Quantized lm_head
# ---------------------------------------------------------------------------


def quantize_lm_head(g, sdec, ddir, bits):
    node = next(
        (n for n in g.node if n.op_type == "MatMul" and n.output[0] == "logits"), None
    )
    if node is None:
        fail("no fp16 MatMul produces logits; is the lm_head already quantized?")
    w_init = next(t for t in g.initializer if t.name == node.input[1])
    K, N = w_init.dims
    ext = ext_info(w_init)
    w = np.memmap(
        os.path.join(sdec, ext["location"]),
        dtype=np.float16,
        mode="r",
        offset=int(ext["offset"]),
        shape=(K, N),
    )
    nb = K // LM_HEAD_BLOCK
    blob = LM_HEAD_BLOCK * bits // 8
    q_all = np.empty((N, nb, blob), dtype=np.uint8)
    s_all = np.empty((N, nb), dtype=np.float16)
    err2 = ref2 = 0.0
    for n0 in range(0, N, 8192):
        wt = np.asarray(w[:, n0 : n0 + 8192], dtype=np.float32).T.reshape(
            -1, nb, LM_HEAD_BLOCK
        )
        if bits == 4:
            # MatMulNBits' default zero point 8: dequant = (q - 8) * scale. The
            # signed extreme maps to -8, so the full [-8, 7] range is used.
            idx = np.abs(wt).argmax(axis=2)[..., None]
            vmax = np.take_along_axis(wt, idx, axis=2)
            scale = (vmax / -8.0).astype(np.float16).astype(np.float32)
            zp = 8
        else:
            scale = (
                (np.abs(wt).max(axis=2, keepdims=True) / 127.0)
                .astype(np.float16)
                .astype(np.float32)
            )
            zp = 128
        safe = np.where(scale == 0, 1.0, scale)
        q = np.clip(np.rint(wt / safe) + zp, 0, 2**bits - 1).astype(np.uint8)
        deq = (q.astype(np.float32) - zp) * scale
        err2 += float(((deq - wt) ** 2).sum())
        ref2 += float((wt**2).sum())
        if bits == 4:
            q = q[..., 0::2] | (q[..., 1::2] << 4)
        q_all[n0 : n0 + 8192] = q
        s_all[n0 : n0 + 8192] = scale[..., 0].astype(np.float16)

    data = f"lmhead_q{bits}.data"
    with new_data_file(os.path.join(ddir, data)) as f:
        f.write(q_all.tobytes())
        s_off = f.tell()
        f.write(s_all.tobytes())
    qname, sname = "decoder.lm_head.qweight", "decoder.lm_head.scales"
    g.initializer.remove(w_init)
    g.initializer.extend(
        [
            external_tensor(
                qname, [N, nb, blob], TensorProto.UINT8, data, 0, q_all.nbytes
            ),
            external_tensor(sname, [N, nb], F16, data, s_off, s_all.nbytes),
        ]
    )
    new = helper.make_node(
        "MatMulNBits",
        [node.input[0], qname, sname],
        list(node.output),
        name="decoder/lm_head/MatMulNBits",
        domain="com.microsoft",
        K=K,
        N=N,
        bits=bits,
        block_size=LM_HEAD_BLOCK,
        accuracy_level=4,
    )
    pos = list(g.node).index(node)
    g.node.remove(node)
    g.node.insert(pos, new)
    print(
        f"lm_head [{K}x{N}] int{bits} block {LM_HEAD_BLOCK}: relative weight RMS error "
        f"{np.sqrt(err2 / ref2):.4%}"
    )


# ---------------------------------------------------------------------------
# 3. Static latent (MLA) cache
# ---------------------------------------------------------------------------


def split_kv_b(node, init, sdec):
    """kv_b_proj MatMulNBits -> ((packed, scales) of the k_nope rows, same of
    the v rows). Rows are head-major [h][64 k_nope | 128 v] and blocks run
    along K, so selecting rows keeps every block intact."""
    attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
    if attrs["bits"] != 4 or len(node.input) != 3:
        fail(f"{node.name}: expected int4 kv_b_proj without zero points")
    if attrs["N"] != H * (D_NOPE + D_V) or attrs["K"] != D_LAT:
        fail(f"{node.name}: unexpected kv_b_proj shape N={attrs['N']} K={attrs['K']}")
    q = read_tensor(init[node.input[1]], sdec)
    s = read_tensor(init[node.input[2]], sdec)
    n = H * (D_NOPE + D_V)
    rows = np.arange(n).reshape(H, D_NOPE + D_V)
    s2 = s.reshape(n, -1)
    out = []
    for sel in (rows[:, :D_NOPE].reshape(-1), rows[:, D_NOPE:].reshape(-1)):
        ss = (
            s2[sel].reshape((len(sel),) + s.shape[1:])
            if s.ndim > 1
            else s2[sel].reshape(-1)
        )
        out.append((np.ascontiguousarray(q[sel]), np.ascontiguousarray(ss)))
    return out, attrs


def mla_consts():
    c = numpy_helper.from_array
    return [
        c(np.array(1, np.int64), "mla/one"),
        c(np.array([1], np.int64), "mla/axis1"),
        c(np.array([2], np.int64), "mla/axis2"),
        c(np.array([D_LAT, D_ROPE], np.int64), "mla/split"),
        c(np.array([0, 0, H, D_NOPE], np.int64), "mla/shape_kn"),
        c(np.array([0, 0, 1, D_ROPE], np.int64), "mla/shape_kr"),
        c(np.array([1, 1, H, 1], np.int64), "mla/heads"),
        c(np.array([0, 0, -1], np.int64), "mla/shape_flat"),
        c(np.array(0.0, np.float32), "mla/zero_f"),
        c(np.array(-65504.0, np.float32), "mla/neg_f"),
    ]


def mla_shared_nodes(mask, x):
    """Valid rows and write position of the static cache: the mask holds ones
    for the attended prefix ([B, max_length] under a shared buffer)."""
    p = "mla/"

    def mk(op, ins, out, **kw):
        return helper.make_node(op, ins, [p + out], name=p + out, **kw)

    return [
        mk("ReduceSum", [mask, p + "axis1"], "nonpad", keepdims=0),
        mk("Shape", [x], "S", start=1, end=2),
        mk("Sub", [p + "nonpad", p + "S"], "write_idx"),
    ]


def mla_layer_nodes(
    p, c_norm, k_pe, q, past_k, past_v, pres_k, pres_v, out, w, nbits, scale
):
    def mk(op, ins, o, **kw):
        return helper.make_node(op, ins, [p + o], name=p + o, **kw)

    cache = [
        helper.make_node(
            "TensorScatter",
            [past, p + "latent4", "mla/write_idx"],
            [pres],
            name=p + name,
            axis=2,
        )
        for past, pres, name in (
            (past_k, pres_k, "cache_k"),
            (past_v, pres_v, "cache_v"),
        )
    ]
    attn = helper.make_node(
        "Attention",
        [q, p + "k", p + "v", "", "", "", "mla/nonpad"],
        [out],
        name=p + "attn",
        q_num_heads=H,
        kv_num_heads=H,
        scale=scale,
        is_causal=1,
    )

    def mm(w_, s_, o, n_):
        return helper.make_node(
            "MatMulNBits",
            [p + "c", w_, s_],
            [p + o],
            name=p + o,
            domain="com.microsoft",
            K=D_LAT,
            N=n_,
            bits=4,
            block_size=nbits["block_size"],
            accuracy_level=nbits.get("accuracy_level", 4),
        )

    return [
        mk("Concat", [c_norm, k_pe], "latent", axis=-1),
        mk("Unsqueeze", [p + "latent", "mla/axis1"], "latent4"),
        *cache,
        mk("Squeeze", [pres_k, "mla/axis1"], "lat"),
        helper.make_node(
            "Split",
            [p + "lat", "mla/split"],
            [p + "c", p + "k_r"],
            name=p + "split",
            axis=-1,
        ),
        mm(w["kb"], w["kb_s"], "k_nope", H * D_NOPE),
        mk("Reshape", [p + "k_nope", "mla/shape_kn"], "k_nope4"),
        mk("Reshape", [p + "k_r", "mla/shape_kr"], "k_r4"),
        mk("Expand", [p + "k_r4", "mla/heads"], "k_rh"),
        mk("Concat", [p + "k_nope4", p + "k_rh"], "k4", axis=-1),
        mk("Reshape", [p + "k4", "mla/shape_flat"], "k"),
        mm(w["vb"], w["vb_s"], "v", H * D_V),
        attn,
    ]


def to_latent_cache(g, sdec, ddir):
    init = {t.name: t for t in g.initializer}
    prod = {o: n for n in g.node for o in n.output}

    def up(t, op):
        n = prod.get(t)
        if n is None or n.op_type != op:
            fail(
                f"unexpected attention graph: {t} is produced by "
                f"{n.op_type if n else 'nothing'}, expected {op}"
            )
        return n

    x_in = next(i.name for i in g.input if i.name in ("inputs_embeds", "input_ids"))
    consts = mla_consts()
    blobs, off = [], 0
    remove, inserts = set(), {}
    gqas = [n for n in g.node if n.op_type == "GroupQueryAttention"]
    if len(gqas) != N_LAYERS:
        fail(
            f"found {len(gqas)} GroupQueryAttention layers, expected {N_LAYERS}; "
            "is the model already converted?"
        )
    for li, gqa in enumerate(gqas):
        p = f"mla/l{li}/"
        up(gqa.input[0], "Mul")  # q * temperature, kept
        r49 = up(gqa.input[1], "Reshape")
        c47 = up(r49.input[0], "Concat")
        s39 = up(c47.input[0], "Split")
        e44 = up(c47.input[1], "Expand")
        r43 = up(e44.input[0], "Reshape")
        k_pe = r43.input[0]  # roped [B, S, 64]
        r38 = up(s39.input[0], "Reshape")
        kvb = up(r38.input[0], "MatMulNBits")
        c_norm = kvb.input[0]  # [B, S, 256]
        r40 = up(gqa.input[2], "Reshape")
        if (
            r40.input[0] != s39.output[1]
            or numpy_helper.to_array(init[r38.input[1]]).tolist()
            != [0, 0, H, D_NOPE + D_V]
            or numpy_helper.to_array(init[s39.input[1]]).tolist() != [D_NOPE, D_V]
        ):
            fail(f"layer {li}: kv_b_proj split does not match Mistral-Small-4")
        up(k_pe, "RotaryEmbedding")
        up(c_norm, "RMSNormalization")
        attrs = {a.name: helper.get_attribute_value(a) for a in gqa.attribute}
        if attrs.get("do_rotary", 0) != 0 or attrs["num_heads"] != H:
            fail(f"layer {li}: unexpected GroupQueryAttention attributes {attrs}")

        parts, nbits = split_kv_b(kvb, init, sdec)
        names = {}
        for key, arr in zip(
            ("kb", "kb_s", "vb", "vb_s"), (a for pair in parts for a in pair)
        ):
            dt = TensorProto.UINT8 if arr.dtype == np.uint8 else F16
            consts.append(
                external_tensor(
                    p + key, arr.shape, dt, "mla_latent.data", off, arr.nbytes
                )
            )
            blobs.append(arr.tobytes())
            off += arr.nbytes
            pad = (-off) % 64
            blobs.append(b"\0" * pad)
            off += pad
            names[key] = p + key

        new = mla_layer_nodes(
            p,
            c_norm,
            k_pe,
            gqa.input[0],
            gqa.input[3],
            gqa.input[4],
            gqa.output[1],
            gqa.output[2],
            gqa.output[0],
            names,
            nbits,
            attrs["scale"],
        )
        for n in (r49, c47, s39, e44, r43, r38, kvb, r40, gqa):
            remove.add(id(n))
        inserts[id(gqa)] = (
            mla_shared_nodes("attention_mask", x_in) if li == 0 else []
        ) + new

    nodes = []
    for n in g.node:
        if id(n) in inserts:
            nodes.extend(inserts[id(n)])
        elif id(n) not in remove:
            nodes.append(n)
    outs = {o.name for o in g.output}
    while True:  # the GQA sequence-length inputs are dead now
        used = {i for n in nodes for i in n.input} | outs
        live = [n for n in nodes if any(o in used for o in n.output)]
        if len(live) == len(nodes):
            break
        nodes = live
    del g.node[:]
    g.node.extend(nodes)

    used = {i for n in g.node for i in n.input}
    keep = [t for t in g.initializer if t.name in used]
    del g.initializer[:]
    g.initializer.extend(keep + consts)

    past_seq = None
    for vi in list(g.input) + list(g.output):
        if vi.name.startswith(("past_key_values.", "present.")):
            d = vi.type.tensor_type.shape.dim
            d[1].dim_value = 1
            d[3].dim_value = D_QK
            # present is past written in place: same buffer, same length.
            if vi.name.startswith("past_key_values."):
                past_seq = past_seq or d[2].dim_param
            else:
                d[2].dim_param = past_seq
    del g.value_info[:]

    with new_data_file(os.path.join(ddir, "mla_latent.data")) as f:
        for b in blobs:
            f.write(b)
    print(
        f"latent cache: rewrote {len(gqas)} attention layers, mla_latent.data "
        f"{off / 2**20:.0f} MiB"
    )


def precheck(g, scale, lm_head_bits, mla):
    """Reject an unsuitable source before anything is written."""
    init = {t.name for t in g.initializer}
    norms = [n for n in g.node if n.op_type == "SkipSimplifiedLayerNormalization"]
    qmoe = [n for n in g.node if n.op_type == "QMoE"]
    if len(norms) != 2 * N_LAYERS or len(qmoe) != N_LAYERS:
        fail(
            f"expected {N_LAYERS} decoder layers (QMoE + 2 skip norms each); is "
            "this the Mistral-Small-4 decoder?"
        )
    if scale > 1:
        if norms[0].input[0] != "inputs_embeds":
            fail("the residual stream is already scaled; pass --residual-scale 1")
        if any(n.input[3] not in init for n in qmoe):
            fail(
                "the QMoE fc1 scales are not initializers. This tool expects the "
                "pre-folded model (QMoE weight Reshape/Transpose chains folded "
                "offline)."
            )
    if lm_head_bits and not any(
        n.op_type == "MatMul" and n.output[0] == "logits" for n in g.node
    ):
        fail(
            "no fp16 MatMul produces logits; the lm_head is already quantized "
            "(pass --lm-head-bits 0)"
        )
    if mla and sum(n.op_type == "GroupQueryAttention" for n in g.node) != N_LAYERS:
        fail(
            f"expected {N_LAYERS} GroupQueryAttention layers; is the model already "
            "converted?"
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("src", help="pre-folded Mistral-Small-4 OGA model directory")
    ap.add_argument("dst", help="new model directory (same filesystem as src)")
    ap.add_argument(
        "--residual-scale",
        type=float,
        default=8.0,
        help="power of two the residual stream is divided by; 1 disables",
    )
    ap.add_argument(
        "--lm-head-bits",
        type=int,
        default=8,
        choices=(0, 4, 8),
        help="lm_head quantization; 0 keeps the fp16 MatMul",
    )
    ap.add_argument(
        "--no-mla",
        action="store_true",
        help="keep GroupQueryAttention (baseline with the same other rewrites)",
    )
    ap.add_argument(
        "--chunk-size",
        type=int,
        default=0,
        help="override search.chunk_size (prefill chunk) in genai_config",
    )
    ap.add_argument(
        "--copy-data",
        action="store_true",
        help="copy the decoder's external data instead of hard-linking it",
    )
    a = ap.parse_args()
    s = a.residual_scale
    if s < 1 or not float(np.log2(s)).is_integer():
        fail("--residual-scale must be a power of two >= 1")

    src, dst = os.path.abspath(a.src), os.path.abspath(a.dst)
    sdec = os.path.join(src, "decoder")
    if src == dst:
        fail("src and dst must differ")
    m = onnx.load(os.path.join(sdec, "model.onnx"), load_external_data=False)
    g = m.graph
    precheck(g, s, a.lm_head_bits, not a.no_mla)
    prepare_dst(src, dst, a.copy_data)
    ddir = os.path.join(dst, "decoder")
    if s > 1:
        scale_residual(g, sdec, ddir, s)
    if a.lm_head_bits:
        quantize_lm_head(g, sdec, ddir, a.lm_head_bits)
    if not a.no_mla:
        to_latent_cache(g, sdec, ddir)
    onnx.save(m, os.path.join(ddir, "model.onnx"))
    write_genai_configs(src, dst, not a.no_mla, a.chunk_size)
    print(f"wrote {dst} ({len(g.node)} decoder nodes)")


if __name__ == "__main__":
    main()
