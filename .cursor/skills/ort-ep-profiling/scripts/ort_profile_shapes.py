#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Shape detail formatters aligned with hip-ep OP_PROFILE strings in lib/Runtime/real/."""

from __future__ import annotations

from typing import Callable


def _tensor(inp: list[dict], idx: int = 0) -> tuple[str, list[int]]:
    spec = inp[idx]
    dtype = next(iter(spec))
    return dtype, list(spec[dtype])


def _short_dtype(dtype: str) -> str:
    return {
        "float16": "f16",
        "half": "f16",
        "bfloat16": "bf16",
        "float": "f32",
        "double": "f64",
        "int64": "i64",
        "int32": "i32",
        "int8": "i8",
        "uint8": "u8",
        "bool": "bool",
    }.get(dtype, dtype)


def _numel(shape: list[int]) -> int:
    n = 1
    for dim in shape:
        n *= dim
    return n


def _pad4(shape: list[int]) -> list[int]:
    dims = list(shape)
    while len(dims) < 4:
        dims.insert(0, 1)
    return dims[-4:]


def _output_tensor(node_args: dict, idx: int = 0) -> tuple[str, list[int]] | None:
    outputs = node_args.get("output_type_shape")
    if not outputs or idx >= len(outputs):
        return None
    return _tensor(outputs, idx)


def default_shape(node_args: dict) -> str:
    dtype, shape = _tensor(node_args["input_type_shape"], 0)
    return f"{'x'.join(map(str, shape))},{_short_dtype(dtype)}"


def conv_shape(node_args: dict) -> str:
    """Conv: NxCxHxW,k=KxKhxKw,dt (matches wrap_miopenConvolutionForward)."""
    dtype, x = _tensor(node_args["input_type_shape"], 0)
    _, w = _tensor(node_args["input_type_shape"], 1)
    dt = _short_dtype(dtype)
    if len(w) == 4:
        out_c, kh, kw = w[0], w[2], w[3]
    elif len(w) == 3:
        out_c, kh, kw = w[0], w[1], w[2]
    else:
        return default_shape(node_args)
    if len(x) == 4:
        n, c, h, wi = x
    elif len(x) == 3:
        n, c, h, wi = 1, x[0], x[1], x[2]
    else:
        return default_shape(node_args)
    return f"{n}x{c}x{h}x{wi},k={out_c}x{kh}x{kw},{dt}"


def conv_transpose_shape(node_args: dict) -> str:
    dtype, x = _tensor(node_args["input_type_shape"], 0)
    _, w = _tensor(node_args["input_type_shape"], 1)
    dt = _short_dtype(dtype)
    if len(x) != 4 or len(w) < 3:
        return default_shape(node_args)
    n, c, h, wi = x
    kh, kw = (w[2], w[3]) if len(w) == 4 else (w[1], w[2])
    out_c = w[0]
    stride = 1
    attrs = node_args.get("attributes") or {}
    if "strides" in attrs:
        strides = attrs["strides"]
        stride = strides[0] if isinstance(strides, list) and strides else strides
    return f"{n}x{c}x{h}x{wi},m={out_c},k={kh}x{kw},s={stride},{dt}"


def gemm_shape(node_args: dict) -> str:
    _, a = _tensor(node_args["input_type_shape"], 0)
    _, b = _tensor(node_args["input_type_shape"], 1)

    def mat_dims(shape: list[int]) -> tuple[int, int]:
        if len(shape) == 1:
            return 1, shape[0]
        return shape[-2], shape[-1]

    m, k = mat_dims(a)
    k_b, n = mat_dims(b)
    if k_b not in (k, 0):
        k = k_b
    return f"m={m},n={n},k={k}"


def elementwise_shape(node_args: dict) -> str:
    outputs = _output_tensor(node_args, 0)
    if outputs:
        _, shape = outputs
    else:
        _, shape = _tensor(node_args["input_type_shape"], 0)
    n, c, h, w = _pad4(shape)
    return f"{n}x{c}x{h}x{w}"


def comparison_shape(node_args: dict) -> str:
    outputs = _output_tensor(node_args, 0)
    if outputs:
        dtype, shape = outputs
    else:
        dtype, shape = _tensor(node_args["input_type_shape"], 0)
    n, c, h, w = _pad4(shape)
    return f"{n}x{c}x{h}x{w}:{_short_dtype(dtype)}"


def unary_numel_shape(node_args: dict) -> str:
    dtype, shape = _tensor(node_args["input_type_shape"], 0)
    return f"{_numel(shape)}:{_short_dtype(dtype)}"


def activation_n_shape(node_args: dict) -> str:
    _, shape = _tensor(node_args["input_type_shape"], 0)
    return f"n={_numel(shape)}"


def gelu_shape(node_args: dict) -> str:
    _, shape = _tensor(node_args["input_type_shape"], 0)
    return f"n={_numel(shape)}"


def bias_gelu_shape(node_args: dict) -> str:
    _, x = _tensor(node_args["input_type_shape"], 0)
    n = _numel(x)
    if len(node_args["input_type_shape"]) > 1:
        _, bias = _tensor(node_args["input_type_shape"], 1)
        return f"n={n},bias={_numel(bias)}"
    return f"n={n},bias=0"


def softmax_shape(node_args: dict) -> str:
    _, shape = _tensor(node_args["input_type_shape"], 0)
    if len(shape) >= 2:
        rows, cols = shape[-2], shape[-1]
        return f"{rows}x{cols}"
    return f"1x{_numel(shape)}"


def layernorm_shape(node_args: dict) -> str:
    inp = node_args["input_type_shape"]
    dtype, x = _tensor(inp, 0)
    scale = _tensor(inp, 1)[1] if len(inp) > 1 else []
    hidden = scale[-1] if scale else (x[-1] if x else 0)
    outer = _numel(x) // hidden if hidden else 0
    dt = "f16" if "16" in dtype else "f32"
    suffix = ":b" if len(inp) > 2 else ""
    return f"{outer}x{hidden}:{dt}{suffix}"


def skip_layernorm_shape(node_args: dict) -> str:
    return layernorm_shape(node_args)


def reduce_shape(node_args: dict) -> str:
    _, x = _tensor(node_args["input_type_shape"], 0)
    in_n = _numel(x)
    out = _output_tensor(node_args, 0)
    out_n = _numel(out[1]) if out else 0
    return f"{in_n}->{out_n}"


def reduce_labeled_shape(label: str) -> Callable[[dict], str]:
    def fmt(node_args: dict) -> str:
        detail = reduce_shape(node_args)
        attrs = node_args.get("attributes") or {}
        keep = attrs.get("keepdims")
        suffix = ":keep" if keep in (1, True) else ""
        return f"{detail}{suffix}"

    _ = label
    return fmt


def cast_shape(node_args: dict) -> str:
    _, shape = _tensor(node_args["input_type_shape"], 0)
    return f"n={_numel(shape)}"


def pool_shape(node_args: dict) -> str:
    _, x = _tensor(node_args["input_type_shape"], 0)
    n, c, in0, in1 = _pad4(x)
    out = _output_tensor(node_args, 0)
    if out:
        _, y = out
        _, _, out0, out1 = _pad4(y)
    else:
        out0, out1 = in0, in1
    spatial_rank = max(len(x) - 2, 1)
    return f"rank={spatial_rank},N={n},C={c},in=[{in0},{in1},1],out=[{out0},{out1},1]"


def global_pool_shape(node_args: dict) -> str:
    _, x = _tensor(node_args["input_type_shape"], 0)
    if len(x) <= 1:
        return default_shape(node_args)
    outer = x[0]
    reduce_size = _numel(x[1:])
    p = (node_args.get("attributes") or {}).get("p", 2)
    return f"outer={outer},reduce={reduce_size},p={p}"


def gqa_shape(node_args: dict) -> str:
    inp = node_args["input_type_shape"]
    _, q = _tensor(inp, 0)
    if len(q) == 3:
        b, sq, hidden = q
        h = (node_args.get("attributes") or {}).get("num_heads", 0)
        d = hidden // h if h else 0
        skv = (
            _tensor(inp, 1)[1][-2]
            if len(inp) > 1 and len(_tensor(inp, 1)[1]) >= 2
            else sq
        )
        return f"b={b},sq={sq},skv={skv},h={h},d={d}"
    return default_shape(node_args)


def multi_head_attention_shape(node_args: dict) -> str:
    inp = node_args["input_type_shape"]
    _, q = _tensor(inp, 0)
    attrs = node_args.get("attributes") or {}
    unidirectional = attrs.get("unidirectional", 0)
    causal = ",causal" if unidirectional in (1, True) else ""
    if len(q) == 3:
        b, sq, hidden = q
        n = attrs.get("num_heads", 0)
        h = hidden // n if n else 0
        skv = sq
        if len(inp) > 3:
            _, k = _tensor(inp, 3)
            if len(k) >= 2:
                skv = k[-2]
        return f"b={b},sq={sq},skv={skv},n={n},h={h}{causal}"
    return default_shape(node_args)


def rotary_emb_shape(node_args: dict) -> str:
    attrs = node_args.get("attributes") or {}
    inp = node_args["input_type_shape"]
    _, x = _tensor(inp, 0)
    num_heads = attrs.get("num_heads", 0)
    head_dim = attrs.get("head_size", x[-1] if x else 0)
    rotary_dim = attrs.get("rotary_embedding_dim", head_dim)
    is_bnsh = 1 if len(x) == 4 else 0
    return f"h={num_heads},hd={head_dim},rd={rotary_dim},bnsh={is_bnsh}"


def gather_shape(node_args: dict) -> str:
    _, x = _tensor(node_args["input_type_shape"], 0)
    return f"n={_numel(x)}"


def gather_elements_shape(node_args: dict) -> str:
    _, x = _tensor(node_args["input_type_shape"], 0)
    return f"r{len(x)}:n={_numel(x)}"


def gather_nd_shape(node_args: dict) -> str:
    inp = node_args["input_type_shape"]
    _, data = _tensor(inp, 0)
    idx_rank = len(_tensor(inp, 1)[1]) if len(inp) > 1 else 0
    batch = len(data) if data else 0
    dtype = _short_dtype(_tensor(inp, 0)[0])
    return f"dr{len(data)}:ir{idx_rank}:bd={batch}:{dtype}"


def slice_shape(node_args: dict) -> str:
    inp = node_args["input_type_shape"]
    _, x = _tensor(inp, 0)
    starts_rank = len(_tensor(inp, 1)[1]) if len(inp) > 1 else 0
    dtype = _short_dtype(_tensor(inp, 0)[0])
    return f"r{len(x)}:K{starts_rank}:{dtype}"


def transpose_shape(node_args: dict) -> str:
    _, x = _tensor(node_args["input_type_shape"], 0)
    return f"n={_numel(x)} rank={len(x)}"


def pad_shape(node_args: dict) -> str:
    _, x = _tensor(node_args["input_type_shape"], 0)
    pads = node_args["input_type_shape"]
    pad_count = len(pads[1]) if len(pads) > 1 else 0
    dtype = _short_dtype(_tensor(node_args["input_type_shape"], 0)[0])
    return f"r{len(x)}:{dtype}:pads={pad_count}"


def resize_shape(node_args: dict) -> str:
    _, x = _tensor(node_args["input_type_shape"], 0)
    out = _output_tensor(node_args, 0)
    if len(x) == 4:
        _, _, in0, in1 = x
        spatial_rank = 2
        in2 = 1
    else:
        spatial_rank = max(len(x), 1)
        in0, in1, in2 = (x + [1, 1, 1])[:3]
    if out:
        _, y = out
        if len(y) == 4:
            _, _, out0, out1 = y
            out2 = 1
        else:
            out0, out1, out2 = (y + [1, 1, 1])[:3]
    else:
        out0, out1, out2 = in0, in1, in2
    mode = (node_args.get("attributes") or {}).get("mode", 0)
    return f"rank={spatial_rank},in=[{in0},{in1},{in2}],out=[{out0},{out1},{out2}],mode={mode}"


def tile_shape(node_args: dict) -> str:
    _, x = _tensor(node_args["input_type_shape"], 0)
    reps = (
        _tensor(node_args["input_type_shape"], 1)[1]
        if len(node_args["input_type_shape"]) > 1
        else []
    )
    return f"r{len(x)}:{'x'.join(map(str, reps)) if reps else '1'}"


def expand_shape(node_args: dict) -> str:
    _, x = _tensor(node_args["input_type_shape"], 0)
    out = _output_tensor(node_args, 0)
    in_n = _numel(x)
    out_n = _numel(out[1]) if out else in_n
    dtype = _short_dtype(_tensor(node_args["input_type_shape"], 0)[0])
    return f"{in_n}->{out_n}:{dtype}"


def cumsum_shape(node_args: dict) -> str:
    _, x = _tensor(node_args["input_type_shape"], 0)
    axis = (node_args.get("attributes") or {}).get("axis", 0)
    exclusive = (node_args.get("attributes") or {}).get("exclusive", 0)
    reverse = (node_args.get("attributes") or {}).get("reverse", 0)
    return f"r{len(x)}:{axis}:{'ex' if exclusive else ''}{'rev' if reverse else ''}"


def top_k_shape(node_args: dict) -> str:
    _, x = _tensor(node_args["input_type_shape"], 0)
    axis = (node_args.get("attributes") or {}).get("axis", -1)
    return f"r{len(x)}:axis={axis}"


def one_hot_shape(node_args: dict) -> str:
    inp = node_args["input_type_shape"]
    _, indices = _tensor(inp, 0)
    axis = (node_args.get("attributes") or {}).get("axis", -1)
    return f"axis={axis}:n={_numel(indices)}"


def scatter_elements_shape(node_args: dict) -> str:
    _, data = _tensor(node_args["input_type_shape"], 0)
    axis = (node_args.get("attributes") or {}).get("axis", 0)
    dtype = _short_dtype(_tensor(node_args["input_type_shape"], 0)[0])
    return f"axis={axis}:{dtype}:n={_numel(data)}"


def scatter_nd_shape(node_args: dict) -> str:
    inp = node_args["input_type_shape"]
    _, data = _tensor(inp, 0)
    idx_rank = len(_tensor(inp, 1)[1]) if len(inp) > 1 else 0
    dtype = _short_dtype(_tensor(inp, 0)[0])
    updates = _short_dtype(_tensor(inp, 2)[0]) if len(inp) > 2 else dtype
    return f"dr{len(data)}:ir{idx_rank}:{dtype}:{updates}"


def compress_shape(node_args: dict) -> str:
    inp = node_args["input_type_shape"]
    _, data = _tensor(inp, 0)
    axis = (node_args.get("attributes") or {}).get("axis", 0)
    return f"flat={_numel(data)}:axis={axis}:n={len(data)}"


def nonzero_shape(node_args: dict) -> str:
    _, x = _tensor(node_args["input_type_shape"], 0)
    dtype = _short_dtype(_tensor(node_args["input_type_shape"], 0)[0])
    return f"numel={_numel(x)}:rank={len(x)}:{dtype}"


def power_shape(node_args: dict) -> str:
    _, x = _tensor(node_args["input_type_shape"], 0)
    return f"n={_numel(x)}"


def mod_shape(node_args: dict) -> str:
    dtype, shape = _tensor(node_args["input_type_shape"], 0)
    fmod = (node_args.get("attributes") or {}).get("fmod", 0)
    return f"{_numel(shape)}:{_short_dtype(dtype)}{':fmod' if fmod else ''}"


def qmoe_shape(node_args: dict) -> str:
    inp = node_args["input_type_shape"]
    _, x = _tensor(inp, 0)
    experts = _tensor(inp, 1)[1][0] if len(inp) > 1 and _tensor(inp, 1)[1] else 0
    if len(x) == 3:
        return f"{x[0]}x{x[1]}x{x[2]},e={experts}"
    return default_shape(node_args)


def linear_attention_shape(node_args: dict) -> str:
    inp = node_args["input_type_shape"]
    _, q = _tensor(inp, 0)
    attrs = node_args.get("attributes") or {}
    if len(q) == 4:
        b, sq, hq, dk = q[0], q[1], q[2], q[3]
        hkv = attrs.get("kv_num_heads", hq)
        dv = attrs.get("v_head_size", dk)
        return f"b={b},sq={sq},hq={hq},hkv={hkv},dk={dk},dv={dv}"
    return default_shape(node_args)


def causal_conv_shape(node_args: dict) -> str:
    dtype, x = _tensor(node_args["input_type_shape"], 0)
    _, w = _tensor(node_args["input_type_shape"], 1)
    dt = _short_dtype(dtype)
    if len(x) >= 2 and len(w) >= 1:
        return f"{x[0]}x{x[1]}x{len(x) if len(x) > 2 else 1},k={w[-1]},{dt}"
    return default_shape(node_args)


def memcpy_2d_shape(node_args: dict) -> str:
    _, x = _tensor(node_args["input_type_shape"], 0)
    if len(x) >= 2:
        return f"w={x[-1]} h={x[-2]}"
    return default_shape(node_args)


def concat_shape(node_args: dict) -> str:
    inp = node_args["input_type_shape"]
    if not inp:
        return ""
    _, first = _tensor(inp, 0)
    axis = (node_args.get("attributes") or {}).get("axis", 0)
    return f"r{len(first)}:axis={axis}:inputs={len(inp)}"


# ONNX / ORT op_name -> formatter (mirrors lib/Runtime/real/ OP_PROFILE labels)
SHAPE_FNS: dict[str, Callable[[dict], str]] = {
    # conv
    "Conv": conv_shape,
    "DmlFusedConv": conv_shape,
    "ConvTranspose": conv_transpose_shape,
    # gemm / matmul
    "Gemm": gemm_shape,
    "MatMul": gemm_shape,
    "MatMulNBits": gemm_shape,
    "FusedMatMul": gemm_shape,
    "DmlFusedGemm": gemm_shape,
    "QLinearMatMul": gemm_shape,
    # pool
    "MaxPool": pool_shape,
    "AveragePool": pool_shape,
    "LpPool": pool_shape,
    "GlobalMaxPool": global_pool_shape,
    "GlobalAveragePool": global_pool_shape,
    "GlobalLpPool": global_pool_shape,
    # elementwise binary
    "Add": elementwise_shape,
    "Sub": elementwise_shape,
    "Mul": elementwise_shape,
    "Div": elementwise_shape,
    "And": elementwise_shape,
    "Or": elementwise_shape,
    "Xor": elementwise_shape,
    "Min": elementwise_shape,
    "Max": elementwise_shape,
    "BitwiseAnd": elementwise_shape,
    "BitwiseOr": elementwise_shape,
    "BitwiseXor": elementwise_shape,
    "DmlFusedAdd": elementwise_shape,
    "DmlFusedMul": elementwise_shape,
    "Prelu": elementwise_shape,
    # comparisons
    "Equal": comparison_shape,
    "Less": comparison_shape,
    "Greater": comparison_shape,
    "LessOrEqual": comparison_shape,
    "GreaterOrEqual": comparison_shape,
    # unary
    "Abs": unary_numel_shape,
    "Neg": unary_numel_shape,
    "Sign": unary_numel_shape,
    "Sin": unary_numel_shape,
    "Cos": unary_numel_shape,
    "Exp": unary_numel_shape,
    "Log": unary_numel_shape,
    "Ceil": unary_numel_shape,
    "Floor": unary_numel_shape,
    "Not": unary_numel_shape,
    "IsNaN": unary_numel_shape,
    "Reciprocal": unary_numel_shape,
    "Sqrt": unary_numel_shape,
    "Erf": unary_numel_shape,
    # activations
    "Relu": activation_n_shape,
    "Sigmoid": activation_n_shape,
    "Tanh": activation_n_shape,
    "Softplus": activation_n_shape,
    "Softsign": activation_n_shape,
    "HardSigmoid": activation_n_shape,
    "LeakyRelu": activation_n_shape,
    "Elu": activation_n_shape,
    "Selu": activation_n_shape,
    "ThresholdedRelu": activation_n_shape,
    "Gelu": gelu_shape,
    "FastGelu": bias_gelu_shape,
    "BiasGelu": bias_gelu_shape,
    "QuickGelu": gelu_shape,
    # softmax
    "Softmax": softmax_shape,
    "LogSoftmax": softmax_shape,
    # norm
    "LayerNormalization": layernorm_shape,
    "SimplifiedLayerNormalization": skip_layernorm_shape,
    "SkipLayerNormalization": skip_layernorm_shape,
    "SkipSimplifiedLayerNormalization": skip_layernorm_shape,
    # reduce
    "ReduceSum": reduce_shape,
    "ReduceMean": reduce_shape,
    "ReduceMax": reduce_labeled_shape("reduce_max"),
    "ReduceMin": reduce_labeled_shape("reduce_min"),
    "ReduceProd": reduce_labeled_shape("reduce_prod"),
    "ReduceL2": reduce_shape,
    "ArgMax": reduce_shape,
    "ArgMin": reduce_shape,
    # attention
    "GroupQueryAttention": gqa_shape,
    "MultiHeadAttention": multi_head_attention_shape,
    "RotaryEmbedding": rotary_emb_shape,
    "LinearAttention": linear_attention_shape,
    # data movement / layout
    "Cast": cast_shape,
    "Gather": gather_shape,
    "GatherElements": gather_elements_shape,
    "GatherND": gather_nd_shape,
    "Slice": slice_shape,
    "Transpose": transpose_shape,
    "Pad": pad_shape,
    "Resize": resize_shape,
    "Tile": tile_shape,
    "Expand": expand_shape,
    "CumSum": cumsum_shape,
    "TopK": top_k_shape,
    "OneHot": one_hot_shape,
    "ScatterElements": scatter_elements_shape,
    "ScatterND": scatter_nd_shape,
    "Compress": compress_shape,
    "NonZero": nonzero_shape,
    "Pow": power_shape,
    "Mod": mod_shape,
    "Concat": concat_shape,
    "Qmoe": qmoe_shape,
    "CausalConv": causal_conv_shape,
}


def shape_detail(node_args: dict) -> str:
    op_type = node_args["op_name"]
    shape_fn = SHAPE_FNS.get(op_type, default_shape)
    return shape_fn(node_args)
