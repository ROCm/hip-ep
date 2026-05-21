//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

#ifndef HIP_OPS
#define HIP_OPS

include "hip/Dialect/IR/HipDialect.td"
include "hip/Dialect/IR/HipTypes.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"

class Hip_Op<string mnemonic, list<Trait> traits = []> :
    Op<Hip_Dialect, mnemonic, traits>;

// Base class for all HIP destination-passing-style (DPS) compute ops.
// Provides MemoryEffectsOpInterface, DestinationStyleOpInterface, and the
// Variadic<AnyRankedTensor>:$result_tensors result (non-empty only at the
// tensor level; empty after bufferization).
class Hip_DpsOp<string mnemonic, list<Trait> traits = []> :
    Hip_Op<mnemonic, !listconcat([
      DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
      DeclareOpInterfaceMethods<DestinationStyleOpInterface,
                                ["getDpsInitsMutable"]>
    ], traits)> {
  let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
}

// Accepts both ranked tensor (pre-bufferization) and memref (post-bufferization).
def Hip_TensorOrMemRef : AnyTypeOf<[AnyRankedTensor, AnyMemRef],
                                    "ranked tensor or memref">;

def Hip_AllocOp : Hip_Op<"alloc",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Allocate device memory";
  let description = [{
    Allocates device memory using HIP and returns a memref.

    Example:
    ```mlir
    %x = hip.alloc(%ctx, %N) : memref<?x128xf32, 1>
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Variadic<Index>:$dynamicSizes
  );
  let results = (outs AnyMemRef:$memref);

  let assemblyFormat = [{
    `(` $ctx (`,` $dynamicSizes^ )? `)` attr-dict `:` type($memref)
  }];
}

def Hip_GetPoolOp : Hip_Op<"get_pool",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Get the runtime-owned GPU memory pool (grow-on-demand)";
  let description = [{
    Returns a reference to the runtime-managed GPU memory pool.  The runtime
    guarantees that the returned buffer is at least %pool_size bytes.  If the
    current pool is smaller, the runtime grows it (realloc) transparently.

    Example:
    ```mlir
    %pool = hip.get_pool(%ctx, %size) : memref<?xi8>
    ```
  }];

  let arguments = (ins Hip_ContextType:$ctx, Index:$pool_size);
  let results = (outs AnyMemRef:$pool);
  let assemblyFormat = "`(` $ctx `,` $pool_size `)` attr-dict `:` type($pool)";
}

// ===== Control flow ==========================================================

def Hip_LoopOp : Hip_Op<"loop",
    [AttrSizedOperandSegments,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     DeclareOpInterfaceMethods<DestinationStyleOpInterface,
                               ["getDpsInitsMutable"]>,
     DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Outlined-body counted/conditional loop";
  let description = [{
    Runtime-driven loop with an outlined body function. Each iteration
    invokes `$body_func` with the iteration index, the current condition,
    the loop-carried values, and the captured outer-graph values. The
    body updates the loop-carried values and returns a new condition.

    Operand layout:
      - `$ctx`: `!hip.context` (always first).
      - `$max_trip_count`: `index` — the trip count `M` (ONNX Loop input 0).
      - `$cond_init`: optional `i1` — initial condition (ONNX Loop input 1).
      - `$v_init`: variadic, length must equal `$num_loop_carried` — initial
        values for loop-carried tensors/memrefs; types must match the op's
        result types.
      - `$captures`: variadic — outer-graph values used inside the body
        that are not loop-carried (e.g. constant tensors A, B).

    Result types match `$v_init` types and correspond to the loop-carried
    values after the final iteration.

    If `$cond_is_passthrough` is set, the outlining pass has verified that
    the body's yielded `cond_out` is SSA-equal to its `cond_in` block
    argument. The LLVM lowering then picks the fast-path runtime function
    `hipdnn_ep_run_counted_loop` (no per-iter device->host cond sync).
    Otherwise the slow-path `hipdnn_ep_run_loop` is used (matches ORT CUDA
    EP / MIGraphX behavior).

    Example:
    ```mlir
    %v_final = hip.loop(%ctx, %M, %cond_init)
                   iter_args(%acc_init : tensor<16x16xf32>)
                   captures(%A, %B : tensor<?x16x16xf32>,
                                     tensor<16x16xf32>)
                   -> (tensor<16x16xf32>)
                   body @loop_body_n0
                   {num_loop_carried = 1 : i32, cond_is_passthrough}
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Index:$max_trip_count,
    Optional<I1>:$cond_init,
    Variadic<Hip_TensorOrMemRef>:$v_init,
    Variadic<AnyType>:$captures,
    FlatSymbolRefAttr:$body_func,
    I32Attr:$num_loop_carried,
    UnitAttr:$cond_is_passthrough
  );

  // v_final is non-empty only at the tensor level (DPS convention).
  // After BufferResultsToOutParams / OneShotBufferize, the results are
  // eliminated and v_init carries the writes via its memref descriptors.
  let results = (outs Variadic<AnyRankedTensor>:$v_final);

  let assemblyFormat = [{
    `(` $ctx `,` $max_trip_count (`,` $cond_init^)? `)`
    ( `iter_args` `(` $v_init^ `:` type($v_init) `)` )?
    ( `captures` `(` $captures^ `:` type($captures) `)` )?
    ( `->` `(` type($v_final)^ `)` )?
    `body` $body_func
    attr-dict
  }];

  let hasVerifier = 1;
}

def Hip_FreeOp : Hip_Op<"free",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Free device memory";
  let description = [{
    Frees device memory allocated with hip.alloc.

    Example:
    ```mlir
    hip.free(%ctx, %x) : memref<?x128xf32, 1>
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    AnyMemRef:$memref
  );

  let assemblyFormat = [{
    `(` $ctx `,` $memref `)` attr-dict `:` type($memref)
  }];
}

// ===== Constant Management ===================================================

def Hip_GetConstantOp : Hip_Op<"get_constant", [
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Retrieve constant from GPU memory";
  let description = [{
    Retrieves a pre-uploaded constant from state->gpu_constants[index].

    Arguments:
    - ctx: HIP execution context (contains gpu_constants array)
    - index: Index in gpu_constants array (0, 1, 2, ...)

    Returns:
    - memref: GPU memory reference to the constant

    Example:
    ```mlir
    %weight = hip.get_constant(%ctx, %index_0) : memref<64x3x3x3xf32, 1>
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    I64:$index
  );
  let results = (outs AnyMemRef:$result);

  let assemblyFormat = [{
    `(` $ctx `,` $index `)` attr-dict `:` type($result)
  }];
}

// ===== Region ops (structural grouping, no runtime) =========================

def Hip_MiopenGraphOp : Hip_Op<"miopen.graph", [SingleBlock, NoTerminator]> {
  let summary = "MIOpen backend dispatch region";
  let description = [{
    Groups operations that should be dispatched to the MIOpen library.
    Purely structural -- inlined and erased during lowering.
  }];
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$body attr-dict";
}

def Hip_HipblasltGraphOp : Hip_Op<"hipblaslt.graph", [SingleBlock, NoTerminator]> {
  let summary = "hipBLASLt backend dispatch region";
  let description = [{
    Groups operations that should be dispatched to the hipBLASLt library.
    Purely structural -- inlined and erased during lowering.
  }];
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$body attr-dict";
}

// ===== MIOpen DNN Operations =================================================

def Hip_ConvOp : Hip_DpsOp<"conv"> {
  let summary = "2D convolution operation using MIOpen (in-place)";
  let description = [{
    Performs 2D convolution using MIOpen (miopenConvolutionForward).

    Uses destination-passing style: output buffer is provided as argument,
    operation writes result in-place.

    Attributes:
    - kernel_shape: Array of kernel dimensions [H, W]
    - strides: Array of stride values [H, W]
    - pads: Array of padding values [top, left, bottom, right]
    - dilations: Array of dilation values [H, W]
    - group: Number of groups for grouped convolution

    Example:
    ```mlir
    hip.conv(%ctx) ins(%input, %weights, %bias :
                       memref<1x3x224x224xf32, 1>,
                       memref<64x3x3x3xf32, 1>,
                       memref<64xf32, 1>)
                   outs(%output : memref<1x64x112x112xf32, 1>)
                   {kernel_shape = [3, 3], strides = [1, 1],
                    pads = [1, 1, 1, 1], dilations = [1, 1], group = 1}
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$input,
    Hip_TensorOrMemRef:$weights,
    Optional<Hip_TensorOrMemRef>:$bias,
    Hip_TensorOrMemRef:$output,
    I64ArrayAttr:$kernel_shape,
    I64ArrayAttr:$strides,
    I64ArrayAttr:$pads,
    I64ArrayAttr:$dilations,
    I64Attr:$group
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $input `,` $weights (`,` $bias^)? `:`
    type($input) `,` type($weights) (`,` type($bias)^)? `)`
    `outs` `(` $output `:` type($output) `)`
    attr-dict (`:` type($result_tensors)^ )?
  }];
}

// ===== hipBLASLt ops =========================================================

def Hip_MatmulOp : Hip_DpsOp<"matmul"> {
  let summary = "Matrix multiplication (supports batched N-D x N-D)";
  let description = [{
    Performs matrix multiplication: output = A @ B
    Supports batched operation (leading dimensions are batch dimensions).

    Uses destination-passing style: output buffer is provided as argument.

    Unlike hip.gemm, this has no alpha/beta/transA/transB/bias — it maps
    directly to onnx.MatMul semantics.

    Example (2D):
    ```mlir
    hip.matmul(%ctx) ins(%A, %B : memref<128x4096xf32, 1>,
                                  memref<4096x1024xf32, 1>)
                     outs(%output : memref<128x1024xf32, 1>)
    ```

    Example (3D batched):
    ```mlir
    hip.matmul(%ctx) ins(%A, %B : memref<1x128x4096xf16, 1>,
                                  memref<4096x1024xf16, 1>)
                     outs(%output : memref<1x128x1024xf16, 1>)
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$A,
    Hip_TensorOrMemRef:$B,
    Hip_TensorOrMemRef:$output
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $A `,` $B `:` type($A) `,` type($B) `)`
    `outs` `(` $output `:` type($output) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

//===----------------------------------------------------------------------===//
// Quantized Operations
//===----------------------------------------------------------------------===//

def Hip_MatMulNBitsOp : Hip_DpsOp<"matmul_nbits", [AttrSizedOperandSegments]> {
  let summary = "Quantized N-bit matrix multiplication (com.microsoft MatMulNBits)";
  let description = [{
    Performs matrix multiplication with N-bit quantized weights.
    Dequantizes packed int4/int8 weights using per-block scale factors,
    then computes Y = A @ dequantized_weight^T.

    Implements the com.microsoft MatMulNBits operator (inputs 3-6).

    Weight packing: B is stored as uint8 with shape (N, k_blocks, blob_size)
    where k_blocks = ceil(K / block_size), blob_size = block_size * bits / 8.
    For 4-bit: lower nibble = first value, upper nibble = second value.

    Dequantization: dequantized = (quantized - zero_point) * scale
    Default zero_point = 2^(bits-1) (= 8 for 4-bit).

    Optional inputs per spec:
      zero_points: per-block zero points for dequantization
      g_idx: GPTQ group ordering indices (deprecated)
      bias: output bias vector [N]

    Uses destination-passing style: output buffer is provided as argument.

    Example:
    ```mlir
    hip.matmul_nbits(%handle) ins(%A, %B, %scales :
        memref<1x128x2880xf16, 1>, memref<5120x90x16xui8, 1>,
        memref<5120x90xf16, 1>)
        outs(%output : memref<1x128x5120xf16, 1>)
        {K = 2880, N = 5120, bits = 4, block_size = 32, accuracy_level = 4}
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$handle,
    Hip_TensorOrMemRef:$A,                      // activation [..., K]
    Hip_TensorOrMemRef:$B,                      // packed uint8 [N, k_blocks, blob_size]
    Hip_TensorOrMemRef:$scales,                 // dequant scales [N, k_blocks]
    Optional<Hip_TensorOrMemRef>:$zero_points,  // dequant zero points (optional)
    Optional<Hip_TensorOrMemRef>:$g_idx,        // GPTQ group indices (optional, deprecated)
    Optional<Hip_TensorOrMemRef>:$bias,         // output bias [N] (optional)
    Hip_TensorOrMemRef:$output,                 // result [..., N]
    I64Attr:$K,
    I64Attr:$N,
    I64Attr:$bits,
    I64Attr:$block_size,
    I64Attr:$accuracy_level,
    I64Attr:$zp_elem_size  // zero_points element size: 1=uint8 packed, 2=fp16
  );

  let assemblyFormat = [{
    `(` $handle `)` `ins` `(` $A `,` $B `,` $scales `:`
    type($A) `,` type($B) `,` type($scales) `)`
    (`zero_points` `(` $zero_points^ `:` type($zero_points) `)`)?
    (`g_idx` `(` $g_idx^ `:` type($g_idx) `)`)?
    (`bias` `(` $bias^ `:` type($bias) `)`)?
    `outs` `(` $output `:` type($output) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_QMoEOp : Hip_DpsOp<"qmoe", [AttrSizedOperandSegments]> {
  let summary = "Quantized Mixture-of-Experts (com.microsoft QMoE)";
  let description = [{
    Implements the com.microsoft QMoE operator (ONNX v1.25+ compatible): quantized
    Mixture-of-Experts with top-k expert routing and optional SwiGLU activation.

    Routes each token to its top-k experts, performs quantized linear
    projections (fc1 gate+up, fc2 down) per expert with int4 dequantization,
    applies activation (SwiGLU by default), and combines expert outputs.

    SwiGLU formula (when swiglu_fusion=1, interleaved in fc1 output):
      g, l = interleaved_split(fc1_out)
      G = clamp(g, max=swiglu_limit)
      L = clamp(l, -swiglu_limit, swiglu_limit)
      result = G * sigmoid(activation_alpha * G) * (L + activation_beta)

    Weight packing follows MatMulNBits convention: uint8 packed,
    dequantized = (quantized - 8) * scale for 4-bit weights.

    Optional inputs per spec:
      fc1_experts_bias, fc2_experts_bias: bias vectors (optional)
      fc3_*: unfused SwiGLU second GEMM weights/scales/bias (optional)
      fc1/fc2/fc3_zero_points: per-block zero points (optional)
      router_weights: alternative aggregation weights (optional, ONNX v1.25+, takes precedence over router_probs)

    Uses destination-passing style: output buffer is provided as argument.

    Example:
    ```mlir
    hip.qmoe(%handle) ins(%input, %router,
                           %fc1w, %fc1s, %fc2w, %fc2s :
        memref<1x128x2880xf16, 1>, memref<128x32xf16, 1>,
        memref<32x5760x1440xui8, 1>, memref<32x5760x90xf16, 1>,
        memref<32x2880x1440xui8, 1>, memref<32x2880x90xf16, 1>)
        fc1_bias(%fc1b : memref<32x5760xf16, 1>)
        fc2_bias(%fc2b : memref<32x2880xf16, 1>)
        outs(%output : memref<1x128x2880xf16, 1>)
        {expert_weight_bits = 4, k = 4, block_size = 32, ...}
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$handle,
    Hip_TensorOrMemRef:$input,                  // [batch, seq, hidden]
    Hip_TensorOrMemRef:$router_probs,           // [num_tokens, num_experts]
    Hip_TensorOrMemRef:$fc1_experts_weights,    // [num_experts, fusion*inter, hidden/pack]
    Hip_TensorOrMemRef:$fc1_scales,             // [num_experts, fusion*inter, hidden/bs]
    Hip_TensorOrMemRef:$fc2_experts_weights,    // [num_experts, hidden, inter/pack]
    Hip_TensorOrMemRef:$fc2_scales,             // [num_experts, hidden, inter/bs]
    Optional<Hip_TensorOrMemRef>:$fc1_experts_bias,       // [num_experts, fusion*inter]
    Optional<Hip_TensorOrMemRef>:$fc2_experts_bias,       // [num_experts, hidden]
    Optional<Hip_TensorOrMemRef>:$fc3_experts_weights,    // [num_experts, inter, hidden/pack]
    Optional<Hip_TensorOrMemRef>:$fc3_scales,             // [num_experts, inter, hidden/bs]
    Optional<Hip_TensorOrMemRef>:$fc3_experts_bias,       // [num_experts, inter]
    Optional<Hip_TensorOrMemRef>:$fc1_zero_points,        // fc1 dequant zero points
    Optional<Hip_TensorOrMemRef>:$fc2_zero_points,        // fc2 dequant zero points
    Optional<Hip_TensorOrMemRef>:$fc3_zero_points,        // fc3 dequant zero points
    Optional<Hip_TensorOrMemRef>:$router_weights,         // [num_tokens, num_experts] alternative routing weights (ONNX v1.25+)
    Hip_TensorOrMemRef:$output,                 // DPS init [batch, seq, hidden]
    I64Attr:$expert_weight_bits,
    I64Attr:$k,
    I64Attr:$block_size,
    I64Attr:$normalize_routing_weights,
    I64Attr:$swiglu_fusion,
    I64Attr:$use_sparse_mixer,
    F32Attr:$activation_alpha,
    F32Attr:$activation_beta,
    F32Attr:$swiglu_limit,
    StrAttr:$activation_type
  );

  let assemblyFormat = [{
    `(` $handle `)` `ins` `(`
        $input `,` $router_probs `,`
        $fc1_experts_weights `,` $fc1_scales `,`
        $fc2_experts_weights `,` $fc2_scales `:`
        type($input) `,` type($router_probs) `,`
        type($fc1_experts_weights) `,` type($fc1_scales) `,`
        type($fc2_experts_weights) `,` type($fc2_scales)
    `)`
    (`fc1_bias` `(` $fc1_experts_bias^ `:` type($fc1_experts_bias) `)`)?
    (`fc2_bias` `(` $fc2_experts_bias^ `:` type($fc2_experts_bias) `)`)?
    (`fc3_weights` `(` $fc3_experts_weights^ `:` type($fc3_experts_weights) `)`)?
    (`fc3_scales` `(` $fc3_scales^ `:` type($fc3_scales) `)`)?
    (`fc3_bias` `(` $fc3_experts_bias^ `:` type($fc3_experts_bias) `)`)?
    (`fc1_zero_points` `(` $fc1_zero_points^ `:` type($fc1_zero_points) `)`)?
    (`fc2_zero_points` `(` $fc2_zero_points^ `:` type($fc2_zero_points) `)`)?
    (`fc3_zero_points` `(` $fc3_zero_points^ `:` type($fc3_zero_points) `)`)?
    (`router_weights` `(` $router_weights^ `:` type($router_weights) `)`)?
    `outs` `(` $output `:` type($output) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

// ===== MIOpen ops ============================================================

def Hip_RmsNormOp : Hip_DpsOp<"rms_norm"> {
  let summary = "RMS normalization via miopenT5LayerNormForward";
  let description = [{
    Applies simplified (RMS) layer normalization.

    Formula: output = input / sqrt(mean(input^2) + epsilon) * scale

    Uses destination-passing style: output buffer is provided as argument.

    Example:
    ```mlir
    hip.rms_norm(%ctx)
        ins(%input, %scale : memref<1x128x4096xf16, 1>, memref<4096xf16, 1>)
        outs(%output : memref<1x128x4096xf16, 1>)
        {axis = -1 : i64, epsilon = 9.99999974e-06 : f32, stash_type = 1 : i64}
    ```

    Supports dynamic shapes:
    ```mlir
    hip.rms_norm(%ctx)
        ins(%input, %scale : memref<?x?xf16, 1>, memref<?xf16, 1>)
        outs(%output : memref<?x?xf16, 1>)
        {axis = -1 : i64, epsilon = 1.0e-05 : f32, stash_type = 1 : i64}
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$input,
    Hip_TensorOrMemRef:$scale,
    Hip_TensorOrMemRef:$output,
    I64Attr:$axis,
    F32Attr:$epsilon,
    I64Attr:$stash_type
  );

  // result_tensors inherited from Hip_DpsOp
  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $input `,` $scale `:` type($input) `,` type($scale) `)`
    `outs` `(` $output `:` type($output) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
  let hasVerifier = 1;
}

def Hip_LayerNormOp : Hip_DpsOp<"layer_norm",
    [AttrSizedOperandSegments]> {
  let summary = "Standard layer normalization (ONNX LayerNormalization opset 17)";
  let description = [{
    Full ONNX LayerNormalization (opset 17+) with mean subtraction and an
    optional bias term, fused into a single runtime call.

    Stage 1 (in stash_type precision):
      Mean       = ReduceMean(X, axes=[axis ... rank-1])
      D          = X - Mean
      Var        = ReduceMean(D * D, axes=[axis ... rank-1])
      InvStdDev  = 1 / sqrt(Var + epsilon)
      Normalized = D * InvStdDev   (cast back to T)

    Stage 2 (in T precision):
      NormalizedScaled = Normalized * Scale
      Y                = NormalizedScaled + B   (only when B is present)

    Inputs (2-3, MS spec order):
      input  (required, T) - N-D tensor to normalize
      scale  (required, T) - unidirectional broadcastable scale
      bias   (optional, T) - unidirectional broadcastable bias

    DPS output buffers (1-3, Variadic outputs):
      outputs[0] = output     (required, T) - normalized result, same shape as input
      outputs[1] = mean       (optional, U) - saved mean (training-only)
      outputs[2] = inv_std    (optional, U) - saved inverse standard deviation
                                              (training-only)

    Attributes:
      axis       - first normalization dimension (negative = from rank-1, default -1)
      epsilon    - small value added to variance (default 1e-5)
      stash_type - internal computation precision (1 = float, default)

    The runtime treats all dims < axis as the "row" dimension and all dims >= axis
    as the "feature" dimension being normalized. mean / inv_std outputs may be
    nullptr in the runtime call when the corresponding optional output buffer
    is omitted.

    Example (inference, output only):
    ```mlir
    hip.layer_norm(%ctx)
        ins(%input, %scale, %bias :
            memref<1x128x4096xf16, 1>, memref<4096xf16, 1>, memref<4096xf16, 1>)
        outs(%output : memref<1x128x4096xf16, 1>)
        {axis = -1 : i64, epsilon = 9.99999974e-06 : f32, stash_type = 1 : i64}
    ```

    Example (training, all 3 outputs, no bias):
    ```mlir
    hip.layer_norm(%ctx)
        ins(%input, %scale :
            memref<1x128x4096xf16, 1>, memref<4096xf16, 1>)
        outs(%output, %mean, %inv_std :
            memref<1x128x4096xf16, 1>,
            memref<1x128x1xf32, 1>, memref<1x128x1xf32, 1>)
        {axis = -1 : i64, epsilon = 1.0e-05 : f32, stash_type = 1 : i64}
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,

    // ONNX spec inputs 1-2 (required)
    Hip_TensorOrMemRef:$input,
    Hip_TensorOrMemRef:$scale,

    // ONNX spec input 3 (optional)
    Optional<Hip_TensorOrMemRef>:$bias,

    // DPS output buffers (1-3): [output] | [output, mean] | [output, mean, inv_std]
    Variadic<Hip_TensorOrMemRef>:$outputs,

    // Attributes (mirror ONNX LayerNormalization)
    DefaultValuedAttr<I64Attr, "-1">:$axis,
    DefaultValuedAttr<F32Attr, "9.99999974E-6">:$epsilon,
    DefaultValuedAttr<I64Attr, "1">:$stash_type
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(`
        $input `,` $scale
        ( `,` $bias^ )?
        `:` type($input) `,` type($scale)
        ( `,` type($bias)^ )?
    `)`
    `outs` `(` $outputs `:` type($outputs) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
  let hasVerifier = 1;
}

def Hip_SkipRmsNormOp : Hip_DpsOp<"skip_rms_norm",
    [AttrSizedOperandSegments]> {
  let summary = "Fused skip + RMS normalization (com.microsoft.SkipSimplifiedLayerNormalization)";
  let description = [{
    Full alignment with Microsoft ONNX Runtime SkipSimplifiedLayerNormalization
    specification (v1).

    Performs fused skip connection addition followed by RMS normalization:
      input_skip_bias_sum = input + skip [+ bias]
      output = RMSNorm(input_skip_bias_sum) * gamma

    RMSNorm(x) = x / sqrt(mean(x^2) + epsilon)

    Inputs (3-4):
      1. input   (required, T) - 3D or 2D input tensor
      2. skip    (required, T) - skip connection tensor (same shape as input)
      3. gamma   (required, T) - 1D normalization weight (hidden_size)
      4. bias    (optional, T) - 1D bias (hidden_size)

    DPS output buffers (1-2, passed as Variadic `outputs`):
      outputs[0] = output             (required, T) - normalized result
      outputs[1] = input_skip_bias_sum (optional, T) - residual sum (input + skip [+ bias])

    Uses destination-passing style: output buffers are provided as arguments.

    Example (without bias, 1 output):
    ```mlir
    hip.skip_rms_norm(%ctx)
        ins(%input, %skip, %gamma : memref<1x128x4096xf16, 1>,
                                     memref<1x128x4096xf16, 1>,
                                     memref<4096xf16, 1>)
        outs(%output : memref<1x128x4096xf16, 1>)
        {epsilon = 1.0e-05 : f32}
    ```

    Example (with bias, 2 outputs):
    ```mlir
    hip.skip_rms_norm(%ctx)
        ins(%input, %skip, %gamma, %bias :
            memref<1x128x4096xf16, 1>, memref<1x128x4096xf16, 1>,
            memref<4096xf16, 1>, memref<4096xf16, 1>)
        outs(%output, %input_skip_bias_sum : memref<1x128x4096xf16, 1>,
                                              memref<1x128x4096xf16, 1>)
        {epsilon = 1.0e-05 : f32}
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,

    // MS spec inputs 1-3 (required)
    Hip_TensorOrMemRef:$input,
    Hip_TensorOrMemRef:$skip,
    Hip_TensorOrMemRef:$gamma,

    // MS spec input 4 (optional)
    Optional<Hip_TensorOrMemRef>:$bias,

    // DPS output buffers: [output] or [output, input_skip_bias_sum]
    Variadic<Hip_TensorOrMemRef>:$outputs,

    // Attributes
    F32Attr:$epsilon
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(`
        $input `,` $skip `,` $gamma
        ( `,` $bias^ )?
        `:` type($input) `,` type($skip) `,` type($gamma)
        ( `,` type($bias)^ )?
    `)`
    `outs` `(` $outputs `:` type($outputs) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
  let hasVerifier = 1;
}

def Hip_RopeOp : Hip_DpsOp<"rope"> {
  let summary = "Rotary position embedding (RoPE)";
  let description = [{
    Applies rotary position embedding to the input tensor.
    Implements the Microsoft RotaryEmbedding operator.

    Uses destination-passing style: output buffer is provided as argument.

    Attributes:
    - interleaved: Rotary embedding interleaved format (0 or 1)
    - num_heads: Number of attention heads
    - rotary_embedding_dim: Dimension of rotary embedding

    Example:
    ```mlir
    hip.rope(%ctx)
        ins(%input, %position_ids, %cos_cache, %sin_cache :
            memref<1x128x4096xf16, 1>, memref<1x128xi64, 1>,
            memref<131072x64xf16, 1>, memref<131072x64xf16, 1>)
        outs(%output : memref<1x128x4096xf16, 1>)
        {interleaved = 0 : i64, num_heads = 32 : i64, rotary_embedding_dim = 128 : i64}
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$input,
    Hip_TensorOrMemRef:$position_ids,
    Hip_TensorOrMemRef:$cos_cache,
    Hip_TensorOrMemRef:$sin_cache,
    Hip_TensorOrMemRef:$output,
    I64Attr:$interleaved,
    I64Attr:$num_heads,
    I64Attr:$rotary_embedding_dim
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(`
        $input `,` $position_ids `,` $cos_cache `,` $sin_cache `:`
        type($input) `,` type($position_ids) `,` type($cos_cache) `,` type($sin_cache)
    `)`
    `outs` `(` $output `:` type($output) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_MiopenAddOp : Hip_DpsOp<"miopen.add"> {
  let summary = "Element-wise add via miopenOpTensor(miopenTensorOpAdd)";
  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$A,
    Hip_TensorOrMemRef:$B,
    Hip_TensorOrMemRef:$C
  );
  // result_tensors inherited from Hip_DpsOp
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def Hip_MulOp : Hip_DpsOp<"mul"> {
  let summary = "Elementwise multiplication";
  let description = [{
    Performs elementwise multiplication: output = lhs * rhs

    Uses destination-passing style: output buffer is provided as argument.

    Example:
    ```mlir
    hip.mul(%ctx) ins(%lhs, %rhs : memref<1x128x14336xf16, 1>,
                                   memref<1x128x14336xf16, 1>)
                  outs(%output : memref<1x128x14336xf16, 1>)
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$lhs,
    Hip_TensorOrMemRef:$rhs,
    Hip_TensorOrMemRef:$output
  );

  // result_tensors inherited from Hip_DpsOp
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def Hip_AddOp : Hip_DpsOp<"add"> {
  let summary = "Elementwise addition";
  let description = [{
    Performs elementwise addition: output = lhs + rhs

    Uses destination-passing style: output buffer is provided as argument.

    Example:
    ```mlir
    hip.add(%ctx) ins(%lhs, %rhs : memref<1x128x32xf16, 1>,
                                   memref<1x128x32xf16, 1>)
                  outs(%output : memref<1x128x32xf16, 1>)
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$lhs,
    Hip_TensorOrMemRef:$rhs,
    Hip_TensorOrMemRef:$output
  );

  // result_tensors inherited from Hip_DpsOp
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def Hip_MinOp : Hip_DpsOp<"min"> {
  let summary = "Elementwise minimum";
  let description = [{
    Performs elementwise minimum: output = min(lhs, rhs)

    Supports multidirectional (NumPy-style) broadcasting via MIOpen's
    miopenOpTensor with miopenTensorOpMin.

    Uses destination-passing style: output buffer is provided as argument.

    Example:
    ```mlir
    hip.min(%ctx) ins(%lhs, %rhs : memref<3xf32, 1>,
                                   memref<3xf32, 1>)
                  outs(%output : memref<3xf32, 1>)
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$lhs,
    Hip_TensorOrMemRef:$rhs,
    Hip_TensorOrMemRef:$output
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $lhs `,` $rhs `:` type($lhs) `,` type($rhs) `)`
    `outs` `(` $output `:` type($output) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_MiopenSoftmaxOp : Hip_DpsOp<"miopen.softmax"> {
  let summary = "Softmax via miopenSoftmaxForward_V2 (row-wise, last dim)";
  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$input,
    Hip_TensorOrMemRef:$output
  );
  // result_tensors inherited from Hip_DpsOp
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

// ===== Custom HIP kernel ops (no library equivalent) =========================

def Hip_TransposeOp : Hip_DpsOp<"transpose"> {
  let summary = "N-D transpose with arbitrary permutation (ONNX Transpose)";
  let description = [{
    Permutes the dimensions of the input tensor according to the `perm`
    attribute. Output dim i corresponds to input dim perm[i]. Implements
    the ONNX Transpose operator.

    The `perm` attribute is a list of integers and must be a permutation of
    [0, 1, ..., rank-1]. Its length must equal the rank of the input. When
    the ONNX `perm` attribute is omitted, the converter materializes the
    default reverse permutation `[rank-1, ..., 0]` here.

    Uses destination-passing style: output buffer is provided as argument.

    Example:
    ```mlir
    hip.transpose(%ctx) ins(%input : memref<1x2x3xf32, 1>)
                        outs(%output : memref<2x3x1xf32, 1>)
                        {perm = [1, 2, 0]}
    ```
  }];
  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$input,
    Hip_TensorOrMemRef:$output,
    I64ArrayAttr:$perm
  );
  // result_tensors inherited from Hip_DpsOp
  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $input `:` type($input) `)`
    `outs` `(` $output `:` type($output) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
  let hasVerifier = 1;
}

def Hip_GatherOp : Hip_DpsOp<"gather"> {
  let summary = "Gather elements along an axis";
  let description = [{
    Gathers elements from data tensor along the given axis using indices.
    Implements the ONNX Gather operator.

    Uses destination-passing style: output buffer is provided as argument.

    Example:
    ```mlir
    hip.gather(%ctx) ins(%data, %indices : memref<3x4xf32, 1>, memref<2xi64, 1>)
                     outs(%output : memref<2x4xf32, 1>) {axis = 0 : i64}
    ```
  }];
  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$data,
    Hip_TensorOrMemRef:$indices,
    Hip_TensorOrMemRef:$output,
    DefaultValuedAttr<I64Attr, "0">:$axis
  );
  // result_tensors inherited from Hip_DpsOp
  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $data `,` $indices `:` type($data) `,` type($indices) `)`
    `outs` `(` $output `:` type($output) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_RangeOp : Hip_DpsOp<"range"> {
  let summary = "Generate 1-D range tensor";
  let description = [{
    Implements ONNX Range semantics:
      output[i] = start + i * delta, limit is exclusive.

    start/limit/delta are scalar tensors (rank-0), output is rank-1.
    Uses destination-passing style: output buffer is provided as argument.
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$start,
    Hip_TensorOrMemRef:$limit,
    Hip_TensorOrMemRef:$delta,
    Hip_TensorOrMemRef:$output
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $start `,` $limit `,` $delta `:`
    type($start) `,` type($limit) `,` type($delta) `)`
    `outs` `(` $output `:` type($output) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_SiluOp : Hip_DpsOp<"silu"> {
  let summary = "SiLU activation: output = input * sigmoid(input)";
  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$input,
    Hip_TensorOrMemRef:$output
  );
  // result_tensors inherited from Hip_DpsOp
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def Hip_SigmoidOp : Hip_DpsOp<"sigmoid"> {
  let summary = "Sigmoid activation";
  let description = [{
    Applies sigmoid activation: y = 1 / (1 + exp(-x))

    Uses destination-passing style: output buffer is provided as argument.

    Example:
    ```mlir
    hip.sigmoid(%ctx) ins(%x : memref<1x128x512xf32, 1>)
                      outs(%y : memref<1x128x512xf32, 1>)
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$x,
    Hip_TensorOrMemRef:$y
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $x `:` type($x) `)`
    `outs` `(` $y `:` type($y) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_SoftplusOp : Hip_DpsOp<"softplus"> {
  let summary = "Softplus activation: y = ln(exp(x) + 1)";
  let description = [{
    Applies the softplus activation function element-wise.
    Implements the standard ONNX Softplus operator.

    Formula: Y = ln(exp(X) + 1)

    Uses destination-passing style: output buffer is provided as argument.

    Example:
    ```mlir
    hip.softplus(%ctx) ins(%x : memref<1x128x512xf32, 1>)
                       outs(%y : memref<1x128x512xf32, 1>)
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$x,
    Hip_TensorOrMemRef:$y
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $x `:` type($x) `)`
    `outs` `(` $y `:` type($y) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_GeluOp : Hip_DpsOp<"gelu"> {
  let summary = "GELU activation (Gaussian Error Linear Unit)";
  let description = [{
    Applies the GELU activation function element-wise.
    Implements the ONNX Gelu operator with support for both exact and approximate modes.

    Approximate mode (approximate="tanh"):
      Y ≈ 0.5 * X * (1 + tanh(sqrt(2/π) * (X + 0.044715 * X³)))

    Exact mode (approximate="none", default):
      Y = X * 0.5 * (1.0 + erf(X / sqrt(2.0)))

    Uses destination-passing style: output buffer is provided as argument.

    Example:
    ```mlir
    hip.gelu(%ctx) ins(%x : memref<1x128x768xf16, 1>)
                   outs(%y : memref<1x128x768xf16, 1>)
                   {approximate = "tanh"}
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$input,
    Hip_TensorOrMemRef:$output,
    DefaultValuedAttr<StrAttr, "\"none\"">:$approximate
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $input `:` type($input) `)`
    `outs` `(` $output `:` type($output) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_ReciprocalOp : Hip_DpsOp<"reciprocal"> {
  let summary = "Element-wise reciprocal: y = 1/x";
  let description = [{
    Computes the reciprocal of each element.
    Implements the standard ONNX Reciprocal operator.

    Formula: Y = 1 / X

    Uses destination-passing style: output buffer is provided as argument.

    Example:
    ```mlir
    hip.reciprocal(%ctx) ins(%x : memref<1x128x512xf32, 1>)
                         outs(%y : memref<1x128x512xf32, 1>)
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$x,
    Hip_TensorOrMemRef:$y
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $x `:` type($x) `)`
    `outs` `(` $y `:` type($y) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_SqrtOp : Hip_DpsOp<"sqrt"> {
  let summary = "Element-wise square root: y = sqrt(x)";
  let description = [{
    Computes the square root of each element.
    Formula: Y = sqrt(X) = X^0.5

    Negative inputs return NaN (per ONNX Sqrt specification).

    Uses destination-passing style: output buffer is provided as argument.

    Example:
    ```mlir
    hip.sqrt(%ctx) ins(%x : memref<1x128x512xf32, 1>)
                   outs(%y : memref<1x128x512xf32, 1>)
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$x,
    Hip_TensorOrMemRef:$y
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $x `:` type($x) `)`
    `outs` `(` $y `:` type($y) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_DivOp : Hip_DpsOp<"div"> {
  let summary = "Elementwise division";
  let description = [{
    Performs elementwise division: output = lhs / rhs

    Supports multidirectional (Numpy-style) broadcasting.

    Uses destination-passing style: output buffer is provided as argument.

    Example:
    ```mlir
    hip.div(%ctx) ins(%lhs, %rhs : memref<3x4xf32, 1>,
                                   memref<3x4xf32, 1>)
                  outs(%output : memref<3x4xf32, 1>)
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$lhs,
    Hip_TensorOrMemRef:$rhs,
    Hip_TensorOrMemRef:$output
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $lhs `,` $rhs `:` type($lhs) `,` type($rhs) `)`
    `outs` `(` $output `:` type($output) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_EqualOp : Hip_DpsOp<"equal"> {
  let summary = "Element-wise equality comparison: c = (a == b)";
  let description = [{
    Compares two input tensors element-wise for equality.
    Output is a boolean tensor (i1) with the same broadcast shape.
    Implements the standard ONNX Equal operator.

    Supports multidirectional (Numpy-style) broadcasting.

    Uses destination-passing style: output buffer is provided as argument.

    Example:
    ```mlir
    hip.equal(%ctx) ins(%a, %b : memref<3x4xf32, 1>, memref<3x4xf32, 1>)
                    outs(%c : memref<3x4xi1, 1>)
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$lhs,
    Hip_TensorOrMemRef:$rhs,
    Hip_TensorOrMemRef:$output
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $lhs `,` $rhs `:` type($lhs) `,` type($rhs) `)`
    `outs` `(` $output `:` type($output) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_AndOp : Hip_DpsOp<"and"> {
  let summary = "Element-wise logical AND: c = a && b";
  let description = [{
    Computes element-wise logical AND of two boolean tensors (i1).
    Implements the standard ONNX And operator.

    Supports multidirectional (NumPy-style) broadcasting between `lhs`
    and `rhs`. Both inputs and the output have boolean (i1) element type.

    Uses destination-passing style: output buffer is provided as argument.

    Example:
    ```mlir
    hip.and(%ctx) ins(%a, %b : memref<3x4xi1, 1>, memref<3x4xi1, 1>)
                  outs(%c : memref<3x4xi1, 1>)
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$lhs,
    Hip_TensorOrMemRef:$rhs,
    Hip_TensorOrMemRef:$output
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $lhs `,` $rhs `:` type($lhs) `,` type($rhs) `)`
    `outs` `(` $output `:` type($output) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_NotOp : Hip_DpsOp<"not"> {
  let summary = "Element-wise logical NOT: y = !x";
  let description = [{
    Computes the logical NOT of each element (boolean/i1).
    Implements the standard ONNX Not operator.

    Formula: Y = !X

    Uses destination-passing style: output buffer is provided as argument.

    Example:
    ```mlir
    hip.not(%ctx) ins(%x : memref<3x4xi1, 1>)
                  outs(%y : memref<3x4xi1, 1>)
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$x,
    Hip_TensorOrMemRef:$y
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $x `:` type($x) `)`
    `outs` `(` $y `:` type($y) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_CosOp : Hip_DpsOp<"cos"> {
  let summary = "Element-wise cosine: y = cos(x)";
  let description = [{
    Computes the cosine of each element.
    Implements the standard ONNX Cos operator.

    Formula: Y = cos(X)

    Uses destination-passing style: output buffer is provided as argument.

    Example:
    ```mlir
    hip.cos(%ctx) ins(%x : memref<1x128x512xf32, 1>)
                  outs(%y : memref<1x128x512xf32, 1>)
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$x,
    Hip_TensorOrMemRef:$y
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $x `:` type($x) `)`
    `outs` `(` $y `:` type($y) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_SinOp : Hip_DpsOp<"sin"> {
  let summary = "Element-wise sine: y = sin(x)";
  let description = [{
    Computes the sine of each element.
    Implements the standard ONNX Sin operator.

    Formula: Y = sin(X)

    Uses destination-passing style: output buffer is provided as argument.

    Example:
    ```mlir
    hip.sin(%ctx) ins(%x : memref<1x128x512xf32, 1>)
                  outs(%y : memref<1x128x512xf32, 1>)
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$x,
    Hip_TensorOrMemRef:$y
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $x `:` type($x) `)`
    `outs` `(` $y `:` type($y) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_NegOp : Hip_DpsOp<"neg"> {
  let summary = "Element-wise negation: y = -x";
  let description = [{
    Negates each element of the input tensor.
    Implements the standard ONNX Neg operator.

    Formula: Y = -X

    Lowered via the unified unary elementwise path (wrap_neg).

    Uses destination-passing style: output buffer is provided as argument.

    Example:
    ```mlir
    hip.neg(%ctx) ins(%x : memref<3x4xf32, 1>)
                  outs(%y : memref<3x4xf32, 1>)
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$x,
    Hip_TensorOrMemRef:$y
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $x `:` type($x) `)`
    `outs` `(` $y `:` type($y) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_SubOp : Hip_DpsOp<"sub"> {
  let summary = "Elementwise subtraction";
  let description = [{
    Performs elementwise subtraction: output = lhs - rhs

    Both inputs must have the same shape. Broadcasting is not currently supported
    and must be handled before conversion to HIP dialect.

    Uses destination-passing style: output buffer is provided as argument.

    Example:
    ```mlir
    hip.sub(%ctx) ins(%lhs, %rhs : memref<1x1xi64, 1>, memref<1x1xi64, 1>)
                  outs(%output : memref<1x1xi64, 1>)
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$lhs,
    Hip_TensorOrMemRef:$rhs,
    Hip_TensorOrMemRef:$output
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $lhs `,` $rhs `:` type($lhs) `,` type($rhs) `)`
    `outs` `(` $output `:` type($output) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_WhereOp : Hip_DpsOp<"where"> {
  let summary = "Elementwise selection: output = condition ? x : y";
  let description = [{
    Implements ONNX Where: output[i] = condition[i] ? x[i] : y[i].

    Supports multidirectional (NumPy-style) broadcasting between condition,
    x and y. Each operand's shape is broadcast to the output shape; dims of
    1 are broadcast against larger dims of the other operands.

    The condition tensor must have a boolean (i1) element type. The x and y
    tensors must share the same element type, which is also the output type.

    Uses destination-passing style: output buffer is provided as argument.

    Example:
    ```mlir
    hip.where(%ctx) ins(%cond, %x, %y :
                        memref<1x4xi1, 1>,
                        memref<2x1xf32, 1>,
                        memref<2x4xf32, 1>)
                    outs(%out : memref<2x4xf32, 1>)
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$condition,
    Hip_TensorOrMemRef:$x,
    Hip_TensorOrMemRef:$y,
    Hip_TensorOrMemRef:$output
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $condition `,` $x `,` $y `:`
        type($condition) `,` type($x) `,` type($y) `)`
    `outs` `(` $output `:` type($output) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_CastOp : Hip_DpsOp<"cast"> {
  let summary = "Type conversion operation";
  let description = [{
    Converts input tensor from one data type to another.

    Input and output may have different element types but the same shape.

    Uses destination-passing style: output buffer is provided as argument.

    Example:
    ```mlir
    hip.cast(%ctx) ins(%input : memref<128x256xf32, 1>)
                   outs(%output : memref<128x256xf16, 1>) {to = 10 : i64}
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$input,
    Hip_TensorOrMemRef:$output,
    I64Attr:$to
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $input `:` type($input) `)`
    `outs` `(` $output `:` type($output) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_ReduceSumOp : Hip_DpsOp<"reduce_sum"> {
  let summary = "Reduce tensor by summing along axes";
  let description = [{
    Computes the sum of input tensor elements along the specified axes.
    Implements the ONNX ReduceSum operator (opset 13+).

    Uses destination-passing style: output buffer is provided as argument.

    When axes is not provided in ONNX, an empty tensor<0xi64> is passed.
    Combined with noop_with_empty_axes attribute, this enables ONNX-compliant
    optional axes semantics while keeping all operands required.

    Example:
    ```mlir
    hip.reduce_sum(%ctx) ins(%data, %axes : memref<1x128xi64, 1>, memref<i64, 1>)
                         outs(%output : memref<1x1xi64, 1>) {keepdims = 1 : i64, noop_with_empty_axes = 0 : i64}
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$data,
    Hip_TensorOrMemRef:$axes,
    Hip_TensorOrMemRef:$output,
    DefaultValuedAttr<I64Attr, "1">:$keepdims,
    DefaultValuedAttr<I64Attr, "0">:$noop_with_empty_axes
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $data `,` $axes `:` type($data) `,` type($axes) `)`
    `outs` `(` $output `:` type($output) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_ReduceMaxOp : Hip_DpsOp<"reduce_max"> {
  let summary = "Reduce tensor by taking maximum along axes";
  let description = [{
    Computes the max of input tensor elements along the specified axes.
    Implements the ONNX ReduceMax operator (opset 18+).

    Uses destination-passing style: output buffer is provided as argument.

    When axes is not provided in ONNX, an empty tensor<0xi64> is passed.
    Combined with noop_with_empty_axes attribute, this enables ONNX-compliant
    optional axes semantics while keeping all operands required.

    Example:
    ```mlir
    hip.reduce_max(%ctx) ins(%data, %axes : memref<1x128xi64, 1>, memref<i64, 1>)
                         outs(%output : memref<1x1xi64, 1>) {keepdims = 1 : i64, noop_with_empty_axes = 0 : i64}
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$data,
    Hip_TensorOrMemRef:$axes,
    Hip_TensorOrMemRef:$output,
    DefaultValuedAttr<I64Attr, "1">:$keepdims,
    DefaultValuedAttr<I64Attr, "0">:$noop_with_empty_axes
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $data `,` $axes `:` type($data) `,` type($axes) `)`
    `outs` `(` $output `:` type($output) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

// ===== Causal Conv with State ================================================

def Hip_CausalConvWithStateOp : Hip_DpsOp<"causal_conv_with_state",
    [AttrSizedOperandSegments]> {
  let summary = "Stateful causal depthwise convolution";
  let description = [{
    Causal depthwise convolution with carry state for incremental decoding.

    Used by Gated DeltaNet (Qwen3.5) and Mamba (Jamba, FalconMamba) as a
    preprocessing step. Replaces the 3-op pattern (Concat + Conv + Slice)
    with a single fused operation.

    The convolution is causal (looks only at current and past positions along
    the last spatial dimension) and depthwise (each channel is convolved
    independently with its own kernel).

    Input layout is channels-first: (batch_size, channels, ...).
    Weight layout: (channels, 1, k_1, ...) for depthwise convolution.
    The carry state stores the last (k-1) positions along the causal axis
    for incremental decode.

    Example (1D):
    ```mlir
    hip.causal_conv_with_state(%ctx)
        ins(%input, %weight, %bias, %past_state :
            tensor<1x64x128xf16>, tensor<64x1x4xf16>,
            tensor<64xf16>, tensor<1x64x3xf16>)
        outs(%output, %present_state :
             tensor<1x64x128xf16>, tensor<1x64x3xf16>)
        {activation = "silu", ndim = 1 : i64}
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,

    // Required inputs
    Hip_TensorOrMemRef:$input,           // (batch, channels, ...)
    Hip_TensorOrMemRef:$weight,          // (channels, 1, k_1, ...)

    // Optional inputs
    Optional<Hip_TensorOrMemRef>:$bias,        // (channels)
    Optional<Hip_TensorOrMemRef>:$past_state,  // (batch, channels, k-1)

    // DPS output buffers
    Hip_TensorOrMemRef:$output,          // same shape as input
    Hip_TensorOrMemRef:$present_state,   // (batch, channels, k-1)

    // Attributes
    DefaultValuedAttr<StrAttr, "\"none\"">:$activation,  // "none", "silu", "swish"
    DefaultValuedAttr<I64Attr, "1">:$ndim                // 1, 2, or 3
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(`
        $input `,` $weight
        (`,` $bias^)?
        (`,` $past_state^)?
        `:`
        type($input) `,` type($weight)
        (`,` type($bias)^)?
        (`,` type($past_state)^)?
    `)`
    `outs` `(`
        $output `,` $present_state
        `:`
        type($output) `,` type($present_state)
    `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

// ===== Group Query Attention =================================================

def Hip_GqaOp : Hip_DpsOp<"gqa", [AttrSizedOperandSegments]> {
  let summary = "Group Query Attention - Full MS spec implementation";
  let description = [{
    Complete implementation of com.microsoft.GroupQueryAttention (v1) specification.

    Supported features:
    - Grouped-query/Multi-query attention (GQA/MQA)
    - KV cache (BNSH format, supports past/present buffer sharing)
    - RoPE (rotary position embedding, supports cos/sin cache + position_ids)
    - Attention bias (ALiBi, T5 relative position bias, etc.)
    - Local attention (Mistral sliding window)
    - KV cache quantization (float8/int8/int4, per-tensor/per-channel)
    - Smooth softmax (numerical stability optimization)
    - Debug output (QK matrix visualization)

    Cache format: BNSH (batch_size, num_heads, seq_len, head_size)

    MS spec reference:
    https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#commicrosoftgroupqueryattention

    Example:
    ```mlir
    hip.gqa(%ctx)
        ins(%query, %key, %value, %past_key, %past_value,
            %seqlens_k, %total_seq_len :
            tensor<1x128x4096xf16>, ...)
        outs(%output, %present_key, %present_value :
             tensor<1x128x4096xf16>, ...)
        {num_heads = 32, kv_num_heads = 8, scale = 0.0884}
    ```
  }];

  let arguments = (ins
    // Internal context (HIP-specific, not ONNX spec)
    Hip_ContextType:$ctx,

    // === MS GQA Standard Inputs (spec order 1-14) ===

    // Input 1: query (required)
    Hip_TensorOrMemRef:$query,

    // Inputs 2-3: key/value (optional - supports packed QKV)
    Optional<Hip_TensorOrMemRef>:$key,
    Optional<Hip_TensorOrMemRef>:$value,

    // Inputs 4-5: past KV cache (optional - first inference has no past)
    Optional<Hip_TensorOrMemRef>:$past_key,
    Optional<Hip_TensorOrMemRef>:$past_value,

    // Inputs 6-7: sequence length info (required)
    Hip_TensorOrMemRef:$seqlens_k,        // 1D (batch_size)
    Hip_TensorOrMemRef:$total_seq_len,    // Scalar

    // Inputs 8-10: RoPE related (optional)
    Optional<Hip_TensorOrMemRef>:$cos_cache,      // 2D (max_seq_len, head_size/2)
    Optional<Hip_TensorOrMemRef>:$sin_cache,      // 2D (max_seq_len, head_size/2)
    Optional<Hip_TensorOrMemRef>:$position_ids,   // 2D (batch, seq_len), int64

    // Input 11: attention bias (optional - ALiBi etc.)
    Optional<Hip_TensorOrMemRef>:$attention_bias, // 4D (batch/1, heads/1, seq_q, seq_kv)

    // Input 12: smooth softmax factor (optional)
    Optional<Hip_TensorOrMemRef>:$head_sink,      // 1D (num_heads)

    // Inputs 13-14: KV cache quantization scale (optional)
    Optional<Hip_TensorOrMemRef>:$k_scale,        // Broadcastable to present_key shape
    Optional<Hip_TensorOrMemRef>:$v_scale,        // Broadcastable to present_value shape

    // === DPS-style output buffers ===
    Hip_TensorOrMemRef:$output,        // 3D (batch, seq_len, hidden_size)
    Hip_TensorOrMemRef:$present_key,   // 4D BNSH format
    Hip_TensorOrMemRef:$present_value, // 4D BNSH format
    Optional<Hip_TensorOrMemRef>:$output_qk,  // Optional QK matrix output (debug)

    // === MS GQA Standard Attributes ===

    // Required attributes
    I64Attr:$num_heads,           // Query head count
    I64Attr:$kv_num_heads,        // KV head count (GQA core)

    // Optional attributes (with defaults)
    DefaultValuedAttr<F32Attr, "0.0">:$scale,  // 0.0 = auto-compute 1/√head_size at runtime (matches ORT sentinel)
    DefaultValuedAttr<I64Attr, "0">:$do_rotary,
    DefaultValuedAttr<I64Attr, "0">:$rotary_interleaved,
    DefaultValuedAttr<F32Attr, "0.0">:$softcap,
    DefaultValuedAttr<I64Attr, "-1">:$local_window_size,
    DefaultValuedAttr<I64Attr, "0">:$smooth_softmax,
    DefaultValuedAttr<I64Attr, "0">:$qk_output,  // 0=no output, 1=before softmax, 2=after softmax

    // Quantization related attributes
    DefaultValuedAttr<StrAttr, "\"NONE\"">:$k_quant_type,  // "NONE"/"PER_TENSOR"/"PER_CHANNEL"
    DefaultValuedAttr<StrAttr, "\"NONE\"">:$v_quant_type,
    DefaultValuedAttr<I64Attr, "8">:$kv_cache_bit_width  // 8 or 4
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(`
        $query
        ( `,` $key^ )?
        ( `,` $value^ )?
        ( `,` $past_key^ )?
        ( `,` $past_value^ )?
        `,` $seqlens_k `,` $total_seq_len
        ( `,` $cos_cache^ )?
        ( `,` $sin_cache^ )?
        ( `,` $position_ids^ )?
        ( `,` $attention_bias^ )?
        ( `,` $head_sink^ )?
        ( `,` $k_scale^ )?
        ( `,` $v_scale^ )?
        `:`
        type($query)
        ( `,` type($key)^ )?
        ( `,` type($value)^ )?
        ( `,` type($past_key)^ )?
        ( `,` type($past_value)^ )?
        `,` type($seqlens_k) `,` type($total_seq_len)
        ( `,` type($cos_cache)^ )?
        ( `,` type($sin_cache)^ )?
        ( `,` type($position_ids)^ )?
        ( `,` type($attention_bias)^ )?
        ( `,` type($head_sink)^ )?
        ( `,` type($k_scale)^ )?
        ( `,` type($v_scale)^ )?
    `)`
    `outs` `(`
        $output `,` $present_key `,` $present_value
        ( `,` $output_qk^ )?
        `:`
        type($output) `,` type($present_key) `,` type($present_value)
        ( `,` type($output_qk)^ )?
    `)`
    attr-dict (`:` type($result_tensors)^)?
  }];

  let hasVerifier = 1;
}

// ===== hipDNN Graph Outlining + Execution ====================================

def Hip_YieldOp : Hip_Op<"yield", [Pure, Terminator,
    HasParent<"HipDNNGraphOutlineOp">]> {
  let summary = "Yield results from a hipDNN graph outline region";
  let description = [{
    Terminates the body of a `hip.hipdnn_graph_outline` operation, forwarding
    the computed results back to the parent op.

    Example:
    ```mlir
    hip.yield %result : tensor<1x1x8x8xf32>
    ```
  }];

  let arguments = (ins Variadic<AnyRankedTensor>:$values);

  let builders = [
    OpBuilder<(ins), [{
      build($_builder, $_state, ::mlir::ValueRange{});
    }]>
  ];

  let assemblyFormat = "($values^ `:` type($values))? attr-dict";
}

def Hip_HipDNNGraphOutlineOp : Hip_Op<"hipdnn_graph_outline", [
    SingleBlockImplicitTerminator<"YieldOp">,
    IsolatedFromAbove
]> {
  let summary = "Outlined ONNX subgraph for hipDNN graph compilation";
  let description = [{
    Wraps one or more ONNX operations that are candidates for hipDNN graph
    compilation. The region captures only tensor operands as block arguments
    and is isolated from above.

    Created by the OutlineOnnxToHipDNN pass (handle-free, CLI-testable).
    Consumed by the CompileHipDNNGraphs pass, which either:
    - Compiles the region via hipDNN and replaces this op with hip.hipdnn_graph
    - Un-outlines (inlines the region back) on compilation failure

    Example:
    ```mlir
    %0 = hip.hipdnn_graph_outline
        ins(%x, %w : tensor<1x1x8x8xf32>, tensor<1x1x3x3xf32>)
        -> tensor<1x1x8x8xf32> {
      ^bb0(%arg0: tensor<1x1x8x8xf32>, %arg1: tensor<1x1x3x3xf32>):
        %1 = "onnx.Conv"(%arg0, %arg1) {auto_pad = "NOTSET", group = 1 : si64,
             kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]}
             : (tensor<1x1x8x8xf32>, tensor<1x1x3x3xf32>) -> tensor<1x1x8x8xf32>
        hip.yield %1 : tensor<1x1x8x8xf32>
    }
    ```
  }];

  let arguments = (ins Variadic<AnyRankedTensor>:$inputs);
  let results = (outs Variadic<AnyRankedTensor>:$results);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{
    `ins` `(` $inputs `:` type($inputs) `)` `->` type($results)
    $body attr-dict
  }];
}

// ===== hipDNN Graph Execution ================================================

def Hip_HipDNNGraphOp : Hip_DpsOp<"hipdnn_graph", [
    AttrSizedOperandSegments
]> {
  let summary = "Execute a pre-compiled hipDNN graph";
  let description = [{
    Dispatches to a pre-compiled hipDNN graph identified by graph_id.
    The graph was compiled during the CompileHipDNNGraphs pass and stored
    in the graph registry attached to RuntimeState.

    At runtime, looks up the graph by ID and calls graph->Execute() with
    a variant_pack built from the provided tensor UIDs and GPU pointers.

    Tensor UIDs (input_uids, output_uids) are assigned during graph building
    and baked into the IR as compile-time constants.

    Example:
    ```mlir
    hip.hipdnn_graph(%ctx) graph_id(0)
        ins(%input, %weights : tensor<1x3x224x224xf32>, tensor<64x3x7x7xf32>)
        outs(%output : tensor<1x64x112x112xf32>)
        {input_uids = [0 : i64, 1 : i64], output_uids = [2 : i64]}
        : tensor<1x64x112x112xf32>
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Variadic<Hip_TensorOrMemRef>:$inputs,
    Variadic<Hip_TensorOrMemRef>:$outputs,
    I32Attr:$graph_id,
    I64ArrayAttr:$input_uids,
    I64ArrayAttr:$output_uids
  );

  let assemblyFormat = [{
    `(` $ctx `)` `graph_id` `(` $graph_id `)`
    `ins` `(` $inputs `:` type($inputs) `)`
    `outs` `(` $outputs `:` type($outputs) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

// ===== Multi-Head Attention =================================================

def Hip_MultiHeadAttentionOp : Hip_DpsOp<"multi_head_attention",
    [AttrSizedOperandSegments]> {
  let summary = "Multi-Head Attention (com.microsoft.MultiHeadAttention v1)";
  let description = [{
    Implements the com.microsoft.MultiHeadAttention operator (opset v1).

    Multi-Head Self/Cross Attention with optional bias from input projection.
    Supports packed QKV in `query`, packed KV in `key`, separate Q/K/V,
    optional key padding mask, optional additive attention bias, KV cache
    (past_key/past_value) with optional buffer-sharing
    (past_sequence_length), unidirectional (causal) masking, and an optional
    cache_indirection buffer for beam-search decoding.

    Inputs (1-10, MS spec order):
      1. query                 (required, T) - either:
                                  * (batch, seq_q, hidden)         standard
                                  * (batch, seq_kv, num_heads, 3, head_size)
                                                                   packed QKV
      2. key                   (optional, T) - either:
                                  * (batch, seq_kv, hidden)
                                  * (batch, seq_kv, num_heads, 2, head_size)
                                                                   packed KV
                                  * (batch, num_heads, seq_kv, head_size)
                                                                   past_key
      3. value                 (optional, T) - (batch, seq_kv, v_hidden) or
                                                past_value layout
      4. bias                  (optional, T) - 1-D (hidden + hidden + v_hidden)
      5. key_padding_mask      (optional, M) - 1-D / 2-D / 3-D mask
      6. attention_bias        (optional, T) - bias added to QxK'
      7. past_key              (optional, T) - past KV cache for key
      8. past_value            (optional, T) - past KV cache for value
      9. past_sequence_length  (optional, M) - used with buffer sharing
     10. cache_indirection     (optional, M) - beam-search cache routing

    DPS output buffers (1-4):
      output         (required, T) - (batch, seq_q, v_hidden)
      present_key    (optional, T) - present KV cache for key
      present_value  (optional, T) - present KV cache for value
      qk             (optional, QK) - debug QK matrix

    Attributes:
      num_heads          (required) - number of attention heads
      mask_filter_value  (optional, default -10000.0) - value to fill masked
                                                       positions
      scale              (optional, default 0.0) - 0.0 means
                                                  1/sqrt(head_size) at runtime
      unidirectional     (optional, default 0) - 1 = causal mask

    Example (basic self-attention with separate Q/K/V):
    ```mlir
    hip.multi_head_attention(%ctx)
        ins(%query, %key, %value :
            memref<1x128x4096xf16, 1>, memref<1x128x4096xf16, 1>,
            memref<1x128x4096xf16, 1>)
        outs(%output : memref<1x128x4096xf16, 1>)
        {num_heads = 32 : i64, scale = 0.0 : f32,
         mask_filter_value = -1.000000e+04 : f32, unidirectional = 0 : i64}
    ```
  }];

  let arguments = (ins
    // Internal context (HIP-specific, not ONNX spec)
    Hip_ContextType:$ctx,

    // === MS MultiHeadAttention spec inputs 1-10 ===

    // Input 1: query (required)
    Hip_TensorOrMemRef:$query,

    // Inputs 2-3: key/value (optional - packed QKV in query when both absent)
    Optional<Hip_TensorOrMemRef>:$key,
    Optional<Hip_TensorOrMemRef>:$value,

    // Input 4: bias (optional - input-projection bias)
    Optional<Hip_TensorOrMemRef>:$bias,

    // Input 5: key_padding_mask (optional)
    Optional<Hip_TensorOrMemRef>:$key_padding_mask,

    // Input 6: attention_bias (optional - added to Q*K')
    Optional<Hip_TensorOrMemRef>:$attention_bias,

    // Inputs 7-8: past_key / past_value (optional - KV cache)
    Optional<Hip_TensorOrMemRef>:$past_key,
    Optional<Hip_TensorOrMemRef>:$past_value,

    // Input 9: past_sequence_length (optional - used with buffer sharing)
    Optional<Hip_TensorOrMemRef>:$past_sequence_length,

    // Input 10: cache_indirection (optional - beam-search cache routing)
    Optional<Hip_TensorOrMemRef>:$cache_indirection,

    // === DPS output buffers (1-4) ===
    Hip_TensorOrMemRef:$output,
    Optional<Hip_TensorOrMemRef>:$present_key,
    Optional<Hip_TensorOrMemRef>:$present_value,
    Optional<Hip_TensorOrMemRef>:$qk,

    // === Attributes ===
    I64Attr:$num_heads,
    DefaultValuedAttr<F32Attr, "-1.000000e+04">:$mask_filter_value,
    DefaultValuedAttr<F32Attr, "0.0">:$scale,
    DefaultValuedAttr<I64Attr, "0">:$unidirectional
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(`
        $query
        ( `,` $key^ )?
        ( `,` $value^ )?
        ( `,` $bias^ )?
        ( `,` $key_padding_mask^ )?
        ( `,` $attention_bias^ )?
        ( `,` $past_key^ )?
        ( `,` $past_value^ )?
        ( `,` $past_sequence_length^ )?
        ( `,` $cache_indirection^ )?
        `:`
        type($query)
        ( `,` type($key)^ )?
        ( `,` type($value)^ )?
        ( `,` type($bias)^ )?
        ( `,` type($key_padding_mask)^ )?
        ( `,` type($attention_bias)^ )?
        ( `,` type($past_key)^ )?
        ( `,` type($past_value)^ )?
        ( `,` type($past_sequence_length)^ )?
        ( `,` type($cache_indirection)^ )?
    `)`
    `outs` `(`
        $output
        ( `,` $present_key^ )?
        ( `,` $present_value^ )?
        ( `,` $qk^ )?
        `:`
        type($output)
        ( `,` type($present_key)^ )?
        ( `,` type($present_value)^ )?
        ( `,` type($qk)^ )?
    `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_LinearAttentionOp : Hip_DpsOp<"linear_attention",
    [AttrSizedOperandSegments]> {
  let summary = "Linear attention with recurrent state (com.microsoft.LinearAttention)";
  let description = [{
    Unified linear attention operator for autoregressive decoding (T=1) and
    prefill (T>1).

    All inputs use 3D packed format [B, T, H*D]; q_num_heads and kv_num_heads
    are always required. The op internally unpacks to 4D for computation.

    The update_rule attribute selects the recurrence type:
      "linear":       S_t = S_{t-1} + k_t (x) v_t
      "gated":        S_t = exp(g_t) * S_{t-1} + k_t (x) v_t
      "delta":        S_t = S_{t-1} + b_t * k_t (x) (v_t - S_{t-1}^T k_t)
      "gated_delta":  S_t = exp(g_t) * S_{t-1} + b_t * k_t (x) (v_t - exp(g_t) * S_{t-1}^T k_t)

    Head counts are three-way (subject to divisibility constraints):
      - H_q   = q_num_heads             (query heads, attribute)
      - H_kv  = kv_num_heads            (KV state heads, attribute)
      - n_k   = key.shape[-1] / d_k     (key heads, derived at runtime)
    Constraints:
      - n_k divides H_kv  (multiple KV heads may share one K head when n_k<H_kv)
      - Either H_q % H_kv == 0  (standard GQA, H_q >= H_kv)
        or     H_kv % H_q == 0  (inverse GQA,   H_q <  H_kv)

    Output head count H_o = max(H_q, H_kv):
      - Standard GQA (H_q >= H_kv): output packed in Q-head order.
      - Inverse GQA  (H_q <  H_kv): output packed in KV-head order, i.e. each
        KV state produces its own output slot.

    Inputs (3-6):
      1. query       (required, T) - [B, T, H_q * d_k]
      2. key         (required, T) - [B, T, n_k * d_k], n_k | H_kv
      3. value       (required, T) - [B, T, H_kv * d_v]
      4. past_state  (optional, S) - [B, H_kv, d_k, d_v]
      5. decay       (optional, T) - [B, T, H_kv * d_k] or [B, T, H_kv]
      6. beta        (optional, T) - [B, T, H_kv] or [B, T, 1]

    DPS output buffers:
      output        (required, T) - [B, T, max(H_q, H_kv) * d_v]
      present_state (required, S) - [B, H_kv, d_k, d_v]

    Example:
    ```mlir
    hip.linear_attention(%ctx)
        ins(%query, %key, %value :
            memref<1x1x4096xf16, 1>, memref<1x1x1024xf16, 1>,
            memref<1x1x1024xf16, 1>)
        past_state(%past : memref<1x8x128x128xf16, 1>)
        decay(%g : memref<1x1x1024xf16, 1>)
        beta(%b : memref<1x1x8xf16, 1>)
        outs(%output, %state : memref<1x1x4096xf16, 1>,
                                memref<1x8x128x128xf16, 1>)
        {q_num_heads = 32 : i64, kv_num_heads = 8 : i64,
         scale = 0.0 : f32, update_rule = "gated_delta"}
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,

    // Required inputs (spec order 1-3)
    Hip_TensorOrMemRef:$query,
    Hip_TensorOrMemRef:$key,
    Hip_TensorOrMemRef:$value,

    // Optional inputs (spec order 4-6)
    Optional<Hip_TensorOrMemRef>:$past_state,
    Optional<Hip_TensorOrMemRef>:$decay,
    Optional<Hip_TensorOrMemRef>:$beta,

    // DPS output buffers
    Hip_TensorOrMemRef:$output,
    Hip_TensorOrMemRef:$present_state,

    // Required attributes
    I64Attr:$q_num_heads,
    I64Attr:$kv_num_heads,

    // Optional attributes (with defaults)
    DefaultValuedAttr<F32Attr, "0.0">:$scale,
    DefaultValuedAttr<I64Attr, "0">:$chunk_size,
    DefaultValuedAttr<StrAttr, "\"gated_delta\"">:$update_rule
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(`
        $query `,` $key `,` $value `:`
        type($query) `,` type($key) `,` type($value)
    `)`
    (`past_state` `(` $past_state^ `:` type($past_state) `)`)?
    (`decay` `(` $decay^ `:` type($decay) `)`)?
    (`beta` `(` $beta^ `:` type($beta) `)`)?
    `outs` `(` $output `,` $present_state `:`
              type($output) `,` type($present_state) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_GemmOp : Hip_DpsOp<"gemm">{
  let summary = "General matrix multiplication (ONNX Gemm)";
  let description = [{
    Computes Y = alpha * A' * B' + beta * C, where:
      A' = transpose(A) if transA else A
      B' = transpose(B) if transB else B
    Input C is optional and broadcastable to (M, N).
    Implements ONNX Gemm operator (opset 13).
    Uses destination-passing style: output buffer is provided as argument.
    Example:
    ```mlir
    hip.gemm(%ctx) ins(%A, %B, %C :
        memref<128x256xf32, 1>, memref<256x512xf32, 1>, memref<512xf32, 1>)
        outs(%Y : memref<128x512xf32, 1>)
        {alpha = 1.0 : f32, beta = 1.0 : f32, transA = 0 : i64, transB = 0 : i64}
    ```
  }];
  let arguments = (ins
    Hip_ContextType:$ctx,

    // inputs
    Hip_TensorOrMemRef:$input_a,
    Hip_TensorOrMemRef:$input_b,
    Optional<Hip_TensorOrMemRef>:$input_c,  // optional

    //outputs
    Hip_TensorOrMemRef:$output,

    // attributes
    DefaultValuedAttr<F32Attr, "1.0">:$alpha,
    DefaultValuedAttr<F32Attr, "1.0">:$beta,
    DefaultValuedAttr<I64Attr, "0">:$transA,
    DefaultValuedAttr<I64Attr, "0">:$transB
  );
  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(`
        $input_a `,` $input_b (`,` $input_c^)?
        `:` type($input_a) `,` type($input_b) (`,` type($input_c)^)?
    `)`
    `outs` `(` $output `:` type($output) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_CumSumOp : Hip_DpsOp<"cumsum"> {
  let summary = "Cumulative sum along an axis";
  let description = [{
    Computes the cumulative sum of input tensor elements along the given axis.
    Implements the standard ONNX CumSum operator (opset 14+).

    The `axis` input is a 0-D scalar tensor selecting the reduction axis. The
    `exclusive` attribute (0 / 1) toggles inclusive vs exclusive accumulation,
    and `reverse` (0 / 1) reverses the accumulation direction along the axis.

    Uses destination-passing style: output buffer is provided as argument.

    Example:
    ```mlir
    hip.cumsum(%ctx) ins(%x, %axis : memref<3x4xf32, 1>, memref<i64, 1>)
                     outs(%y : memref<3x4xf32, 1>)
                     {exclusive = 0 : i64, reverse = 0 : i64}
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$x,
    Hip_TensorOrMemRef:$axis,
    Hip_TensorOrMemRef:$y,
    DefaultValuedAttr<I64Attr, "0">:$exclusive,
    DefaultValuedAttr<I64Attr, "0">:$reverse
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $x `,` $axis `:` type($x) `,` type($axis) `)`
    `outs` `(` $y `:` type($y) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_PadOp : Hip_DpsOp<"pad", [AttrSizedOperandSegments]> {
  let summary = "Pad a tensor with constant / reflect / edge / wrap";
  let description = [{
    Pads input tensor along the axes selected by `axes` (or all axes if
    `axes` is empty) according to the `pads` tensor.
    Implements the standard ONNX Pad operator (opset 25+).

    `pads` is a 1-D int64 tensor of shape `[2 * num_axes]` formatted as
    `[x1_begin, ..., xN_begin, x1_end, ..., xN_end]`. `constant_value` is
    an optional scalar used in `constant` mode (default 0). `axes` is an
    optional 1-D int64 tensor selecting which axes the pads apply to;
    when omitted (rank-0 / size-0), all axes are padded.

    The `mode` attribute is one of `"constant"`, `"reflect"`, `"edge"`,
    `"wrap"`.

    Uses destination-passing style: output buffer is provided as argument.

    Example:
    ```mlir
    hip.pad(%ctx) ins(%data, %pads : memref<3x2xf32, 1>, memref<4xi64, 1>)
                  cval(%cval : memref<f32, 1>)
                  axes(%axes : memref<2xi64, 1>)
                  outs(%out : memref<3x4xf32, 1>) {mode = "constant"}
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$data,
    Hip_TensorOrMemRef:$pads,
    Optional<Hip_TensorOrMemRef>:$constant_value,
    Optional<Hip_TensorOrMemRef>:$axes,
    Hip_TensorOrMemRef:$output,
    DefaultValuedAttr<StrAttr, "\"constant\"">:$mode
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $data `,` $pads `:` type($data) `,` type($pads) `)`
    (`cval` `(` $constant_value^ `:` type($constant_value) `)`)?
    (`axes` `(` $axes^ `:` type($axes) `)`)?
    `outs` `(` $output `:` type($output) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_TileOp : Hip_DpsOp<"tile"> {
  let summary = "Tile a tensor by repeating along each dimension";
  let description = [{
    Constructs a new tensor by repeating `input` along each dimension by the
    counts specified in `repeats`. Implements the standard ONNX Tile operator.

    `repeats` is a 1-D int64 tensor with the same length as `input`'s rank;
    `output_dim[i] = input_dim[i] * repeats[i]`.

    Uses destination-passing style: output buffer is provided as argument.

    Example:
    ```mlir
    hip.tile(%ctx) ins(%x, %repeats : memref<2x2xf32, 1>, memref<2xi64, 1>)
                   outs(%y : memref<2x4xf32, 1>)
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$input,
    Hip_TensorOrMemRef:$repeats,
    Hip_TensorOrMemRef:$output
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $input `,` $repeats `:` type($input) `,` type($repeats) `)`
    `outs` `(` $output `:` type($output) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_ExpandOp : Hip_DpsOp<"expand"> {
  let summary = "Broadcast a tensor to a target shape";
  let description = [{
    Broadcasts `input` to the target `shape` following NumPy broadcast rules.
    Implements the standard ONNX Expand operator.

    `shape` is a 1-D int64 tensor describing the target shape (which is
    broadcast against the input shape, so output rank may exceed `shape`'s
    length).

    Uses destination-passing style: output buffer is provided as argument.

    Example:
    ```mlir
    hip.expand(%ctx) ins(%x, %shape : memref<3x1xf32, 1>, memref<3xi64, 1>)
                     outs(%y : memref<2x3x6xf32, 1>)
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$input,
    Hip_TensorOrMemRef:$shape,
    Hip_TensorOrMemRef:$output
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $input `,` $shape `:` type($input) `,` type($shape) `)`
    `outs` `(` $output `:` type($output) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_ReduceProdOp : Hip_DpsOp<"reduce_prod"> {
  let summary = "Reduce tensor by multiplying elements along axes";
  let description = [{
    Computes the product of input tensor elements along the specified axes.
    Implements the ONNX ReduceProd operator (opset 18+).

    Uses destination-passing style: output buffer is provided as argument.

    When axes is not provided in ONNX, an empty tensor<0xi64> is passed.
    Combined with noop_with_empty_axes attribute, this enables ONNX-compliant
    optional axes semantics while keeping all operands required.

    Example:
    ```mlir
    hip.reduce_prod(%ctx) ins(%data, %axes : memref<3x2x2xf32, 1>, memref<0xi64, 1>)
                          outs(%output : memref<1x1x1xf32, 1>) {keepdims = 1 : i64, noop_with_empty_axes = 0 : i64}
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$data,
    Hip_TensorOrMemRef:$axes,
    Hip_TensorOrMemRef:$output,
    DefaultValuedAttr<I64Attr, "1">:$keepdims,
    DefaultValuedAttr<I64Attr, "0">:$noop_with_empty_axes
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $data `,` $axes `:` type($data) `,` type($axes) `)`
    `outs` `(` $output `:` type($output) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_LessOp : Hip_DpsOp<"less"> {
  let summary = "Element-wise less-than comparison: c = (a < b)";
  let description = [{
    Compares two input tensors element-wise. Output is a boolean tensor (i1)
    with the broadcast shape. Implements the standard ONNX Less operator.

    Supports multidirectional (NumPy-style) broadcasting.

    Uses destination-passing style: output buffer is provided as argument.

    Example:
    ```mlir
    hip.less(%ctx) ins(%a, %b : memref<3x4xf32, 1>, memref<3x4xf32, 1>)
                   outs(%c : memref<3x4xi1, 1>)
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$lhs,
    Hip_TensorOrMemRef:$rhs,
    Hip_TensorOrMemRef:$output
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $lhs `,` $rhs `:` type($lhs) `,` type($rhs) `)`
    `outs` `(` $output `:` type($output) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_SliceOp : Hip_DpsOp<"slice", [AttrSizedOperandSegments]> {
  let summary = "Slice a tensor along multiple axes (ONNX Slice fallback)";
  let description = [{
    Native fallback for ONNX Slice when the OnnxToHip pass cannot decompose
    the op to `tensor.extract_slice` (e.g. non-constant slice parameters or
    negative `steps`). Implements the standard ONNX Slice operator semantics:

      output[k] = data[ start[k0]:end[k0]:step[k0],
                         start[k1]:end[k1]:step[k1], ... ]

    `starts` / `ends` are 1-D integer tensors (int32 or int64), one entry per
    axis listed in `axes`. `axes` defaults to `[0, ..., r-1]` when absent and
    `steps` defaults to all-ones. Per-axis negative indices and negative
    steps follow ONNX clamping rules.

    The common case (compile-time constant indices, positive unit stride) is
    handled by an upstream decompose pattern that lowers `onnx.Slice` to
    `tensor.extract_slice` — that path produces a zero-cost view without any
    runtime call. This op exists only to give the fallback path a stable
    DPS-shaped representation to bufferize and lower from.

    Uses destination-passing style: output buffer is provided as argument.

    Example:
    ```mlir
    hip.slice(%ctx) ins(%data, %starts, %ends : memref<3x4xf32, 1>,
                                                  memref<2xi64, 1>,
                                                  memref<2xi64, 1>)
                    axes(%axes : memref<2xi64, 1>)
                    steps(%steps : memref<2xi64, 1>)
                    outs(%out : memref<3x2xf32, 1>)
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$data,
    Hip_TensorOrMemRef:$starts,
    Hip_TensorOrMemRef:$ends,
    Optional<Hip_TensorOrMemRef>:$axes,
    Optional<Hip_TensorOrMemRef>:$steps,
    Hip_TensorOrMemRef:$output
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $data `,` $starts `,` $ends `:`
        type($data) `,` type($starts) `,` type($ends) `)`
    (`axes` `(` $axes^ `:` type($axes) `)`)?
    (`steps` `(` $steps^ `:` type($steps) `)`)?
    `outs` `(` $output `:` type($output) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_GatherNDOp : Hip_DpsOp<"gather_nd"> {
  let summary = "GatherND: gather slices using N-D index tuples";
  let description = [{
    Gathers slices of `data` indexed by the trailing dimension of `indices`.
    Implements the standard ONNX GatherND operator.

    `indices` is an integer tensor whose last dimension defines the rank of
    each index tuple, optionally with `batch_dims` leading batch dims that
    must match `data`. Output rank is
    `q + r - indices_shape[-1] - 1 - batch_dims`, where `q = rank(indices)`
    and `r = rank(data)`.

    Uses destination-passing style: output buffer is provided as argument.

    Example:
    ```mlir
    hip.gather_nd(%ctx) ins(%data, %indices : memref<2x2xf32, 1>, memref<2x2xi64, 1>)
                        outs(%out : memref<2xf32, 1>) {batch_dims = 0 : i64}
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$data,
    Hip_TensorOrMemRef:$indices,
    Hip_TensorOrMemRef:$output,
    DefaultValuedAttr<I64Attr, "0">:$batch_dims
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $data `,` $indices `:` type($data) `,` type($indices) `)`
    `outs` `(` $output `:` type($output) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_ScatterNDOp : Hip_DpsOp<"scatter_nd"> {
  let summary = "ScatterND: scatter updates into a data tensor using N-D index tuples";
  let description = [{
    Implements the standard ONNX ScatterND operator: produce an `output`
    tensor with the same shape as `data` whose values are `data` copied,
    then `updates` written back at positions specified by `indices`.

    `indices` is an integer tensor whose last dimension `k` (with
    `k <= rank(data)`) defines the rank of each index tuple. When
    `k == rank(data)` each update entry is a scalar; otherwise each entry
    is a contiguous slice of trailing shape `data.shape[k:]`.

    `reduction` selects how an existing value at the target position is
    combined with the new update value:

      * `"none"` (default) — overwrite (`a = b`)
      * `"add"` — accumulate (`a = a + b`)
      * `"mul"` — multiply (`a = a * b`)
      * `"min"` — minimum (`a = min(a, b)`)
      * `"max"` — maximum (`a = max(a, b)`)

    This is the inverse of GatherND.

    Uses destination-passing style: the `output` buffer is provided as an
    argument, and the runtime is responsible for both the initial
    `data → output` copy and the subsequent scatter writes.

    Example:
    ```mlir
    hip.scatter_nd(%ctx) ins(%data, %indices, %updates :
                              memref<4x4x4xf32, 1>,
                              memref<2x1xi64, 1>,
                              memref<2x4x4xf32, 1>)
                          outs(%out : memref<4x4x4xf32, 1>)
                          {reduction = "none"}
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$data,
    Hip_TensorOrMemRef:$indices,
    Hip_TensorOrMemRef:$updates,
    Hip_TensorOrMemRef:$output,
    DefaultValuedStrAttr<StrAttr, "none">:$reduction
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $data `,` $indices `,` $updates `:`
        type($data) `,` type($indices) `,` type($updates) `)`
    `outs` `(` $output `:` type($output) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_SignOp : Hip_DpsOp<"sign"> {
  let summary = "Element-wise sign: y = sign(x)";
  let description = [{
    Computes the element-wise sign of the input. Returns +1 / 0 / -1 for
    positive / zero / negative elements respectively. Implements the standard
    ONNX Sign operator.

    Lowered via the unified unary elementwise path (wrap_sign).

    Uses destination-passing style: output buffer is provided as argument.

    Example:
    ```mlir
    hip.sign(%ctx) ins(%x : memref<3x4xf32, 1>)
                   outs(%y : memref<3x4xf32, 1>)
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$x,
    Hip_TensorOrMemRef:$y
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $x `:` type($x) `)`
    `outs` `(` $y `:` type($y) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_ModOp : Hip_DpsOp<"mod"> {
  let summary = "Element-wise modulo / fmod operation";
  let description = [{
    Performs element-wise binary modulo. Implements the standard ONNX Mod
    operator. The semantics depend on the `fmod` attribute:

    * `fmod = 0` (default): integer modulo; sign follows divisor (Python `%`).
    * `fmod = 1`: floating-point fmod; sign follows dividend (C `fmod`).

    Supports multidirectional (NumPy-style) broadcasting.

    Uses destination-passing style: output buffer is provided as argument.

    Example:
    ```mlir
    hip.mod(%ctx) ins(%a, %b : memref<3x4xi32, 1>, memref<3x4xi32, 1>)
                  outs(%c : memref<3x4xi32, 1>) {fmod = 0 : i64}
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$lhs,
    Hip_TensorOrMemRef:$rhs,
    Hip_TensorOrMemRef:$output,
    DefaultValuedAttr<I64Attr, "0">:$fmod
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $lhs `,` $rhs `:` type($lhs) `,` type($rhs) `)`
    `outs` `(` $output `:` type($output) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_SizeOp : Hip_DpsOp<"size"> {
  let summary = "Total element count of a tensor (ONNX Size, dynamic-shape path)";
  let description = [{
    Implements the runtime path of the standard ONNX Size operator
    (opset 1+) for inputs whose shape is not fully static.

    Statically-shaped Size ops are folded to a single `arith.constant` at
    OnnxToHip time (see SizeConversion.cpp::SizeToConstant), so this op
    is only emitted when at least one input dimension is `?`. The
    HipToLLVM lowering computes `prod(input.shape)` as a runtime i64
    value (using compile-time constants for static dims and descriptor
    `sizes[]` for dynamic dims) and stores it into the rank-0 i64
    output buffer.

    Input  x: ranked tensor / memref of any element type, any rank,
              with at least one dynamic dim.
    Output y: rank-0 i64 tensor / memref (`tensor<i64>` / `memref<i64, 1>`).

    Uses destination-passing style: the output buffer is provided as
    argument.

    Example:
    ```mlir
    hip.size(%ctx) ins(%x : memref<?x?xf16, 1>)
                    outs(%y : memref<i64, 1>)
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$x,
    Hip_TensorOrMemRef:$y
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $x `:` type($x) `)`
    `outs` `(` $y `:` type($y) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_NonZeroOp : Hip_DpsOp<"nonzero"> {
  let summary = "Indices of non-zero elements (ONNX NonZero)";
  let description = [{
    Implements the standard ONNX NonZero operator (opset 13+):
    returns the indices of the elements that are non-zero in row-major
    order (by dimension), behaving similarly to `numpy.nonzero`.

    Input `x` is a ranked tensor of shape `[D0, D1, ..., D(R-1)]`. Output
    `y` has shape `[R, N]` of element type `i64`, where `R` is the input
    rank and `N` is the (data-dependent) number of non-zero elements.
    The `N` dimension is represented as `?` in MLIR; the compiler
    over-allocates using `numel(x)` as the upper bound (since the input
    has fully static shape).

    Uses destination-passing style: the output buffer is provided as
    argument.

    Example:
    ```mlir
    hip.nonzero(%ctx) ins(%x : memref<3x4xf16, 1>)
                      outs(%y : memref<2x?xi64, 1>)
                      {input_data_type = 1 : i64}
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$x,
    Hip_TensorOrMemRef:$y,
    I64Attr:$input_data_type
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $x `:` type($x) `)`
    `outs` `(` $y `:` type($y) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}
#endif // HIP_OPS
