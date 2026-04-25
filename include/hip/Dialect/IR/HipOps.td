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
    I64Attr:$accuracy_level
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
    Implements the com.microsoft QMoE operator (inputs 7-14): quantized
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
  let summary = "N-D transpose swapping two specified dimensions";
  let arguments = (ins
    Hip_ContextType:$ctx,
    Index:$dim0,
    Index:$dim1,
    Hip_TensorOrMemRef:$input,
    Hip_TensorOrMemRef:$output
  );
  // result_tensors inherited from Hip_DpsOp
  let hasCustomAssemblyFormat = 1;
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

// ===== Generic Element-wise Unary ============================================
// Backed by a single HIP custom kernel that dispatches by `kind`.
// Used for ONNX Sin / Cos / Exp / Tanh / Floor / Round / Atan / LeakyRelu /
// Clip.  alpha and beta carry op-specific scalars (LeakyRelu: alpha = slope;
// Clip: alpha = min, beta = max; ignored for the rest).
//
// Kind values must match hip_unary_kind_t in
// 3rd-party/custom_kernels/include/hip_custom_kernels.h:
//   0=Sin, 1=Cos, 2=Exp, 3=Tanh, 4=Floor, 5=Round, 6=Atan, 7=LeakyRelu,
//   8=Clip
def Hip_UnaryElementwiseOp : Hip_DpsOp<"unary_elementwise"> {
  let summary = "Generic element-wise unary op (Sin/Cos/Exp/Tanh/Floor/Round/Atan/LeakyRelu/Clip)";
  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$x,
    Hip_TensorOrMemRef:$y,
    I64Attr:$kind,
    DefaultValuedAttr<F32Attr, "0.0">:$alpha,
    DefaultValuedAttr<F32Attr, "0.0">:$beta
  );
  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $x `:` type($x) `)`
    `outs` `(` $y `:` type($y) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

// ===== Generic Element-wise Binary (with broadcasting) =======================
// Used for ONNX Div and Pow.  Broadcasting is computed at lowering time
// from lhs/rhs/output shapes.
//
// Kind values must match hip_binary_kind_t:
//   0=Div, 1=Pow
def Hip_BinaryElementwiseOp : Hip_DpsOp<"binary_elementwise"> {
  let summary = "Generic element-wise binary op (Div/Pow) with broadcasting";
  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$lhs,
    Hip_TensorOrMemRef:$rhs,
    Hip_TensorOrMemRef:$output,
    I64Attr:$kind
  );
  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $lhs `,` $rhs `:` type($lhs) `,` type($rhs) `)`
    `outs` `(` $output `:` type($output) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

// ===== Strided Slice =========================================================
// ONNX Slice (opset 13).  The `starts`, `ends`, `steps` arrays are normalised
// at conversion time into per-axis (start_offset, step) carried as i64 array
// attributes.  Negative ends, omitted axes, and step != 1 are all resolved
// before reaching the runtime.
def Hip_SliceOp : Hip_DpsOp<"slice"> {
  let summary = "Strided slice (ONNX Slice)";
  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$input,
    Hip_TensorOrMemRef:$output,
    // Per-axis input start offset (already step-corrected).  Length == output
    // rank.
    I64ArrayAttr:$starts,
    // Per-axis step. Length == output rank.  step != 1 is supported.
    I64ArrayAttr:$steps
  );
  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $input `:` type($input) `)`
    `outs` `(` $output `:` type($output) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

// ===== Tier 2 shape / data movement ops ======================================

def Hip_ReduceMeanOp : Hip_DpsOp<"reduce_mean"> {
  let summary = "Mean reduction over the innermost reduce_size elements";
  let description = [{
    Reduces `data` to `output` by averaging contiguous slabs of
    `num_input_elements / num_output_elements` elements.  The conversion pass
    is responsible for ensuring the reduced axes are innermost.
  }];
  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$data,
    Hip_TensorOrMemRef:$output
  );
  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $data `:` type($data) `)`
    `outs` `(` $output `:` type($output) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_ConcatOp : Hip_DpsOp<"concat"> {
  let summary = "Concatenate variadic inputs along a single axis";
  let description = [{
    Concatenates `inputs` (variadic) along the integer attribute `axis`.  The
    conversion pass already validated that every input has identical shapes
    on every non-axis dimension and computed the output type accordingly.
  }];
  let arguments = (ins
    Hip_ContextType:$ctx,
    Variadic<Hip_TensorOrMemRef>:$inputs,
    Hip_TensorOrMemRef:$output,
    I64Attr:$axis
  );
  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $inputs `:` type($inputs) `)`
    `outs` `(` $output `:` type($output) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

// ===== Tier 3 / 4 ops (compare / select / layernorm) =========================

// Element-wise comparison / logical op producing a bool (i8) output.
// Kind values match hip_compare_kind_t in hip_custom_kernels.h:
//   0=Equal, 1=Greater, 2=Less, 3=GreaterOrEqual, 4=And
def Hip_CompareOp : Hip_DpsOp<"compare"> {
  let summary = "Element-wise compare/logical op (Equal/Greater/Less/GE/And)";
  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$lhs,
    Hip_TensorOrMemRef:$rhs,
    Hip_TensorOrMemRef:$output,
    I64Attr:$kind
  );
  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $lhs `,` $rhs `:` type($lhs) `,` type($rhs) `)`
    `outs` `(` $output `:` type($output) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_WhereOp : Hip_DpsOp<"where"> {
  let summary = "Element-wise select: cond ? x : y";
  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$cond,
    Hip_TensorOrMemRef:$x,
    Hip_TensorOrMemRef:$y,
    Hip_TensorOrMemRef:$output
  );
  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $cond `,` $x `,` $y `:`
        type($cond) `,` type($x) `,` type($y) `)`
    `outs` `(` $output `:` type($output) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_LayerNormOp : Hip_DpsOp<"layer_norm"> {
  let summary = "ONNX LayerNormalization (with mean subtraction)";
  let description = [{
    Computes y = (x - mean) / sqrt(var + epsilon) * gamma + beta along the
    innermost `norm_size` elements of each row.  beta is optional (treated
    as zero when absent).  See SimplifiedLayerNormalization for the
    epsilon-only/RMS variant.
  }];
  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$input,
    Hip_TensorOrMemRef:$gamma,
    Optional<Hip_TensorOrMemRef>:$beta,
    Hip_TensorOrMemRef:$output,
    F32Attr:$epsilon,
    I64Attr:$axis
  );
  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(`
        $input `,` $gamma (`,` $beta^)?
        `:` type($input) `,` type($gamma) (`,` type($beta)^)?
    `)`
    `outs` `(` $output `:` type($output) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_ConstantOfShapeOp : Hip_DpsOp<"constant_of_shape"> {
  let summary = "Fill a buffer with a scalar (ONNX ConstantOfShape)";
  let description = [{
    Writes `scalar_bits` (a 64-bit blob containing the bit pattern of the
    scalar fill value, narrowed to the output element type) into every
    element of `output`.  The output tensor type encodes the element width.
  }];
  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$output,
    I64Attr:$scalar_bits
  );
  let assemblyFormat = [{
    `(` $ctx `)`
    `outs` `(` $output `:` type($output) `)`
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
#endif // HIP_OPS
