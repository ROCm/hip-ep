/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef HIP_OPS
#define HIP_OPS

include "hip/Dialect/IR/HipDialect.td"
include "hip/Dialect/IR/HipTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/IR/OpBase.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"

class Hip_Op<string mnemonic, list<Trait> traits = []> :
    Op<Hip_Dialect, mnemonic, traits>;

// Type constraint that accepts either a ranked tensor or a memref.
// Used for data operands and DPS init operands on all compute ops so that
// ops are valid at both the tensor level (pre-bufferization) and the memref
// level (post-bufferization).
def Hip_TensorOrMemRef : AnyTypeOf<[AnyRankedTensor, AnyMemRef]>;

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

def Hip_GetPoolOp : Hip_Op<"get_pool", [
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Get the base pointer of the GPU memory pool";
  let description = [{
    Returns a raw byte-addressable view of the per-inference GPU memory pool
    allocated by the runtime.  The pool is obtained via
    `hipdnn_ep_get_pool_base(state)` and returned as `memref<?xi8, 1>`.

    After memory-pooling analysis, every `memref.alloc` is replaced by a
    `memref.view` of this pool at a statically computed offset.

    Example:
    ```mlir
    %pool = hip.get_pool(%ctx) : memref<?xi8, 1>
    %buf  = memref.view %pool[%offset][] : memref<?xi8, 1> to memref<1x64x112x112xf32, 1>
    ```
  }];

  let arguments = (ins Hip_ContextType:$ctx);
  let results   = (outs AnyMemRef:$pool);

  let assemblyFormat = [{
    `(` $ctx `)` attr-dict `:` type($pool)
  }];
}

//===----------------------------------------------------------------------===//
// MIOpen DNN Operations
//===----------------------------------------------------------------------===//

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
    hip.conv(%handle) ins(%input, %weights, %bias : memref<1x3x224x224xf32, 1>,
                                                    memref<64x3x3x3xf32, 1>,
                                                    memref<64xf32, 1>)
                      outs(%output : memref<1x64x112x112xf32, 1>)
                      {kernel_shape = [3, 3], strides = [1, 1],
                       pads = [1, 1, 1, 1], dilations = [1, 1], group = 1}
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$handle,
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
    `(` $handle `)` `ins` `(` $input `,` $weights (`,` $bias^)? `:`
    type($input) `,` type($weights) (`,` type($bias)^)? `)`
    `outs` `(` $output `:` type($output) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_GemmOp : Hip_DpsOp<"gemm"> {
  let summary = "General matrix multiplication using hipBLASLt (in-place)";
  let description = [{
    Performs matrix multiplication: result = alpha * A * B + beta * C
    using hipBLASLt (miopenGEMM or hipblasLtMatmul).

    Uses destination-passing style: output buffer is provided as argument.

    Attributes:
    - transA: Whether to transpose matrix A (0 or 1)
    - transB: Whether to transpose matrix B (0 or 1)
    - alpha: Scalar multiplier for A*B (default 1.0)
    - beta: Scalar multiplier for C (default 0.0, if C is provided)

    Example:
    ```mlir
    hip.gemm(%handle) ins(%A, %B, %C : memref<MxKxf32, 1>,
                                       memref<KxNxf32, 1>,
                                       memref<MxNxf32, 1>)
                      outs(%output : memref<MxNxf32, 1>)
                      {transA = 0, transB = 0, alpha = 1.0, beta = 1.0}
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$handle,
    Hip_TensorOrMemRef:$A,
    Hip_TensorOrMemRef:$B,
    Hip_TensorOrMemRef:$C,
    Hip_TensorOrMemRef:$output,
    I64Attr:$transA,
    I64Attr:$transB,
    F32Attr:$alpha,
    F32Attr:$beta
  );

  let assemblyFormat = [{
    `(` $handle `)` `ins` `(` $A `,` $B `,` $C `:` type($A) `,` type($B) `,` type($C) `)`
    `outs` `(` $output `:` type($output) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_MaxPoolOp : Hip_DpsOp<"maxpool"> {
  let summary = "Max pooling operation using MIOpen (in-place)";
  let description = [{
    Performs max pooling using MIOpen (miopenPoolingForward with max mode).

    Uses destination-passing style: output buffer is provided as argument.

    Attributes:
    - kernel_shape: Array of pooling window dimensions [H, W]
    - strides: Array of stride values [H, W]
    - pads: Array of padding values [top, left, bottom, right]

    Example:
    ```mlir
    hip.maxpool(%handle) ins(%input : memref<1x64x112x112xf32, 1>)
                         outs(%output : memref<1x64x56x56xf32, 1>)
                         {kernel_shape = [2, 2], strides = [2, 2], pads = [0, 0, 0, 0]}
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$handle,
    Hip_TensorOrMemRef:$input,
    Hip_TensorOrMemRef:$output,
    I64ArrayAttr:$kernel_shape,
    I64ArrayAttr:$strides,
    I64ArrayAttr:$pads
  );

  let assemblyFormat = [{
    `(` $handle `)` `ins` `(` $input `:` type($input) `)`
    `outs` `(` $output `:` type($output) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_AvgPoolOp : Hip_DpsOp<"avgpool"> {
  let summary = "Average pooling operation using MIOpen (in-place)";
  let description = [{
    Performs average pooling using MIOpen (miopenPoolingForward with average mode).

    Uses destination-passing style: output buffer is provided as argument.

    Attributes:
    - kernel_shape: Array of pooling window dimensions [H, W]
    - strides: Array of stride values [H, W]
    - pads: Array of padding values [top, left, bottom, right]

    Example:
    ```mlir
    hip.avgpool(%handle) ins(%input : memref<1x64x112x112xf32, 1>)
                         outs(%output : memref<1x64x56x56xf32, 1>)
                         {kernel_shape = [2, 2], strides = [2, 2], pads = [0, 0, 0, 0]}
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$handle,
    Hip_TensorOrMemRef:$input,
    Hip_TensorOrMemRef:$output,
    I64ArrayAttr:$kernel_shape,
    I64ArrayAttr:$strides,
    I64ArrayAttr:$pads
  );

  let assemblyFormat = [{
    `(` $handle `)` `ins` `(` $input `:` type($input) `)`
    `outs` `(` $output `:` type($output) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

//===----------------------------------------------------------------------===//
// Activation Operations
//===----------------------------------------------------------------------===//

def Hip_ReluOp : Hip_DpsOp<"relu"> {
  let summary = "ReLU activation";
  let description = [{
    Applies ReLU activation: output = max(0, input)
    Uses MIOpen's miopenActivationForward with MIOPEN_ACTIVATION_RELU mode.

    Uses destination-passing style: output buffer is provided as argument.

    Example:
    ```mlir
    hip.relu(%handle) ins(%input : memref<1x64x224x224xf32, 1>)
                      outs(%output : memref<1x64x224x224xf32, 1>)
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$handle,
    Hip_TensorOrMemRef:$input,
    Hip_TensorOrMemRef:$output
  );

  let assemblyFormat = [{
    `(` $handle `)` `ins` `(` $input `:` type($input) `)`
    `outs` `(` $output `:` type($output) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_CastOp : Hip_DpsOp<"cast"> {
  let summary = "Element type cast";
  let description = [{
    Casts tensor elements from one data type to another.
    Input and output may have different element types but the same shape.

    Uses destination-passing style: output buffer is provided as argument.

    Example:
    ```mlir
    hip.cast(%handle) ins(%input : memref<i64, 1>)
                      outs(%output : memref<i32, 1>) {to = 6 : i64}
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$handle,
    Hip_TensorOrMemRef:$input,
    Hip_TensorOrMemRef:$output,
    I64Attr:$to
  );

  let assemblyFormat = [{
    `(` $handle `)` `ins` `(` $input `:` type($input) `)`
    `outs` `(` $output `:` type($output) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

//===----------------------------------------------------------------------===//
// Constant Management Operations
//===----------------------------------------------------------------------===//

def Hip_GetConstantOp : Hip_Op<"get_constant", [
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Retrieve constant from GPU memory";
  let description = [{
    Retrieves a pre-uploaded constant from state->gpu_constants[index].

    This operation is used in inference functions to access constants that
    were uploaded during initialization.

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

//===----------------------------------------------------------------------===//
// MatMul Operation
//===----------------------------------------------------------------------===//

def Hip_MatMulOp : Hip_DpsOp<"matmul"> {
  let summary = "Matrix multiplication (supports batched N-D x N-D)";
  let description = [{
    Performs matrix multiplication: output = A @ B
    Supports batched operation (leading dimensions are batch dimensions).

    Uses destination-passing style: output buffer is provided as argument.

    Unlike hip.gemm, this has no alpha/beta/transA/transB/bias — it maps
    directly to onnx.MatMul semantics.

    Example (2D):
    ```mlir
    hip.matmul(%handle) ins(%A, %B : memref<128x4096xf32, 1>,
                                     memref<4096x1024xf32, 1>)
                        outs(%output : memref<128x1024xf32, 1>)
    ```

    Example (3D batched):
    ```mlir
    hip.matmul(%handle) ins(%A, %B : memref<1x128x4096xf16, 1>,
                                     memref<4096x1024xf16, 1>)
                        outs(%output : memref<1x128x1024xf16, 1>)
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$handle,
    Hip_TensorOrMemRef:$A,
    Hip_TensorOrMemRef:$B,
    Hip_TensorOrMemRef:$output
  );

  let assemblyFormat = [{
    `(` $handle `)` `ins` `(` $A `,` $B `:` type($A) `,` type($B) `)`
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

//===----------------------------------------------------------------------===//
// Attention Operations
//===----------------------------------------------------------------------===//

def Hip_GroupQueryAttentionOp : Hip_DpsOp<"group_query_attention"> {
  let summary = "Group Query Attention (com.microsoft GQA)";
  let description = [{
    Multi-head attention with grouped key/value heads (GQA/MQA).
    Implements the com.microsoft GroupQueryAttention operator for LLM inference.

    Concatenates new key/value with past KV cache and computes scaled
    dot-product attention with grouped heads.

    Uses destination-passing style: output buffers are provided as arguments.

    Operand layout (for getDpsInitsMutable): operands 0..7 are inputs
    (handle, query, key, value, past_key, past_value, seqlens_k, total_seq_len);
    operands 8..10 are DPS inits (output, present_key, present_value).

    Attributes:
    - num_heads: Number of query attention heads
    - kv_num_heads: Number of key/value heads (< num_heads for GQA)
    - scale: Attention score scaling factor (1/sqrt(head_dim))
    - softcap: Soft capping value for attention logits (0 = disabled)
    - do_rotary: Apply rotary positional embedding (0 or 1)
    - rotary_interleaved: Rotary embedding interleaved format (0 or 1)

    Example:
    ```mlir
    hip.group_query_attention(%ctx)
        ins(%query, %key, %value, %past_key, %past_value, %seqlens_k, %total_seq_len :
            memref<1x1x4096xf16, 1>, ...)
        outs(%output, %present_key, %present_value : ...)
        {num_heads = 32, kv_num_heads = 8, scale = 0.0884,
         softcap = 0.0, do_rotary = 0, rotary_interleaved = 0}
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$handle,
    Hip_TensorOrMemRef:$query,
    Hip_TensorOrMemRef:$key,
    Hip_TensorOrMemRef:$value,
    Hip_TensorOrMemRef:$past_key,
    Hip_TensorOrMemRef:$past_value,
    Hip_TensorOrMemRef:$seqlens_k,
    Hip_TensorOrMemRef:$total_seq_len,
    Hip_TensorOrMemRef:$output,
    Hip_TensorOrMemRef:$present_key,
    Hip_TensorOrMemRef:$present_value,
    I64Attr:$num_heads,
    I64Attr:$kv_num_heads,
    F32Attr:$scale,
    F32Attr:$softcap,
    I64Attr:$do_rotary,
    I64Attr:$rotary_interleaved
  );

  let assemblyFormat = [{
    `(` $handle `)` `ins` `(`
        $query `,` $key `,` $value `,`
        $past_key `,` $past_value `,` $seqlens_k `,` $total_seq_len `:`
        type($query) `,` type($key) `,` type($value) `,`
        type($past_key) `,` type($past_value) `,` type($seqlens_k) `,` type($total_seq_len)
    `)`
    `outs` `(` $output `,` $present_key `,` $present_value `:`
              type($output) `,` type($present_key) `,` type($present_value) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

//===----------------------------------------------------------------------===//
// Elementwise Operations
//===----------------------------------------------------------------------===//

def Hip_MulOp : Hip_DpsOp<"mul"> {
  let summary = "Elementwise multiplication";
  let description = [{
    Performs elementwise multiplication: output = lhs * rhs
    Both inputs must have the same shape (broadcasting handled before lowering).

    Uses destination-passing style: output buffer is provided as argument.

    Example:
    ```mlir
    hip.mul(%handle) ins(%lhs, %rhs : memref<1x128x14336xf16, 1>,
                                      memref<1x128x14336xf16, 1>)
                     outs(%output : memref<1x128x14336xf16, 1>)
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$handle,
    Hip_TensorOrMemRef:$lhs,
    Hip_TensorOrMemRef:$rhs,
    Hip_TensorOrMemRef:$output
  );

  let assemblyFormat = [{
    `(` $handle `)` `ins` `(` $lhs `,` $rhs `:` type($lhs) `,` type($rhs) `)`
    `outs` `(` $output `:` type($output) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_SubOp : Hip_DpsOp<"sub"> {
  let summary = "Elementwise subtraction";
  let description = [{
    Performs elementwise subtraction: output = lhs - rhs
    Both inputs must have the same shape (broadcasting handled before lowering).

    Uses destination-passing style: output buffer is provided as argument.

    Example:
    ```mlir
    hip.sub(%handle) ins(%lhs, %rhs : memref<1x1xi64, 1>, memref<1x1xi64, 1>)
                     outs(%output : memref<1x1xi64, 1>)
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$handle,
    Hip_TensorOrMemRef:$lhs,
    Hip_TensorOrMemRef:$rhs,
    Hip_TensorOrMemRef:$output
  );

  let assemblyFormat = [{
    `(` $handle `)` `ins` `(` $lhs `,` $rhs `:` type($lhs) `,` type($rhs) `)`
    `outs` `(` $output `:` type($output) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_SigmoidOp : Hip_DpsOp<"sigmoid"> {
  let summary = "Sigmoid activation";
  let description = [{
    Applies sigmoid activation: output = 1 / (1 + exp(-input))

    Uses destination-passing style: output buffer is provided as argument.

    Example:
    ```mlir
    hip.sigmoid(%handle) ins(%input : memref<1x128x14336xf16, 1>)
                         outs(%output : memref<1x128x14336xf16, 1>)
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$handle,
    Hip_TensorOrMemRef:$input,
    Hip_TensorOrMemRef:$output
  );

  let assemblyFormat = [{
    `(` $handle `)` `ins` `(` $input `:` type($input) `)`
    `outs` `(` $output `:` type($output) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

//===----------------------------------------------------------------------===//
// Data Movement Operations
//===----------------------------------------------------------------------===//

def Hip_GatherOp : Hip_DpsOp<"gather"> {
  let summary = "Gather elements along an axis";
  let description = [{
    Gathers elements from data along the given axis using indices.
    Implements the ONNX Gather operator.

    Uses destination-passing style: output buffer is provided as argument.

    Example:
    ```mlir
    hip.gather(%handle) ins(%data, %indices : memref<2xi64, 1>, memref<i64, 1>)
                        outs(%output : memref<i64, 1>) {axis = 0 : i64}
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$handle,
    Hip_TensorOrMemRef:$data,
    Hip_TensorOrMemRef:$indices,
    Hip_TensorOrMemRef:$output,
    I64Attr:$axis
  );

  let assemblyFormat = [{
    `(` $handle `)` `ins` `(` $data `,` $indices `:` type($data) `,` type($indices) `)`
    `outs` `(` $output `:` type($output) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

//===----------------------------------------------------------------------===//
// Reduction Operations
//===----------------------------------------------------------------------===//

def Hip_ReduceSumOp : Hip_DpsOp<"reduce_sum"> {
  let summary = "Reduce tensor by summing along axes";
  let description = [{
    Computes the sum of input tensor elements along the specified axes.
    Implements the ONNX ReduceSum operator (opset 13+).

    Uses destination-passing style: output buffer is provided as argument.

    Example:
    ```mlir
    hip.reduce_sum(%handle) ins(%data, %axes : memref<1x128xi64, 1>, memref<i64, 1>)
                            outs(%output : memref<1x1xi64, 1>) {keepdims = 1 : i64}
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$handle,
    Hip_TensorOrMemRef:$data,
    Hip_TensorOrMemRef:$axes,
    Hip_TensorOrMemRef:$output,
    I64Attr:$keepdims
  );

  let assemblyFormat = [{
    `(` $handle `)` `ins` `(` $data `,` $axes `:` type($data) `,` type($axes) `)`
    `outs` `(` $output `:` type($output) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_RotaryEmbeddingOp : Hip_DpsOp<"rotary_embedding"> {
  let summary = "Rotary position embedding (RoPE)";
  let description = [{
    Applies rotary position embedding to the input tensor.
    Implements the Microsoft RotaryEmbedding operator.

    Uses destination-passing style: output buffer is provided as argument.

    Example:
    ```mlir
    hip.rotary_embedding(%handle)
        ins(%input, %position_ids, %cos_cache, %sin_cache :
            memref<1x128x4096xf16, 1>, memref<1x128xi64, 1>,
            memref<131072x64xf16, 1>, memref<131072x64xf16, 1>)
        outs(%output : memref<1x128x4096xf16, 1>)
        {interleaved = 0 : i64, num_heads = 0 : i64, rotary_embedding_dim = 0 : i64}
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$handle,
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
    `(` $handle `)` `ins` `(`
        $input `,` $position_ids `,` $cos_cache `,` $sin_cache `:`
        type($input) `,` type($position_ids) `,` type($cos_cache) `,` type($sin_cache)
    `)`
    `outs` `(` $output `:` type($output) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_SimplifiedLayerNormOp : Hip_DpsOp<"simplified_layer_norm"> {
  let summary = "Simplified layer normalization";
  let description = [{
    Applies simplified (RMS) layer normalization.
    output = input / sqrt(mean(input^2) + epsilon) * scale

    Uses destination-passing style: output buffer is provided as argument.

    Example:
    ```mlir
    hip.simplified_layer_norm(%handle)
        ins(%input, %scale : memref<1x128x4096xf16, 1>, memref<4096xf16, 1>)
        outs(%output : memref<1x128x4096xf16, 1>)
        {axis = -1 : i64, epsilon = 9.99999974e-06 : f32, stash_type = 1 : i64}
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$handle,
    Hip_TensorOrMemRef:$input,
    Hip_TensorOrMemRef:$scale,
    Hip_TensorOrMemRef:$output,
    I64Attr:$axis,
    F32Attr:$epsilon,
    I64Attr:$stash_type
  );

  let assemblyFormat = [{
    `(` $handle `)` `ins` `(` $input `,` $scale `:` type($input) `,` type($scale) `)`
    `outs` `(` $output `:` type($output) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_SkipSimplifiedLayerNormOp : Hip_DpsOp<"skip_simplified_layer_norm"> {
  let summary = "Skip connection + simplified layer normalization";
  let description = [{
    Adds a skip connection then applies simplified layer normalization.
    Implements the Microsoft SkipSimplifiedLayerNormalization operator.

    Has two outputs: the normalized result and the skip-sum for residual.

    Uses destination-passing style: output buffers are provided as arguments.

    Operand layout (for getDpsInitsMutable): operands 0..3 are inputs
    (handle, input, skip, gamma); operands 4..5 are DPS inits (output, skip_output).

    Example:
    ```mlir
    hip.skip_simplified_layer_norm(%handle)
        ins(%input, %skip, %gamma : memref<1x128x4096xf16, 1>,
                                    memref<1x128x4096xf16, 1>,
                                    memref<4096xf16, 1>)
        outs(%output, %skip_output : memref<1x128x4096xf16, 1>,
                                     memref<1x128x4096xf16, 1>)
        {epsilon = 9.99999974e-06 : f32}
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$handle,
    Hip_TensorOrMemRef:$input,
    Hip_TensorOrMemRef:$skip,
    Hip_TensorOrMemRef:$gamma,
    Hip_TensorOrMemRef:$output,
    Hip_TensorOrMemRef:$skip_output,
    F32Attr:$epsilon
  );

  let assemblyFormat = [{
    `(` $handle `)` `ins` `(` $input `,` $skip `,` $gamma `:`
                              type($input) `,` type($skip) `,` type($gamma) `)`
    `outs` `(` $output `,` $skip_output `:` type($output) `,` type($skip_output) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

#endif // HIP_OPS
