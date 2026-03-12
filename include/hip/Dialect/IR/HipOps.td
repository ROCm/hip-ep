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

def Hip_SkipRmsNormOp : Hip_DpsOp<"skip_rms_norm"> {
  let summary = "Fused Add + RMS normalization via miopenAddLayerNormForward (T5 mode)";
  let description = [{
    Performs fused skip connection addition followed by RMS normalization.
    Implements the Microsoft SkipSimplifiedLayerNormalization operator.

    This operation combines two steps:
      1. skip_output = input + skip  (residual connection)
      2. output = RMSNorm(skip_output) * gamma  (RMS normalization)

    Uses destination-passing style: output buffers are provided as arguments.

    The epsilon parameter is used for numerical stability in the normalization:
      RMSNorm(x) = x / sqrt(mean(x^2) + epsilon)

    Example:
    ```mlir
    hip.skip_rms_norm(%ctx)
        ins(%input, %skip, %gamma : memref<1x128x4096xf16, 1>,
                                     memref<1x128x4096xf16, 1>,
                                     memref<4096xf16, 1>)
        outs(%output, %skip_output : memref<1x128x4096xf16, 1>,
                                      memref<1x128x4096xf16, 1>)
        {epsilon = 1.0e-05 : f32}
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$input,
    Hip_TensorOrMemRef:$skip,
    Hip_TensorOrMemRef:$gamma,
    Hip_TensorOrMemRef:$output,
    Hip_TensorOrMemRef:$skip_output,
    F32Attr:$epsilon
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $input `,` $skip `,` $gamma `:` type($input) `,` type($skip) `,` type($gamma) `)`
    `outs` `(` $output `,` $skip_output `:` type($output) `,` type($skip_output) `)`
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
    I64Attr:$axis
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
    Applies sigmoid activation: output = 1 / (1 + exp(-input))

    Uses destination-passing style: output buffer is provided as argument.

    Example:
    ```mlir
    hip.sigmoid(%ctx) ins(%input : memref<1x128x512xf32, 1>)
                      outs(%output : memref<1x128x512xf32, 1>)
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$input,
    Hip_TensorOrMemRef:$output
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $input `:` type($input) `)`
    `outs` `(` $output `:` type($output) `)`
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

    Example:
    ```mlir
    hip.reduce_sum(%ctx) ins(%data, %axes : memref<1x128xi64, 1>, memref<i64, 1>)
                         outs(%output : memref<1x1xi64, 1>) {keepdims = 1 : i64}
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$data,
    Hip_TensorOrMemRef:$axes,
    Hip_TensorOrMemRef:$output,
    I64Attr:$keepdims
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $data `,` $axes `:` type($data) `,` type($axes) `)`
    `outs` `(` $output `:` type($output) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_GqaOp : Hip_DpsOp<"gqa"> {
  let summary = "Group Query Attention (com.microsoft GQA)";
  let description = [{
    Multi-head attention with grouped key/value heads (GQA/MQA).
    Implements the com.microsoft GroupQueryAttention operator for LLM inference.

    Concatenates new key/value with past KV cache and computes scaled
    dot-product attention with grouped heads.

    Uses destination-passing style: output buffers are provided as arguments.

    Operand layout (for getDpsInitsMutable): operands 0..7 are inputs
    (ctx, query, key, value, past_key, past_value, seqlens_k, total_seq_len);
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
    hip.gqa(%ctx)
        ins(%query, %key, %value, %past_key, %past_value, %seqlens_k, %total_seq_len :
            memref<1x1x4096xf16, 1>, ...)
        outs(%output, %present_key, %present_value : ...)
        {num_heads = 32, kv_num_heads = 8, scale = 0.0884,
         softcap = 0.0, do_rotary = 0, rotary_interleaved = 0}
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
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
    `(` $ctx `)` `ins` `(`
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

#endif // HIP_OPS
