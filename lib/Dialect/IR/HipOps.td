#ifndef HIP_OPS
#define HIP_OPS

include "HipDialect.td"
include "HipTypes.td"
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

def Hip_ConvOp : Hip_Op<"conv", [
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  DeclareOpInterfaceMethods<DestinationStyleOpInterface, ["getDpsInitsMutable"]>
]> {
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
    %output = hip.alloc(%handle) : memref<1x64x112x112xf32, 1>
    hip.conv(%handle, %input, %weights, %bias, %output)
             {kernel_shape = [3, 3], strides = [1, 1],
              pads = [1, 1, 1, 1], dilations = [1, 1], group = 1}
             : (!hip.context, memref<1x3x224x224xf32, 1>,
                memref<64x3x3x3xf32, 1>, memref<64xf32, 1>,
                memref<1x64x112x112xf32, 1>)
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
  let results = (outs Variadic<AnyRankedTensor>:$result_tensors);

  let assemblyFormat = [{
    `(` $handle `,` $input `,` $weights (`,` $bias^)? `,` $output `)` attr-dict
    `:` `(` type($handle) `,` type($input) `,` type($weights) (`,` type($bias)^)? `,` type($output) `)`
    (`:` type($result_tensors)^)?
  }];
}

def Hip_GemmOp : Hip_Op<"gemm", [
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  DeclareOpInterfaceMethods<DestinationStyleOpInterface, ["getDpsInitsMutable"]>
]> {
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
    %result = hip.alloc(%handle) : memref<MxNxf32, 1>
    hip.gemm(%handle, %A, %B, %C, %result)
            {transA = 0, transB = 0, alpha = 1.0, beta = 1.0}
            : (!hip.context, memref<MxKxf32, 1>, memref<KxNxf32, 1>, memref<MxNxf32, 1>, memref<MxNxf32, 1>)
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$handle,
    Hip_TensorOrMemRef:$A,
    Hip_TensorOrMemRef:$B,
    Hip_TensorOrMemRef:$C,
    Hip_TensorOrMemRef:$result,
    I64Attr:$transA,
    I64Attr:$transB,
    F32Attr:$alpha,
    F32Attr:$beta
  );
  let results = (outs Variadic<AnyRankedTensor>:$result_tensors);

  let assemblyFormat = [{
    `(` $handle `,` $A `,` $B `,` $C `,` $result `)` attr-dict
    `:` `(` type($handle) `,` type($A) `,` type($B) `,` type($C) `,` type($result) `)`
    (`:` type($result_tensors)^)?
  }];
}

def Hip_MaxPoolOp : Hip_Op<"maxpool", [
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  DeclareOpInterfaceMethods<DestinationStyleOpInterface, ["getDpsInitsMutable"]>
]> {
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
    %output = hip.alloc(%handle) : memref<1x64x56x56xf32, 1>
    hip.maxpool(%handle, %input, %output)
               {kernel_shape = [2, 2], strides = [2, 2], pads = [0, 0, 0, 0]}
               : (!hip.context, memref<1x64x112x112xf32, 1>, memref<1x64x56x56xf32, 1>)
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
  let results = (outs Variadic<AnyRankedTensor>:$result_tensors);

  let assemblyFormat = [{
    `(` $handle `,` $input `,` $output `)` attr-dict
    `:` `(` type($handle) `,` type($input) `,` type($output) `)`
    (`:` type($result_tensors)^)?
  }];
}

def Hip_AvgPoolOp : Hip_Op<"avgpool", [
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  DeclareOpInterfaceMethods<DestinationStyleOpInterface, ["getDpsInitsMutable"]>
]> {
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
    %output = hip.alloc(%handle) : memref<1x64x56x56xf32, 1>
    hip.avgpool(%handle, %input, %output)
               {kernel_shape = [2, 2], strides = [2, 2], pads = [0, 0, 0, 0]}
               : (!hip.context, memref<1x64x112x112xf32, 1>, memref<1x64x56x56xf32, 1>)
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
  let results = (outs Variadic<AnyRankedTensor>:$result_tensors);

  let assemblyFormat = [{
    `(` $handle `,` $input `,` $output `)` attr-dict
    `:` `(` type($handle) `,` type($input) `,` type($output) `)`
    (`:` type($result_tensors)^)?
  }];
}

//===----------------------------------------------------------------------===//
// Activation Operations
//===----------------------------------------------------------------------===//

def Hip_ReluOp : Hip_Op<"relu", [
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  DeclareOpInterfaceMethods<DestinationStyleOpInterface, ["getDpsInitsMutable"]>
]> {
  let summary = "ReLU activation";
  let description = [{
    Applies ReLU activation: output = max(0, input)
    Uses MIOpen's miopenActivationForward with MIOPEN_ACTIVATION_RELU mode.

    Uses destination-passing style: output buffer is provided as argument.

    Example:
    ```mlir
    %output = hip.alloc(%handle) : memref<1x64x224x224xf32, 1>
    hip.relu(%handle, %input, %output)
      : (!hip.context, memref<1x64x224x224xf32, 1>, memref<1x64x224x224xf32, 1>)
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$handle,
    Hip_TensorOrMemRef:$input,
    Hip_TensorOrMemRef:$output
  );
  let results = (outs Variadic<AnyRankedTensor>:$result_tensors);

  let assemblyFormat = [{
    `(` $handle `,` $input `,` $output `)` attr-dict
    `:` `(` type($handle) `,` type($input) `,` type($output) `)`
    (`:` type($result_tensors)^)?
  }];
}

def Hip_CastOp : Hip_Op<"cast", [
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  DeclareOpInterfaceMethods<DestinationStyleOpInterface, ["getDpsInitsMutable"]>
]> {
  let summary = "Element type cast";
  let description = [{
    Casts tensor elements from one data type to another.
    Input and output may have different element types but the same shape.

    Uses destination-passing style: output buffer is provided as argument.

    Example:
    ```mlir
    %output = hip.alloc(%handle) : memref<i32, 1>
    hip.cast(%handle, %input, %output) {to = 6 : i64}
      : (!hip.context, memref<i64, 1>, memref<i32, 1>)
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$handle,
    Hip_TensorOrMemRef:$input,
    Hip_TensorOrMemRef:$output,
    I64Attr:$to
  );
  let results = (outs Variadic<AnyRankedTensor>:$result_tensors);

  let assemblyFormat = [{
    `(` $handle `,` $input `,` $output `)` attr-dict
    `:` `(` type($handle) `,` type($input) `,` type($output) `)`
    (`:` type($result_tensors)^)?
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

def Hip_MatMulOp : Hip_Op<"matmul", [
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  DeclareOpInterfaceMethods<DestinationStyleOpInterface, ["getDpsInitsMutable"]>
]> {
  let summary = "Matrix multiplication (supports batched N-D x N-D)";
  let description = [{
    Performs matrix multiplication: output = A @ B
    Supports batched operation (leading dimensions are batch dimensions).

    Uses destination-passing style: output buffer is provided as argument.

    Unlike hip.gemm, this has no alpha/beta/transA/transB/bias — it maps
    directly to onnx.MatMul semantics.

    Example (2D):
    ```mlir
    %result = hip.alloc(%handle) : memref<128x1024xf32, 1>
    hip.matmul(%handle, %A, %B, %result)
      : (!hip.context, memref<128x4096xf32, 1>,
         memref<4096x1024xf32, 1>, memref<128x1024xf32, 1>)
    ```

    Example (3D batched):
    ```mlir
    %result = hip.alloc(%handle) : memref<1x128x1024xf16, 1>
    hip.matmul(%handle, %A, %B, %result)
      : (!hip.context, memref<1x128x4096xf16, 1>,
         memref<4096x1024xf16, 1>, memref<1x128x1024xf16, 1>)
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$handle,
    Hip_TensorOrMemRef:$A,
    Hip_TensorOrMemRef:$B,
    Hip_TensorOrMemRef:$result
  );
  let results = (outs Variadic<AnyRankedTensor>:$result_tensors);

  let assemblyFormat = [{
    `(` $handle `,` $A `,` $B `,` $result `)` attr-dict
    `:` `(` type($handle) `,` type($A) `,` type($B) `,` type($result) `)`
    (`:` type($result_tensors)^)?
  }];
}

//===----------------------------------------------------------------------===//
// Attention Operations
//===----------------------------------------------------------------------===//

def Hip_GroupQueryAttentionOp : Hip_Op<"group_query_attention", [
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  DeclareOpInterfaceMethods<DestinationStyleOpInterface, ["getDpsInitsMutable"]>
]> {
  let summary = "Group Query Attention (com.microsoft GQA)";
  let description = [{
    Multi-head attention with grouped key/value heads (GQA/MQA).
    Implements the com.microsoft GroupQueryAttention operator for LLM inference.

    Concatenates new key/value with past KV cache and computes scaled
    dot-product attention with grouped heads.

    Uses destination-passing style: output buffers are provided as arguments.

    Attributes:
    - num_heads: Number of query attention heads
    - kv_num_heads: Number of key/value heads (< num_heads for GQA)
    - scale: Attention score scaling factor (1/sqrt(head_dim))
    - softcap: Soft capping value for attention logits (0 = disabled)
    - do_rotary: Apply rotary positional embedding (0 or 1)
    - rotary_interleaved: Rotary embedding interleaved format (0 or 1)

    Example:
    ```mlir
    hip.group_query_attention(%ctx, %query, %key, %value,
        %past_key, %past_value, %seqlens_k, %total_seq_len,
        %output, %present_key, %present_value)
        {num_heads = 32, kv_num_heads = 8, scale = 0.0884,
         softcap = 0.0, do_rotary = 0, rotary_interleaved = 0}
        : (!hip.context, memref<1x1x4096xf16, 1>, ...)
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
  let results = (outs Variadic<AnyRankedTensor>:$result_tensors);

  let assemblyFormat = [{
    `(` $handle `,` $query `,` $key `,` $value `,`
        $past_key `,` $past_value `,` $seqlens_k `,` $total_seq_len `,`
        $output `,` $present_key `,` $present_value `)` attr-dict
    `:` `(` type($handle) `,` type($query) `,` type($key) `,` type($value) `,`
            type($past_key) `,` type($past_value) `,` type($seqlens_k) `,` type($total_seq_len) `,`
            type($output) `,` type($present_key) `,` type($present_value) `)`
    (`:` type($result_tensors)^)?
  }];
}

//===----------------------------------------------------------------------===//
// Elementwise Operations
//===----------------------------------------------------------------------===//

def Hip_MulOp : Hip_Op<"mul", [
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  DeclareOpInterfaceMethods<DestinationStyleOpInterface, ["getDpsInitsMutable"]>
]> {
  let summary = "Elementwise multiplication";
  let description = [{
    Performs elementwise multiplication: output = lhs * rhs
    Both inputs must have the same shape (broadcasting handled before lowering).

    Uses destination-passing style: output buffer is provided as argument.

    Example:
    ```mlir
    %output = hip.alloc(%handle) : memref<1x128x14336xf16, 1>
    hip.mul(%handle, %lhs, %rhs, %output)
      : (!hip.context, memref<1x128x14336xf16, 1>, memref<1x128x14336xf16, 1>,
         memref<1x128x14336xf16, 1>)
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$handle,
    Hip_TensorOrMemRef:$lhs,
    Hip_TensorOrMemRef:$rhs,
    Hip_TensorOrMemRef:$output
  );
  let results = (outs Variadic<AnyRankedTensor>:$result_tensors);

  let assemblyFormat = [{
    `(` $handle `,` $lhs `,` $rhs `,` $output `)` attr-dict
    `:` `(` type($handle) `,` type($lhs) `,` type($rhs) `,` type($output) `)`
    (`:` type($result_tensors)^)?
  }];
}

def Hip_SubOp : Hip_Op<"sub", [
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  DeclareOpInterfaceMethods<DestinationStyleOpInterface, ["getDpsInitsMutable"]>
]> {
  let summary = "Elementwise subtraction";
  let description = [{
    Performs elementwise subtraction: output = lhs - rhs
    Both inputs must have the same shape (broadcasting handled before lowering).

    Uses destination-passing style: output buffer is provided as argument.

    Example:
    ```mlir
    %output = hip.alloc(%handle) : memref<1x1xi64, 1>
    hip.sub(%handle, %lhs, %rhs, %output)
      : (!hip.context, memref<1x1xi64, 1>, memref<1x1xi64, 1>,
         memref<1x1xi64, 1>)
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$handle,
    Hip_TensorOrMemRef:$lhs,
    Hip_TensorOrMemRef:$rhs,
    Hip_TensorOrMemRef:$output
  );
  let results = (outs Variadic<AnyRankedTensor>:$result_tensors);

  let assemblyFormat = [{
    `(` $handle `,` $lhs `,` $rhs `,` $output `)` attr-dict
    `:` `(` type($handle) `,` type($lhs) `,` type($rhs) `,` type($output) `)`
    (`:` type($result_tensors)^)?
  }];
}

def Hip_SigmoidOp : Hip_Op<"sigmoid", [
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  DeclareOpInterfaceMethods<DestinationStyleOpInterface, ["getDpsInitsMutable"]>
]> {
  let summary = "Sigmoid activation";
  let description = [{
    Applies sigmoid activation: output = 1 / (1 + exp(-input))

    Uses destination-passing style: output buffer is provided as argument.

    Example:
    ```mlir
    %output = hip.alloc(%handle) : memref<1x128x14336xf16, 1>
    hip.sigmoid(%handle, %input, %output)
      : (!hip.context, memref<1x128x14336xf16, 1>, memref<1x128x14336xf16, 1>)
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$handle,
    Hip_TensorOrMemRef:$input,
    Hip_TensorOrMemRef:$output
  );
  let results = (outs Variadic<AnyRankedTensor>:$result_tensors);

  let assemblyFormat = [{
    `(` $handle `,` $input `,` $output `)` attr-dict
    `:` `(` type($handle) `,` type($input) `,` type($output) `)`
    (`:` type($result_tensors)^)?
  }];
}

//===----------------------------------------------------------------------===//
// Data Movement Operations
//===----------------------------------------------------------------------===//

def Hip_GatherOp : Hip_Op<"gather", [
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  DeclareOpInterfaceMethods<DestinationStyleOpInterface, ["getDpsInitsMutable"]>
]> {
  let summary = "Gather elements along an axis";
  let description = [{
    Gathers elements from data along the given axis using indices.
    Implements the ONNX Gather operator.

    Uses destination-passing style: output buffer is provided as argument.

    Example:
    ```mlir
    %output = hip.alloc(%handle) : memref<i64, 1>
    hip.gather(%handle, %data, %indices, %output) {axis = 0 : i64}
      : (!hip.context, memref<2xi64, 1>, memref<i64, 1>, memref<i64, 1>)
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$handle,
    Hip_TensorOrMemRef:$data,
    Hip_TensorOrMemRef:$indices,
    Hip_TensorOrMemRef:$output,
    I64Attr:$axis
  );
  let results = (outs Variadic<AnyRankedTensor>:$result_tensors);

  let assemblyFormat = [{
    `(` $handle `,` $data `,` $indices `,` $output `)` attr-dict
    `:` `(` type($handle) `,` type($data) `,` type($indices) `,` type($output) `)`
    (`:` type($result_tensors)^)?
  }];
}

//===----------------------------------------------------------------------===//
// Reduction Operations
//===----------------------------------------------------------------------===//

def Hip_ReduceSumOp : Hip_Op<"reduce_sum", [
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  DeclareOpInterfaceMethods<DestinationStyleOpInterface, ["getDpsInitsMutable"]>
]> {
  let summary = "Reduce tensor by summing along axes";
  let description = [{
    Computes the sum of input tensor elements along the specified axes.
    Implements the ONNX ReduceSum operator (opset 13+).

    Uses destination-passing style: output buffer is provided as argument.

    Example:
    ```mlir
    %output = hip.alloc(%handle) : memref<1x1xi64, 1>
    hip.reduce_sum(%handle, %data, %axes, %output) {keepdims = 1 : i64}
      : (!hip.context, memref<1x128xi64, 1>, memref<i64, 1>, memref<1x1xi64, 1>)
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$handle,
    Hip_TensorOrMemRef:$data,
    Hip_TensorOrMemRef:$axes,
    Hip_TensorOrMemRef:$output,
    I64Attr:$keepdims
  );
  let results = (outs Variadic<AnyRankedTensor>:$result_tensors);

  let assemblyFormat = [{
    `(` $handle `,` $data `,` $axes `,` $output `)` attr-dict
    `:` `(` type($handle) `,` type($data) `,` type($axes) `,` type($output) `)`
    (`:` type($result_tensors)^)?
  }];
}

def Hip_RotaryEmbeddingOp : Hip_Op<"rotary_embedding", [
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  DeclareOpInterfaceMethods<DestinationStyleOpInterface, ["getDpsInitsMutable"]>
]> {
  let summary = "Rotary position embedding (RoPE)";
  let description = [{
    Applies rotary position embedding to the input tensor.
    Implements the Microsoft RotaryEmbedding operator.

    Uses destination-passing style: output buffer is provided as argument.

    Example:
    ```mlir
    %output = hip.alloc(%handle) : memref<1x128x4096xf16, 1>
    hip.rotary_embedding(%handle, %input, %position_ids, %cos_cache, %sin_cache, %output)
        {interleaved = 0 : i64, num_heads = 0 : i64, rotary_embedding_dim = 0 : i64}
      : (!hip.context, memref<1x128x4096xf16, 1>, memref<1x128xi64, 1>,
         memref<131072x64xf16, 1>, memref<131072x64xf16, 1>, memref<1x128x4096xf16, 1>)
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
  let results = (outs Variadic<AnyRankedTensor>:$result_tensors);

  let assemblyFormat = [{
    `(` $handle `,` $input `,` $position_ids `,` $cos_cache `,` $sin_cache `,` $output `)` attr-dict
    `:` `(` type($handle) `,` type($input) `,` type($position_ids) `,`
         type($cos_cache) `,` type($sin_cache) `,` type($output) `)`
    (`:` type($result_tensors)^)?
  }];
}

def Hip_SimplifiedLayerNormOp : Hip_Op<"simplified_layer_norm", [
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  DeclareOpInterfaceMethods<DestinationStyleOpInterface, ["getDpsInitsMutable"]>
]> {
  let summary = "Simplified layer normalization";
  let description = [{
    Applies simplified (RMS) layer normalization.
    output = input / sqrt(mean(input^2) + epsilon) * scale

    Uses destination-passing style: output buffer is provided as argument.

    Example:
    ```mlir
    %output = hip.alloc(%handle) : memref<1x128x4096xf16, 1>
    hip.simplified_layer_norm(%handle, %input, %scale, %output)
        {axis = -1 : i64, epsilon = 9.99999974e-06 : f32, stash_type = 1 : i64}
      : (!hip.context, memref<1x128x4096xf16, 1>, memref<4096xf16, 1>,
         memref<1x128x4096xf16, 1>)
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
  let results = (outs Variadic<AnyRankedTensor>:$result_tensors);

  let assemblyFormat = [{
    `(` $handle `,` $input `,` $scale `,` $output `)` attr-dict
    `:` `(` type($handle) `,` type($input) `,` type($scale) `,` type($output) `)`
    (`:` type($result_tensors)^)?
  }];
}

def Hip_SkipSimplifiedLayerNormOp : Hip_Op<"skip_simplified_layer_norm", [
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  DeclareOpInterfaceMethods<DestinationStyleOpInterface, ["getDpsInitsMutable"]>
]> {
  let summary = "Skip connection + simplified layer normalization";
  let description = [{
    Adds a skip connection then applies simplified layer normalization.
    Implements the Microsoft SkipSimplifiedLayerNormalization operator.

    Has two outputs: the normalized result and the skip-sum for residual.

    Uses destination-passing style: output buffers are provided as arguments.

    Example:
    ```mlir
    %output = hip.alloc(%handle) : memref<1x128x4096xf16, 1>
    %skip_output = hip.alloc(%handle) : memref<1x128x4096xf16, 1>
    hip.skip_simplified_layer_norm(%handle, %input, %skip, %gamma, %output, %skip_output)
        {epsilon = 9.99999974e-06 : f32}
      : (!hip.context, memref<1x128x4096xf16, 1>, memref<1x128x4096xf16, 1>,
         memref<4096xf16, 1>, memref<1x128x4096xf16, 1>, memref<1x128x4096xf16, 1>)
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
  let results = (outs Variadic<AnyRankedTensor>:$result_tensors);

  let assemblyFormat = [{
    `(` $handle `,` $input `,` $skip `,` $gamma `,` $output `,` $skip_output `)` attr-dict
    `:` `(` type($handle) `,` type($input) `,` type($skip) `,`
         type($gamma) `,` type($output) `,` type($skip_output) `)`
    (`:` type($result_tensors)^)?
  }];
}

#endif // HIP_OPS
