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

def Hip_HipblasltMatmulOp : Hip_DpsOp<"hipblaslt.matmul"> {
  let summary = "Matrix multiply C = A @ B via hipBLASLt";
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

// ===== MIOpen ops ============================================================

def Hip_MiopenRmsNormOp : Hip_DpsOp<"miopen.rms_norm"> {
  let summary = "RMS normalization via miopenT5LayerNormForward";
  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$input,
    Hip_TensorOrMemRef:$weight,
    Hip_TensorOrMemRef:$output
  );
  // result_tensors inherited from Hip_DpsOp
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def Hip_MiopenSkipRmsNormOp : Hip_DpsOp<"miopen.skip_rms_norm"> {
  let summary = "Fused Add + RMS normalization via miopenAddLayerNormForward (T5 mode)";
  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$x,
    Hip_TensorOrMemRef:$skip,
    Hip_TensorOrMemRef:$weight,
    Hip_TensorOrMemRef:$output,
    Hip_TensorOrMemRef:$residual
  );
  // result_tensors inherited from Hip_DpsOp
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def Hip_MiopenRopeOp : Hip_DpsOp<"miopen.rope"> {
  let summary = "Rotary positional embeddings (MIOpen experimental)";
  let arguments = (ins
    Hip_ContextType:$ctx,
    Index:$start_pos,
    Hip_TensorOrMemRef:$cos_cache,
    Hip_TensorOrMemRef:$sin_cache,
    Hip_TensorOrMemRef:$q,
    Hip_TensorOrMemRef:$k
  );
  // result_tensors inherited from Hip_DpsOp
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
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

def Hip_MiopenMulOp : Hip_DpsOp<"miopen.mul"> {
  let summary = "Element-wise multiply via miopenOpTensor(miopenTensorOpMul)";
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
  let summary = "Embedding table lookup";
  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$indices,
    Hip_TensorOrMemRef:$table,
    Hip_TensorOrMemRef:$output
  );
  // result_tensors inherited from Hip_DpsOp
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
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

def Hip_CastOp : Hip_DpsOp<"cast"> {
  let summary = "Type conversion operation";
  let description = [{
    Converts input tensor from one data type to another.

    Input and output may have different element types but the same shape.

    Uses destination-passing style: output buffer is provided as argument.

    Example:
    ```mlir
    hip.cast(%ctx) ins(%input : memref<128x256xf32, 1>)
                   outs(%output : memref<128x256xf16, 1>)
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

def Hip_ReduceSumOp : Hip_DpsOp<"reduce_sum"> {
  let summary = "Reduce sum operation";
  let description = [{
    Computes the sum of elements along specified axes.

    Supports multi-axis reduction with optional keepdims flag.

    Uses destination-passing style: output buffer is provided as argument.

    Example:
    ```mlir
    hip.reduce_sum(%ctx) ins(%input : memref<8x128x512xf32, 1>)
                         outs(%output : memref<8x128xf32, 1>)
                         {axes = [2], keepdims = false}
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$input,
    Hip_TensorOrMemRef:$output,
    I64ArrayAttr:$axes,
    BoolAttr:$keepdims
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $input `:` type($input) `)`
    `outs` `(` $output `:` type($output) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_GqaOp : Hip_DpsOp<"gqa"> {
  let summary = "Grouped query attention";
  let arguments = (ins
    Hip_ContextType:$ctx,
    Index:$layer,
    Index:$start_pos,
    Index:$seq_len,
    Hip_TensorOrMemRef:$q,
    Hip_TensorOrMemRef:$k,
    Hip_TensorOrMemRef:$v,
    Hip_TensorOrMemRef:$kv_cache,
    Hip_TensorOrMemRef:$output
  );
  // result_tensors inherited from Hip_DpsOp
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

#endif // HIP_OPS
