#ifndef HIP_OPS
#define HIP_OPS

include "HipDialect.td"
include "HipTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Dialect/Bufferization/IR/AllocationOpInterface.td"
include "mlir/IR/OpBase.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"

class Hip_Op<string mnemonic, list<Trait> traits = []> :
    Op<Hip_Dialect, mnemonic, traits>;

def Hip_AllocOp : Hip_Op<"alloc", [
  DeclareOpInterfaceMethods<AllocationOpInterface, ["buildDealloc", "buildClone"]>,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Allocate device memory";
  let description = [{
    Allocates device memory using HIP and returns a memref.

    Example:
    ```mlir
    %x = hip.alloc(%handle, %N) : memref<?x128xf32, 1>
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$handle,
    Variadic<Index>:$dynamicSizes
  );
  let results = (outs AnyMemRef:$memref);

  let assemblyFormat = [{
    `(` $handle (`,` $dynamicSizes^ )? `)` attr-dict `:` type($memref)
  }];
}

def Hip_FreeOp : Hip_Op<"free", [
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Free device memory";
  let description = [{
    Frees device memory allocated with hip.alloc.

    Example:
    ```mlir
    hip.free(%handle, %x) : memref<?x128xf32, 1>
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$handle,
    AnyMemRef:$memref
  );

  let assemblyFormat = [{
    `(` $handle `,` $memref `)` attr-dict `:` type($memref)
  }];
}

//===----------------------------------------------------------------------===//
// MIOpen DNN Operations
//===----------------------------------------------------------------------===//

def Hip_ConvOp : Hip_Op<"conv", [
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  DeclareOpInterfaceMethods<DestinationStyleOpInterface>
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
    AnyMemRef:$input,
    AnyMemRef:$weights,
    Optional<AnyMemRef>:$bias,
    AnyMemRef:$output,
    I64ArrayAttr:$kernel_shape,
    I64ArrayAttr:$strides,
    I64ArrayAttr:$pads,
    I64ArrayAttr:$dilations,
    I64Attr:$group
  );
  let results = (outs);

  let assemblyFormat = [{
    `(` $handle `,` $input `,` $weights (`,` $bias^)? `,` $output `)` attr-dict
    `:` `(` type($handle) `,` type($input) `,` type($weights) (`,` type($bias)^)? `,` type($output) `)`
  }];
}

def Hip_GemmOp : Hip_Op<"gemm", [
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  DeclareOpInterfaceMethods<DestinationStyleOpInterface>
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
    AnyMemRef:$A,
    AnyMemRef:$B,
    AnyMemRef:$C,
    AnyMemRef:$result,
    I64Attr:$transA,
    I64Attr:$transB,
    F32Attr:$alpha,
    F32Attr:$beta
  );
  let results = (outs);

  let assemblyFormat = [{
    `(` $handle `,` $A `,` $B `,` $C `,` $result `)` attr-dict
    `:` `(` type($handle) `,` type($A) `,` type($B) `,` type($C) `,` type($result) `)`
  }];
}

def Hip_MaxPoolOp : Hip_Op<"maxpool", [
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  DeclareOpInterfaceMethods<DestinationStyleOpInterface>
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
    AnyMemRef:$input,
    AnyMemRef:$output,
    I64ArrayAttr:$kernel_shape,
    I64ArrayAttr:$strides,
    I64ArrayAttr:$pads
  );
  let results = (outs);

  let assemblyFormat = [{
    `(` $handle `,` $input `,` $output `)` attr-dict
    `:` `(` type($handle) `,` type($input) `,` type($output) `)`
  }];
}

def Hip_AvgPoolOp : Hip_Op<"avgpool", [
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  DeclareOpInterfaceMethods<DestinationStyleOpInterface>
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
    AnyMemRef:$input,
    AnyMemRef:$output,
    I64ArrayAttr:$kernel_shape,
    I64ArrayAttr:$strides,
    I64ArrayAttr:$pads
  );
  let results = (outs);

  let assemblyFormat = [{
    `(` $handle `,` $input `,` $output `)` attr-dict
    `:` `(` type($handle) `,` type($input) `,` type($output) `)`
  }];
}

//===----------------------------------------------------------------------===//
// Activation Operations
//===----------------------------------------------------------------------===//

def Hip_ReluOp : Hip_Op<"relu", [
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  DeclareOpInterfaceMethods<DestinationStyleOpInterface>
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
    AnyMemRef:$input,
    AnyMemRef:$output
  );
  let results = (outs);

  let assemblyFormat = [{
    `(` $handle `,` $input `,` $output `)` attr-dict
    `:` `(` type($handle) `,` type($input) `,` type($output) `)`
  }];
}

//===----------------------------------------------------------------------===//
// Memory Operations
//===----------------------------------------------------------------------===//

def Hip_CopyOp : Hip_Op<"copy", [
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "GPU-to-GPU memory copy";
  let description = [{
    Copies data between GPU buffers using hipMemcpyAsync.
    Both source and destination must be in GPU address space (memref with address space 1).

    Uses asynchronous copy on GPU stream from context.

    Example:
    ```mlir
    hip.copy(%ctx, %src, %dst)
      : (!hip.context, memref<1x64x224x224xf32, 1>, memref<1x64x224x224xf32, 1>)
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    AnyMemRef:$source,
    AnyMemRef:$destination
  );
  let results = (outs);

  let assemblyFormat = [{
    `(` $ctx `,` $source `,` $destination `)` attr-dict
    `:` `(` type($ctx) `,` type($source) `,` type($destination) `)`
  }];

  let hasCanonicalizer = 1;
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

#endif // HIP_OPS
