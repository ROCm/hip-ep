#ifndef HIP_OPS
#define HIP_OPS

include "HipDialect.td"
include "HipTypes.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

class Hip_Op<string mnemonic, list<Trait> traits = []> :
    Op<Hip_Dialect, mnemonic, traits>;

// Accepts both ranked tensor (pre-bufferization) and memref (post-bufferization).
def Hip_TensorOrMemRef : AnyTypeOf<[AnyRankedTensor, AnyMemRef],
                                    "ranked tensor or memref">;

def Hip_CreateHandleOp : Hip_Op<"create_handle", []> {
  let summary = "Create a HIP runtime handle";
  let description = [{
    Creates and returns a new HIP runtime handle.

    Example:
    ```mlir
    %handle = hip.create_handle() : !hip.handle
    ```
  }];

  let results = (outs Hip_HandleType:$handle);
  let assemblyFormat = "`(` `)` attr-dict `:` type($handle)";
}

def Hip_DestroyHandleOp : Hip_Op<"destroy_handle", []> {
  let summary = "Destroy a HIP runtime handle";
  let description = [{
    Destroys a HIP runtime handle.

    Example:
    ```mlir
    hip.destroy_handle(%handle) : !hip.handle
    ```
  }];

  let arguments = (ins Hip_HandleType:$handle);
  let assemblyFormat = "`(` $handle `)` attr-dict `:` type($handle)";
}

def Hip_AllocOp : Hip_Op<"alloc", []> {
  let summary = "Allocate device memory";
  let description = [{
    Allocates device memory using HIP and returns a memref.

    Example:
    ```mlir
    %x = hip.alloc(%handle, %N) : memref<?x128xf32, 1>
    ```
  }];

  let arguments = (ins
    Hip_HandleType:$handle,
    Variadic<Index>:$dynamicSizes
  );
  let results = (outs AnyMemRef:$memref);

  let assemblyFormat = [{
    `(` $handle (`,` $dynamicSizes^ )? `)` attr-dict `:` type($memref)
  }];
}

def Hip_FreeOp : Hip_Op<"free", []> {
  let summary = "Free device memory";
  let description = [{
    Frees device memory allocated with hip.alloc.

    Example:
    ```mlir
    hip.free(%handle, %x) : memref<?x128xf32, 1>
    ```
  }];

  let arguments = (ins
    Hip_HandleType:$handle,
    AnyMemRef:$memref
  );

  let assemblyFormat = [{
    `(` $handle `,` $memref `)` attr-dict `:` type($memref)
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

// ===== hipBLASLt ops =========================================================

def Hip_HipblasltMatmulOp : Hip_Op<"hipblaslt.matmul",
    [DeclareOpInterfaceMethods<DestinationStyleOpInterface>,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Matrix multiply C = A @ B via hipBLASLt";
  let arguments = (ins
    Hip_HandleType:$handle,
    Hip_TensorOrMemRef:$A,
    Hip_TensorOrMemRef:$B,
    Hip_TensorOrMemRef:$C
  );
  let results = (outs Optional<AnyRankedTensor>:$result);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

// ===== MIOpen ops ============================================================

def Hip_MiopenAddOp : Hip_Op<"miopen.add",
    [DeclareOpInterfaceMethods<DestinationStyleOpInterface>,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Element-wise add via miopenOpTensor(miopenTensorOpAdd)";
  let arguments = (ins
    Hip_HandleType:$handle,
    Hip_TensorOrMemRef:$A,
    Hip_TensorOrMemRef:$B,
    Hip_TensorOrMemRef:$C
  );
  let results = (outs Optional<AnyRankedTensor>:$result);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def Hip_MiopenMulOp : Hip_Op<"miopen.mul",
    [DeclareOpInterfaceMethods<DestinationStyleOpInterface>,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Element-wise multiply via miopenOpTensor(miopenTensorOpMul)";
  let arguments = (ins
    Hip_HandleType:$handle,
    Hip_TensorOrMemRef:$A,
    Hip_TensorOrMemRef:$B,
    Hip_TensorOrMemRef:$C
  );
  let results = (outs Optional<AnyRankedTensor>:$result);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def Hip_MiopenSoftmaxOp : Hip_Op<"miopen.softmax",
    [DeclareOpInterfaceMethods<DestinationStyleOpInterface>,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Softmax via miopenSoftmaxForward_V2 (row-wise, last dim)";
  let arguments = (ins
    Hip_HandleType:$handle,
    Hip_TensorOrMemRef:$input,
    Hip_TensorOrMemRef:$output
  );
  let results = (outs Optional<AnyRankedTensor>:$result);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

// ===== Custom HIP kernel ops (no library equivalent) =========================

def Hip_TransposeOp : Hip_Op<"transpose",
    [DeclareOpInterfaceMethods<DestinationStyleOpInterface>,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "N-D transpose swapping two specified dimensions";
  let arguments = (ins
    Hip_HandleType:$handle,
    Index:$dim0,
    Index:$dim1,
    Hip_TensorOrMemRef:$input,
    Hip_TensorOrMemRef:$output
  );
  let results = (outs Optional<AnyRankedTensor>:$result);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

#endif // HIP_OPS
