#ifndef HIP_OPS
#define HIP_OPS

include "HipDialect.td"
include "HipTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

class Hip_Op<string mnemonic, list<Trait> traits = []> :
    Op<Hip_Dialect, mnemonic, traits>;

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

#endif // HIP_OPS
