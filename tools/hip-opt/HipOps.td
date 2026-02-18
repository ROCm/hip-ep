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

def Hip_GemmOp : Hip_Op<"gemm", []> {
  let summary = "Matrix multiply C = A @ B using hipDNN";
  let description = [{
    Performs matrix multiplication C = A @ B using hipDNN graph API.
    A is MxK, B is KxN, C is MxN. All pointers must be device memory.

    Example:
    ```mlir
    hip.gemm(%handle, %A, %B, %C, %M, %K, %N)
        : (!hip.handle, !llvm.ptr, !llvm.ptr, !llvm.ptr, index, index, index)
    ```
  }];

  let arguments = (ins
    Hip_HandleType:$handle,
    AnyType:$A,
    AnyType:$B,
    AnyType:$C,
    Index:$M, Index:$K, Index:$N
  );

  let assemblyFormat = [{
    `(` $handle `,` $A `,` $B `,` $C `,` $M `,` $K `,` $N `)` attr-dict `:` `(` type($handle) `,` type($A) `,` type($B) `,` type($C) `,` type($M) `,` type($K) `,` type($N) `)`
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

def Hip_HipblasltMatmulOp : Hip_Op<"hipblaslt.matmul", []> {
  let summary = "Matrix multiply C = A @ B via hipBLASLt";
  let arguments = (ins
    Hip_HandleType:$handle,
    AnyMemRef:$A,
    AnyMemRef:$B,
    AnyMemRef:$C
  );
  let assemblyFormat = [{
    `(` $handle `,` $A `,` $B `,` $C `)`
    attr-dict `:` `(` type($A) `,` type($B) `,` type($C) `)` `->` `(` `)`
  }];
}

// ===== MIOpen ops ============================================================

def Hip_MiopenRmsNormOp : Hip_Op<"miopen.rms_norm", []> {
  let summary = "RMS normalization via miopenT5LayerNormForward";
  let arguments = (ins
    Hip_HandleType:$handle,
    AnyMemRef:$input,
    AnyMemRef:$weight,
    AnyMemRef:$output
  );
  let assemblyFormat = [{
    `(` $handle `,` $input `,` $weight `,` $output `)`
    attr-dict `:` `(` type($input) `,` type($weight) `,` type($output) `)` `->` `(` `)`
  }];
}

def Hip_MiopenSkipRmsNormOp : Hip_Op<"miopen.skip_rms_norm", []> {
  let summary = "Fused Add + RMS normalization via miopenAddLayerNormForward (T5 mode)";
  let arguments = (ins
    Hip_HandleType:$handle,
    AnyMemRef:$x,
    AnyMemRef:$skip,
    AnyMemRef:$weight,
    AnyMemRef:$output,
    AnyMemRef:$residual
  );
  let assemblyFormat = [{
    `(` $handle `,` $x `,` $skip `,` $weight `,` $output `,` $residual `)`
    attr-dict `:` `(` type($x) `,` type($skip) `,` type($weight) `,`
                      type($output) `,` type($residual) `)` `->` `(` `)`
  }];
}

def Hip_MiopenRopeOp : Hip_Op<"miopen.rope", []> {
  let summary = "Rotary positional embeddings (MIOpen experimental)";
  let arguments = (ins
    Hip_HandleType:$handle,
    AnyMemRef:$q,
    AnyMemRef:$k,
    AnyMemRef:$cos_cache,
    AnyMemRef:$sin_cache,
    Index:$start_pos
  );
  let assemblyFormat = [{
    `(` $handle `,` $q `,` $k `,` $cos_cache `,` $sin_cache `,` $start_pos `)`
    attr-dict `:` `(` type($q) `,` type($k) `,`
                      type($cos_cache) `,` type($sin_cache) `,` type($start_pos) `)` `->` `(` `)`
  }];
}

def Hip_MiopenAddOp : Hip_Op<"miopen.add", []> {
  let summary = "Element-wise add via miopenOpTensor(miopenTensorOpAdd)";
  let arguments = (ins
    Hip_HandleType:$handle,
    AnyMemRef:$A,
    AnyMemRef:$B,
    AnyMemRef:$C
  );
  let assemblyFormat = [{
    `(` $handle `,` $A `,` $B `,` $C `)`
    attr-dict `:` `(` type($A) `,` type($B) `,` type($C) `)` `->` `(` `)`
  }];
}

def Hip_MiopenMulOp : Hip_Op<"miopen.mul", []> {
  let summary = "Element-wise multiply via miopenOpTensor(miopenTensorOpMul)";
  let arguments = (ins
    Hip_HandleType:$handle,
    AnyMemRef:$A,
    AnyMemRef:$B,
    AnyMemRef:$C
  );
  let assemblyFormat = [{
    `(` $handle `,` $A `,` $B `,` $C `)`
    attr-dict `:` `(` type($A) `,` type($B) `,` type($C) `)` `->` `(` `)`
  }];
}

// ===== Custom HIP kernel ops (no library equivalent) =========================

def Hip_GatherOp : Hip_Op<"gather", []> {
  let summary = "Embedding table lookup";
  let arguments = (ins
    Hip_HandleType:$handle,
    AnyMemRef:$indices,
    AnyMemRef:$table,
    AnyMemRef:$output
  );
  let assemblyFormat = [{
    `(` $handle `,` $indices `,` $table `,` $output `)`
    attr-dict `:` `(` type($indices) `,` type($table) `,` type($output) `)` `->` `(` `)`
  }];
}

def Hip_SiluOp : Hip_Op<"silu", []> {
  let summary = "SiLU activation: output = input * sigmoid(input)";
  let arguments = (ins
    Hip_HandleType:$handle,
    AnyMemRef:$input,
    AnyMemRef:$output
  );
  let assemblyFormat = [{
    `(` $handle `,` $input `,` $output `)`
    attr-dict `:` `(` type($input) `,` type($output) `)` `->` `(` `)`
  }];
}

def Hip_GqaOp : Hip_Op<"gqa", []> {
  let summary = "Grouped query attention";
  let arguments = (ins
    Hip_HandleType:$handle,
    AnyMemRef:$q,
    AnyMemRef:$k,
    AnyMemRef:$v,
    AnyMemRef:$kv_cache,
    AnyMemRef:$output,
    Index:$layer,
    Index:$start_pos,
    Index:$seq_len
  );
  let assemblyFormat = [{
    `(` $handle `,` $q `,` $k `,` $v `,` $kv_cache `,` $output `,`
        $layer `,` $start_pos `,` $seq_len `)`
    attr-dict `:` `(` type($q) `,` type($k) `,` type($v) `,`
                      type($kv_cache) `,` type($output) `,`
                      type($layer) `,` type($start_pos) `,` type($seq_len) `)` `->` `(` `)`
  }];
}

#endif // HIP_OPS
