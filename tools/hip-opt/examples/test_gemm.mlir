// Test: hip.gemm lowering
//
// This function is callable from C as:
//   void run_gemm(float* A, float* B, float* C, int64_t M, int64_t K, int64_t N)
//
// All pointer arguments are device memory (already allocated by the caller).
// The caller is responsible for hipMalloc, hipMemcpy, and hipFree.
//
// Compile pipeline:
//   hip-opt test_gemm.mlir --convert-hip-to-llvm --convert-func-to-llvm \
//       --reconcile-unrealized-casts | mlir-translate --mlir-to-llvmir -o gemm.ll
//   llc gemm.ll -filetype=obj -o gemm.obj

module {
  func.func @run_gemm(%A: !llvm.ptr, %B: !llvm.ptr, %C: !llvm.ptr,
                      %M: index, %K: index, %N: index) {
    %handle = hip.create_handle() : !hip.handle
    hip.gemm(%handle, %A, %B, %C, %M, %K, %N)
        : (!hip.handle, !llvm.ptr, !llvm.ptr, !llvm.ptr, index, index, index)
    hip.destroy_handle(%handle) : !hip.handle
    return
  }
}
