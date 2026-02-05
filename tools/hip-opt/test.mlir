// RUN: hip-opt %s --convert-hip-to-llvm | FileCheck %s

module {
  func.func @test_hip_ops(%N: index) {
    // Create a HIP handle
    // CHECK: %[[HANDLE:.*]] = llvm.call @hipCreateHandle() : () -> !llvm.ptr
    %handle = hip.create_handle() : !hip.handle
    
    // Allocate device memory for a dynamic ?x128 f32 tensor
    // CHECK: %[[MALLOC_RES:.*]] = llvm.call @hipMalloc(%{{.*}}) : (i64) -> !llvm.ptr
    // CHECK: %[[TYPED_PTR:.*]] = llvm.bitcast %[[MALLOC_RES]] : !llvm.ptr to !llvm.ptr<f32>
    %x = hip.alloc(%handle, %N) : memref<?x128xf32, 1>
    
    // Free the device memory
    // CHECK: %[[VOID_PTR:.*]] = llvm.bitcast %[[TYPED_PTR]] : !llvm.ptr<f32> to !llvm.ptr
    // CHECK: llvm.call @hipFree(%[[VOID_PTR]]) : (!llvm.ptr) -> ()
    hip.free(%handle, %x) : memref<?x128xf32, 1>
    
    // Destroy the HIP handle
    // CHECK: llvm.call @hipDestroyHandle(%[[HANDLE]]) : (!llvm.ptr) -> ()
    hip.destroy_handle(%handle) : !hip.handle
    
    return
  }
}
