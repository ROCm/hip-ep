//===- hip_gemm_runtime_hipdnn.cpp - Runtime wrapper for hipDNN GEMM ------===//
//
// This file implements the C function hip_gemm_f32() that the MLIR-compiled
// code calls.  It uses the hipDNN C++ frontend graph API to perform C = A @ B
// on device.
//
// This is a drop-in replacement for hip_gemm_runtime.cpp (hipBLAS-LT).
// The MLIR lowering layer calls hip_gemm_f32(); swapping the runtime .obj
// at link time selects the backend without any MLIR changes.
//
//===----------------------------------------------------------------------===//

#include <hipdnn/frontend/hipdnn_frontend.hpp>
#include <hipdnn/backend/hipdnn_backend.h>
#include <hip/hip_runtime_api.h>

#include <cstdint>
#include <cstdio>
#include <memory>
#include <unordered_map>
#include <vector>

// Stub implementations for hip.create_handle / hip.destroy_handle.
// These are declared by the MLIR-generated code but are no-ops in this demo
// because hipDNN manages its own handle internally.
extern "C" void* hipCreateHandle() { return nullptr; }
extern "C" void  hipDestroyHandle(void*) {}

extern "C" void hip_gemm_f32(float* A, float* B, float* C,
                              int64_t M, int64_t K, int64_t N) {
  using namespace hipdnn_frontend;
  using namespace hipdnn_frontend::graph;

  // 1. Create hipDNN handle
  hipdnnHandle_t handle = nullptr;
  hipdnnStatus_t status = hipdnnCreate(&handle);
  if (status != HIPDNN_STATUS_SUCCESS) {
    fprintf(stderr, "hipdnnCreate failed (status=%d)\n", (int)status);
    return;
  }

  // 2. Build the matmul graph: C = A @ B
  //    A is [M, K], B is [K, N], C is [M, N]  (row-major)
  Graph graph;
  graph.graph_attributes
      .set_name("gemm_graph")
      .set_io_data_type(DataType::FLOAT)
      .set_intermediate_data_type(DataType::FLOAT)
      .set_compute_data_type(DataType::FLOAT);

  // Row-major strides: for a [rows, cols] matrix, stride = {cols, 1}
  auto a_tensor = std::make_shared<TensorAttributes>();
  a_tensor->set_name("A")
      .set_dim({M, K})
      .set_stride({K, 1})
      .set_uid(1)
      .set_data_type(DataType::FLOAT);

  auto b_tensor = std::make_shared<TensorAttributes>();
  b_tensor->set_name("B")
      .set_dim({K, N})
      .set_stride({N, 1})
      .set_uid(2)
      .set_data_type(DataType::FLOAT);

  MatmulAttributes matmul_attrs;
  matmul_attrs.set_name("matmul_node");

  auto c_tensor = graph.matmul(a_tensor, b_tensor, matmul_attrs);
  if (!c_tensor) {
    fprintf(stderr, "hip_gemm_f32: graph.matmul() returned null\n");
    hipdnnDestroy(handle);
    return;
  }

  // Mark C as a concrete (non-virtual) output tensor
  c_tensor->set_output(true);
  c_tensor->set_dim({M, N});
  c_tensor->set_stride({N, 1});
  c_tensor->set_uid(3);
  c_tensor->set_data_type(DataType::FLOAT);

  // 3. Build the execution plan
  Error err = graph.build(handle, {HeuristicMode::FALLBACK});
  if (!err.is_good()) {
    fprintf(stderr, "hip_gemm_f32: graph.build() failed: %s\n",
            err.get_message().c_str());
    hipdnnDestroy(handle);
    return;
  }

  // 4. Query and allocate workspace
  int64_t workspace_size = 0;
  err = graph.get_workspace_size(workspace_size);
  if (!err.is_good()) {
    fprintf(stderr, "hip_gemm_f32: get_workspace_size() failed: %s\n",
            err.get_message().c_str());
    hipdnnDestroy(handle);
    return;
  }

  void* workspace_ptr = nullptr;
  if (workspace_size > 0) {
    hipMalloc(&workspace_ptr, workspace_size);
  }

  // 5. Execute: map tensor UIDs to device pointers
  std::unordered_map<int64_t, void*> variant_pack;
  variant_pack[1] = A;  // UID 1 -> A
  variant_pack[2] = B;  // UID 2 -> B
  variant_pack[3] = C;  // UID 3 -> C

  err = graph.execute(handle, variant_pack, workspace_ptr);
  if (!err.is_good()) {
    fprintf(stderr, "hip_gemm_f32: graph.execute() failed: %s\n",
            err.get_message().c_str());
  }

  hipDeviceSynchronize();

  // 6. Cleanup
  if (workspace_ptr) hipFree(workspace_ptr);
  hipdnnDestroy(handle);
}
