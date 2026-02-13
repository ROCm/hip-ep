//===- hip_gemm_runtime.cpp - Runtime wrapper for hipDNN GEMM -------------===//
//
// This file implements the C function hip_gemm_f32() that the MLIR-compiled
// code calls. It uses the hipDNN graph API to perform C = A @ B on device.
//
// Compile with:
//   cl.exe /c /EHsc /I%THEROCK_DIST%/include hip_gemm_runtime.cpp
//
//===----------------------------------------------------------------------===//

#include <hipdnn_backend.h>
#include <hipdnn_frontend.hpp>
#include <hip/hip_runtime_api.h>

#include <cstdint>
#include <cstdio>
#include <memory>
#include <unordered_map>
#include <vector>

// Compute row-major strides for a 2D tensor: shape [rows, cols] -> strides [cols, 1]
static std::vector<int64_t> ComputeStrides(const std::vector<int64_t>& shape) {
  std::vector<int64_t> strides(shape.size());
  int64_t stride = 1;
  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
    strides[i] = stride;
    stride *= shape[i];
  }
  return strides;
}

extern "C" void hip_gemm_f32(float* A, float* B, float* C,
                              int64_t M, int64_t K, int64_t N) {
  using namespace hipdnn_frontend;
  using namespace hipdnn_frontend::graph;

  // 1. Create hipDNN handle
  hipdnnHandle_t handle = nullptr;
  hipdnnStatus_t status = hipdnnCreate(&handle);
  if (status != HIPDNN_STATUS_SUCCESS) {
    fprintf(stderr, "hip_gemm_f32: hipdnnCreate failed (status=%d)\n", status);
    return;
  }

  // 2. Build the graph
  auto graph = std::make_unique<Graph>();

  // TensorAttributes for A (M x K)
  auto a_attr = std::make_shared<TensorAttributes>();
  a_attr->set_uid(1)
      .set_name("A")
      .set_data_type(DataType::FLOAT)
      .set_dim({M, K})
      .set_stride(ComputeStrides({M, K}))
      .set_is_virtual(false);

  // TensorAttributes for B (K x N)
  auto b_attr = std::make_shared<TensorAttributes>();
  b_attr->set_uid(2)
      .set_name("B")
      .set_data_type(DataType::FLOAT)
      .set_dim({K, N})
      .set_stride(ComputeStrides({K, N}))
      .set_is_virtual(false);

  // MatMul: result = A @ B
  MatmulAttributes matmul_attrs;
  matmul_attrs.set_compute_data_type(DataType::FLOAT);
  auto c_attr = graph->matmul(a_attr, b_attr, matmul_attrs);

  // Mark output as non-virtual with known shape
  c_attr->set_uid(3)
      .set_name("C")
      .set_data_type(DataType::FLOAT)
      .set_dim({M, N})
      .set_stride(ComputeStrides({M, N}))
      .set_is_virtual(false);

  // 3. Compile the graph
  auto error = graph->validate();
  if (error.is_bad()) {
    fprintf(stderr, "hip_gemm_f32: validate failed: %s\n",
            error.get_message().c_str());
    hipdnnDestroy(handle);
    return;
  }

  error = graph->build_operation_graph(handle);
  if (error.is_bad()) {
    fprintf(stderr, "hip_gemm_f32: build_operation_graph failed: %s\n",
            error.get_message().c_str());
    hipdnnDestroy(handle);
    return;
  }

  error = graph->create_execution_plans({HeuristicMode::FALLBACK});
  if (error.is_bad()) {
    fprintf(stderr, "hip_gemm_f32: create_execution_plans failed: %s\n",
            error.get_message().c_str());
    hipdnnDestroy(handle);
    return;
  }

  error = graph->check_support();
  if (error.is_bad()) {
    fprintf(stderr, "hip_gemm_f32: check_support failed: %s\n",
            error.get_message().c_str());
    hipdnnDestroy(handle);
    return;
  }

  error = graph->build_plans();
  if (error.is_bad()) {
    fprintf(stderr, "hip_gemm_f32: build_plans failed: %s\n",
            error.get_message().c_str());
    hipdnnDestroy(handle);
    return;
  }

  // Allocate workspace on device if needed
  int64_t workspace_size = 0;
  error = graph->get_workspace_size(workspace_size);
  if (error.is_bad()) {
    fprintf(stderr, "hip_gemm_f32: get_workspace_size failed: %s\n",
            error.get_message().c_str());
    hipdnnDestroy(handle);
    return;
  }

  void* workspace_ptr = nullptr;
  if (workspace_size > 0) {
    hipError_t hip_err = hipMalloc(&workspace_ptr, workspace_size);
    if (hip_err != hipSuccess) {
      fprintf(stderr, "hip_gemm_f32: hipMalloc workspace failed: %s\n",
              hipGetErrorString(hip_err));
      hipdnnDestroy(handle);
      return;
    }
  }

  // 4. Execute
  std::unordered_map<int64_t, void*> variant_pack;
  variant_pack[1] = static_cast<void*>(A);
  variant_pack[2] = static_cast<void*>(B);
  variant_pack[3] = static_cast<void*>(C);

  error = graph->execute(handle, variant_pack, workspace_ptr);
  if (error.is_bad()) {
    fprintf(stderr, "hip_gemm_f32: execute failed: %s\n",
            error.get_message().c_str());
  }

  // Synchronize to make sure GEMM is done before returning
  hipDeviceSynchronize();

  // 5. Cleanup
  if (workspace_ptr) {
    hipFree(workspace_ptr);
  }
  hipdnnDestroy(handle);
}
