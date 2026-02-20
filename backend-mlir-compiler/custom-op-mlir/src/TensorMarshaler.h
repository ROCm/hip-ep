/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef TENSOR_MARSHALER_H
#define TENSOR_MARSHALER_H

#include "custom_op_mlir.hpp"
#include <cstdint>
#include <vector>

// Forward declarations
struct OrtApi;
struct OrtKernelContext;

namespace mlir_compilation {

namespace customop {

class InputTensors {
public:
  // Marshal input tensors from ORT context
  static InputTensors marshal(OrtKernelContext *context);

  span_t *span();

private:
  InputTensors(std::vector<tensor_t> &&tensors,
               std::vector<std::vector<int64_t>> &&shapes, span_t &&span);

  std::vector<tensor_t> tensors_;
  std::vector<std::vector<int64_t>> shapes_; // Storage for shape arrays
  span_t span_;
};

class OutputTensors {
public:
  // Marshal output tensors from ORT context
  static OutputTensors
  marshal(OrtKernelContext *context,
          const std::vector<std::vector<int64_t>> &output_shapes);

  span_t *span();

private:
  OutputTensors(std::vector<tensor_t> &&tensors,
                std::vector<std::vector<int64_t>> &&shapes, span_t &&span);

  std::vector<tensor_t> tensors_;
  std::vector<std::vector<int64_t>> shapes_; // Storage for shape arrays
  span_t span_;
};

} // namespace customop
} // namespace mlir_compilation

#endif
