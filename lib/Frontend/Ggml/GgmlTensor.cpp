/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- GgmlTensor.cpp - A ggml tensor, as a split describes it ------------===//

#include "hip/Frontend/Ggml/GgmlTensor.h"

#include "hip/Dialect/Ggml/IR/GgmlBlockLayout.h"

#include <cstring>
#include <iterator>
#include <stdexcept>

namespace hip::ggml {

std::optional<std::string> typeName(int32_t type) {
  // enum ggml_type: GGUF files store these numbers. A gap is a type ggml
  // removed.
  static const char *const names[] = {
      "f32",   "f16",     "q4_0",    "q4_1",   nullptr,   nullptr, "q5_0",
      "q5_1",  "q8_0",    "q8_1",    "q2_K",   "q3_K",    "q4_K",  "q5_K",
      "q6_K",  "q8_K",    "iq2_xxs", "iq2_xs", "iq3_xxs", "iq1_s", "iq4_nl",
      "iq3_s", "iq2_s",   "iq4_xs",  "i8",     "i16",     "i32",   "i64",
      "f64",   "iq1_m",   "bf16",    nullptr,  nullptr,   nullptr, "tq1_0",
      "tq2_0", nullptr,   nullptr,   nullptr,  "mxfp4",   "nvfp4", "q1_0",
      "q2_0",  "f8_e4m3",
  };
  if (type < 0 || static_cast<size_t>(type) >= std::size(names) ||
      names[type] == nullptr)
    return std::nullopt;
  return std::string(names[type]);
}

std::optional<TypeLayout> typeLayout(int32_t type) {
  std::optional<std::string> name = typeName(type);
  if (!name)
    return std::nullopt;
  std::optional<mlir::ggml::BlockLayout> layout =
      mlir::ggml::getBlockLayout(*name);
  if (!layout)
    return std::nullopt;
  return TypeLayout{layout->typeSize, layout->blockElements};
}

int32_t opParamI32(const Tensor &tensor, int index) {
  return tensor.opParams[index];
}

float opParamF32(const Tensor &tensor, int index) {
  float value;
  std::memcpy(&value, &tensor.opParams[index], sizeof(value));
  return value;
}

bool isView(const Tensor &tensor) {
  return tensor.op == "VIEW" || tensor.op == "RESHAPE" ||
         tensor.op == "PERMUTE" || tensor.op == "TRANSPOSE";
}

int64_t nelements(const Tensor &tensor) {
  return tensor.ne[0] * tensor.ne[1] * tensor.ne[2] * tensor.ne[3];
}

static TypeLayout layoutOf(const Tensor &tensor) {
  std::optional<TypeLayout> layout = typeLayout(tensor.type);
  if (!layout)
    throw std::runtime_error("tensor '" + tensor.name + "' is of type " +
                             std::to_string(tensor.type) +
                             ", which ggml has none of");
  return *layout;
}

size_t nbytes(const Tensor &tensor) {
  for (int64_t e : tensor.ne)
    if (e <= 0)
      return 0;
  const TypeLayout layout = layoutOf(tensor);
  const size_t block = static_cast<size_t>(layout.blockElements);
  size_t bytes;
  if (block == 1) {
    bytes = static_cast<size_t>(layout.typeSize);
    for (int i = 0; i < kMaxDims; ++i)
      bytes += static_cast<size_t>(tensor.ne[i] - 1) * tensor.nb[i];
  } else {
    bytes = static_cast<size_t>(tensor.ne[0]) * tensor.nb[0] / block;
    for (int i = 1; i < kMaxDims; ++i)
      bytes += static_cast<size_t>(tensor.ne[i] - 1) * tensor.nb[i];
  }
  return bytes;
}

bool isContiguous(const Tensor &tensor) {
  const TypeLayout layout = layoutOf(tensor);
  size_t next = static_cast<size_t>(layout.typeSize);
  if (tensor.ne[0] != layout.blockElements && tensor.nb[0] != next)
    return false;
  next *= static_cast<size_t>(tensor.ne[0] / layout.blockElements);
  for (int i = 1; i < kMaxDims; ++i) {
    if (tensor.ne[i] != 1 && tensor.nb[i] != next)
      return false;
    next *= static_cast<size_t>(tensor.ne[i]);
  }
  return true;
}

} // namespace hip::ggml
