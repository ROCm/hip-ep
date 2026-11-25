/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- GgmlTensor.h - A ggml tensor, as a split describes it --------------===//
//
// What the frontend knows of a ggml tensor: the metadata a backend describes
// (hipep_backend.h), with its sources and the tensor it views as pointers.
// No ggml is linked: the few facts of ggml's the frontend needs - the type
// numbers, which GGUF files store and so never change, the bytes a tensor of
// a type spans, and where an operation keeps its configuration (GgmlOps.td) -
// are written down here and in the dialect.
//
//===----------------------------------------------------------------------===//

#ifndef HIP_FRONTEND_GGML_GGMLTENSOR_H
#define HIP_FRONTEND_GGML_GGMLTENSOR_H

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

namespace hip::ggml {

inline constexpr int kMaxDims = 4;
inline constexpr int kMaxSources = 10;
inline constexpr int kOpParams = 16;

/// ggml's tensor flags, as a description carries them.
inline constexpr int32_t kFlagInput = 1;
inline constexpr int32_t kFlagOutput = 2;

/// A tensor of a split: its metadata, never its data.
struct Tensor {
  std::string name;
  /// ggml's operation name ("MUL_MAT"); "NONE" for a leaf.
  std::string op = "NONE";
  /// ggml's type number (enum ggml_type).
  int32_t type = 0;
  int64_t ne[kMaxDims] = {1, 1, 1, 1};
  size_t nb[kMaxDims] = {0, 0, 0, 0};
  int32_t opParams[kOpParams] = {};
  int32_t flags = 0;
  const Tensor *src[kMaxSources] = {};
  const Tensor *viewSrc = nullptr;
  size_t viewOffs = 0;
};

/// ggml's name of type number `type` ("f32", "q4_K"), as ggml_type_name
/// gives it, or nullopt for a number ggml has no type of.
std::optional<std::string> typeName(int32_t type);

/// Bytes a block of `type` takes and the elements it holds (ggml_type_size,
/// ggml_blck_size); nullopt for an unknown type.
struct TypeLayout {
  int64_t typeSize = 0;
  int64_t blockElements = 0;
};
std::optional<TypeLayout> typeLayout(int32_t type);

/// The int32 and float32 at `index` of the tensor's op_params.
int32_t opParamI32(const Tensor &tensor, int index);
float opParamF32(const Tensor &tensor, int index);

/// A view, reshape, permute or transpose: another look at its source's bytes.
bool isView(const Tensor &tensor);

int64_t nelements(const Tensor &tensor);
/// The bytes the tensor spans (ggml_nbytes); throws for an unknown type.
size_t nbytes(const Tensor &tensor);
/// Rows of its type packed one after another (ggml_is_contiguous).
bool isContiguous(const Tensor &tensor);

} // namespace hip::ggml

#endif // HIP_FRONTEND_GGML_GGMLTENSOR_H
