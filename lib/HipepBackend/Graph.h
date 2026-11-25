/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Graph - a split as the backend described it (hipep_backend.h), rebuilt as
// ggml tensors of the library's own: metadata only, no memory. The frontend
// reads ggml tensors (SplitGraph.h, SplitCompiler.h); which ggml the backend
// was built with does not matter to it but for the operations' parameters
// (the description's ggml_version).

#ifndef HIPEP_BACKEND_GRAPH_H
#define HIPEP_BACKEND_GRAPH_H

#include "hipep_backend.h"

#include "hip/Frontend/Ggml/SplitGraph.h"

#include <string>
#include <vector>

struct ggml_context;

namespace hipep_backend {

namespace capture = hip::ggml::capture;

/// The description's tensor `i`, as long as the caller's tensor is.
const hipep_backend_tensor_desc &tensorAt(const hipep_backend_split_desc &desc,
                                          size_t i);

/// Why the library does not read the description ("" when it does): another
/// ABI, or another ggml.
std::string compatibilityError(const hipep_backend_split_desc &desc);

class Graph {
public:
  /// Throws when the description is compatibilityError, names an operation this
  /// ggml does not have in the split, or does not number the split's tensors
  /// as describeSplit does.
  explicit Graph(const hipep_backend_split_desc &desc);
  ~Graph();
  Graph(const Graph &) = delete;
  Graph &operator=(const Graph &) = delete;

  capture::SplitNodes split;
  capture::SplitTensors tensors;
  /// The split, described.
  capture::SplitDescription description;

  /// The description's tensors of no elements made one row (sorted): a dead
  /// end of the split, as a chunk of a prompt that returns no logits has,
  /// run by the code of one row - its inputs a row of zeros, what it makes
  /// written aside.
  std::vector<int32_t> padded;
  /// Which of the caller's numberings this is ("" for the description's own).
  std::string variant;

private:
  ggml_context *context_ = nullptr;
};

/// A description that outlives the caller's: a compile in the background.
class DescriptionCopy {
public:
  explicit DescriptionCopy(const hipep_backend_split_desc &desc);
  DescriptionCopy(const DescriptionCopy &) = delete;
  DescriptionCopy &operator=(const DescriptionCopy &) = delete;
  const hipep_backend_split_desc &get() const { return desc_; }

private:
  std::vector<hipep_backend_tensor_desc> tensors_;
  std::vector<std::string> strings_;
  hipep_backend_split_desc desc_{};
};

} // namespace hipep_backend

#endif // HIPEP_BACKEND_GRAPH_H
