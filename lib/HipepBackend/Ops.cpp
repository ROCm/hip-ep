/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "Ops.h"

#include "Graph.h"

#include "ggml.h"

#include <cstdlib>
#include <cstring>
#include <set>
#include <sstream>
#include <string>

namespace hipep_backend {
namespace {

bool isDenseType(int32_t type) {
  return type == GGML_TYPE_F32 || type == GGML_TYPE_F16 ||
         type == GGML_TYPE_I32 || type == GGML_TYPE_I64;
}

/// A weight the native K-quant kernels multiply by as llama.cpp stored it.
bool isKQuantType(int32_t type) {
  return type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K ||
         type == GGML_TYPE_Q6_K;
}

bool isView(const std::string &op) {
  return op == "VIEW" || op == "RESHAPE" || op == "PERMUTE" ||
         op == "TRANSPOSE";
}

const std::set<std::string> &supportedOps() {
  static const std::set<std::string> ops = [] {
    std::set<std::string> all = {
        "ADD",     "MUL",      "RMS_NORM",       "MUL_MAT",  "ROPE",
        "GLU",     "GET_ROWS", "FLASH_ATTN_EXT", "SET_ROWS", "VIEW",
        "RESHAPE", "PERMUTE",  "TRANSPOSE",
    };
    const char *filter = std::getenv("HIPEP_BACKEND_OPS");
    if (filter == nullptr || *filter == 0)
      return all;
    std::set<std::string> narrowed;
    std::istringstream names(filter);
    for (std::string name; std::getline(names, name, ',');)
      if (all.count(name) != 0)
        narrowed.insert(name);
    return narrowed;
  }();
  return ops;
}

} // namespace

bool supports(const hipep_backend_split_desc &desc) {
  if (!compatibilityError(desc).empty() || desc.n_tensors == 0)
    return false;
  const hipep_backend_tensor_desc &node = tensorAt(desc, desc.n_tensors - 1);
  const std::string op = node.op ? node.op : "";
  // A leaf in its buffers (a weight) computes nothing.
  if (op == "NONE")
    return true;
  if (supportedOps().count(op) == 0)
    return false;
  // A write into storage the op does not own is a state the split carries:
  // only a cache's rows (SET_ROWS into a view of it).
  if (node.view_src >= 0 && !isView(op) && op != "SET_ROWS")
    return false;
  if (!isDenseType(node.type))
    return false;
  for (int i = 0; i < 10; ++i) {
    const int32_t s = node.src[i];
    if (s < 0 || static_cast<size_t>(s) >= desc.n_tensors)
      continue;
    // A packed weight MUL_MAT multiplies by: the native kernels read its
    // blocks where llama.cpp loaded them.
    const int32_t type = tensorAt(desc, static_cast<size_t>(s)).type;
    if (!isDenseType(type) &&
        !(i == 0 && op == "MUL_MAT" && isKQuantType(type)))
      return false;
  }
  return true;
}

} // namespace hipep_backend
