/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "Ops.h"

#include "Graph.h"

#include "hip/Conversion/GgmlToHip/GgmlToHipLegality.h"
#include "hip/Frontend/Ggml/CaptureGraph.h"
#include "hip/Frontend/Ggml/Error.h"
#include "hip/Frontend/Ggml/GgmlTensor.h"

#include <cstdlib>
#include <cstring>
#include <set>
#include <sstream>
#include <string>
#include <variant>

namespace hipep_backend {
namespace {

bool isDenseType(int32_t type) {
  const std::string name = hip::ggml::typeName(type).value_or("");
  return name == "f32" || name == "f16" || name == "i32" || name == "i64";
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

/// Whether convert-ggml-to-hip lowers the node as it is configured: its
/// op_params, read as the compile reads them, and which optional operands it
/// was given.
bool lowersConfiguration(const hipep_backend_split_desc &desc,
                         const hipep_backend_tensor_desc &node) {
  hip::ggml::Tensor tensor;
  tensor.op = node.op;
  std::memcpy(tensor.opParams, node.op_params, sizeof(tensor.opParams));
  tensor.viewOffs = static_cast<size_t>(node.view_offs);
  capture::NodeParameters parameters;
  try {
    parameters = capture::parametersOf(tensor);
  } catch (const hip::ggml::Error &) {
    return false;
  }
  const auto given = [&](int slot) {
    return node.src[slot] >= 0 &&
           static_cast<size_t>(node.src[slot]) < desc.n_tensors;
  };
  if (const auto *glu = std::get_if<capture::GluParameters>(&parameters))
    return mlir::hip::gluRefusal(glu->kind, given(1), glu->swapped, glu->alpha,
                                 glu->limit)
        .empty();
  if (const auto *rope = std::get_if<capture::RopeParameters>(&parameters))
    return mlir::hip::ropeRefusal(rope->mode, rope->pastTokenCount,
                                  rope->legacyContext, rope->offsetCount,
                                  rope->sections, rope->extensionFactor)
        .empty();
  if (const auto *product = std::get_if<capture::MatmulParameters>(&parameters))
    return mlir::hip::mulMatRefusal(product->precision, product->hint).empty();
  if (const auto *attention =
          std::get_if<capture::FlashAttentionParameters>(&parameters))
    return mlir::hip::flashAttentionRefusal(
               given(3), given(4), attention->scale, attention->maxBias,
               attention->logitSoftcap, attention->precision)
        .empty();
  return !std::holds_alternative<capture::UnsupportedParameters>(parameters);
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
    // A product is by a packed weight: the native kernels read its blocks
    // where llama.cpp loaded them.
    const int32_t type = tensorAt(desc, static_cast<size_t>(s)).type;
    const bool lowered = i == 0 && op == "MUL_MAT"
                             ? mlir::hip::isNativeMatmulWeight(
                                   hip::ggml::typeName(type).value_or(""))
                             : isDenseType(type);
    if (!lowered)
      return false;
  }
  return lowersConfiguration(desc, node);
}

} // namespace hipep_backend
