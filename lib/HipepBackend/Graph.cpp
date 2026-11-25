/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "Graph.h"

#include "ggml.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <map>
#include <stdexcept>

namespace hipep_backend {
namespace {

/// This library's ggml, as the description spells a version.
uint32_t libraryGgmlVersion() {
  static const uint32_t version = [] {
    unsigned major = 0, minor = 0, patch = 0;
    std::sscanf(ggml_version(), "%u.%u.%u", &major, &minor, &patch);
    return major * 10000 + minor * 100 + patch;
  }();
  return version;
}

/// An operation by ggml's name for it, GGML_OP_COUNT for none.
ggml_op opFromName(const char *name) {
  static const std::map<std::string, ggml_op> ops = [] {
    std::map<std::string, ggml_op> result;
    for (int op = 0; op < GGML_OP_COUNT; ++op)
      result.emplace(ggml_op_name(static_cast<ggml_op>(op)),
                     static_cast<ggml_op>(op));
    return result;
  }();
  if (name == nullptr)
    return GGML_OP_COUNT;
  auto found = ops.find(name);
  return found == ops.end() ? GGML_OP_COUNT : found->second;
}

} // namespace

const hipep_backend_tensor_desc &tensorAt(const hipep_backend_split_desc &desc,
                                          size_t i) {
  return *reinterpret_cast<const hipep_backend_tensor_desc *>(
      reinterpret_cast<const char *>(desc.tensors) + i * desc.tensor_size);
}

std::string compatibilityError(const hipep_backend_split_desc &desc) {
  if (desc.abi != HIPEP_BACKEND_ABI_VERSION)
    return "a description of ABI " + std::to_string(desc.abi) + ", not " +
           std::to_string(HIPEP_BACKEND_ABI_VERSION);
  if (desc.tensor_size < sizeof(hipep_backend_tensor_desc))
    return "tensor descriptions of " + std::to_string(desc.tensor_size) +
           " bytes";
  if (desc.n_tensors > 0 && desc.tensors == nullptr)
    return "no tensors";
  // The operations' parameters are laid out as the ggml of one minor version
  // lays them out.
  if (desc.ggml_version / 100 != libraryGgmlVersion() / 100)
    return "a split of ggml " + std::to_string(desc.ggml_version) +
           "; this library reads ggml " +
           std::to_string(libraryGgmlVersion() / 100) + "xx";
  return "";
}

Graph::Graph(const hipep_backend_split_desc &desc) {
  if (std::string error = compatibilityError(desc); !error.empty())
    throw std::runtime_error(error);
  const size_t n = desc.n_tensors;
  ggml_init_params params{(n + 1) * ggml_tensor_overhead(), nullptr,
                          /*no_alloc=*/true};
  context_ = ggml_init(params);
  if (context_ == nullptr)
    throw std::runtime_error("no memory for the split's tensors");
  std::vector<ggml_tensor *> all(n);
  const auto tensorOf = [&](int32_t i) -> ggml_tensor * {
    if (i < 0)
      return nullptr;
    if (static_cast<size_t>(i) >= n)
      throw std::runtime_error("a tensor names tensor " + std::to_string(i) +
                               " of " + std::to_string(n));
    return all[static_cast<size_t>(i)];
  };
  for (size_t i = 0; i < n; ++i) {
    const hipep_backend_tensor_desc &t = tensorAt(desc, i);
    if (t.type < 0 || t.type >= GGML_TYPE_COUNT ||
        ggml_blck_size(static_cast<ggml_type>(t.type)) == 0)
      throw std::runtime_error("tensor " + std::to_string(i) +
                               " is of no type this ggml has");
    for (int64_t e : t.ne)
      if (e < 0)
        throw std::runtime_error("tensor " + std::to_string(i) +
                                 " has a negative extent");
    ggml_tensor *x =
        ggml_new_tensor(context_, static_cast<ggml_type>(t.type), 4, t.ne);
    for (int d = 0; d < 4; ++d)
      x->nb[d] = static_cast<size_t>(t.nb[d]);
    ggml_op op = opFromName(t.op);
    if (op == GGML_OP_COUNT)
      throw std::runtime_error(std::string("the split names ") +
                               (t.op ? t.op : "(null)") +
                               ", which this ggml does not have");
    x->op = op;
    static_assert(sizeof(t.op_params) == sizeof(x->op_params),
                  "op_params differ");
    std::memcpy(x->op_params, t.op_params, sizeof(x->op_params));
    x->flags = t.flags;
    if (t.name != nullptr)
      ggml_set_name(x, t.name);
    all[i] = x;
  }
  for (size_t i = 0; i < n; ++i) {
    const hipep_backend_tensor_desc &t = tensorAt(desc, i);
    ggml_tensor *x = all[i];
    if (x->op != GGML_OP_NONE)
      for (int s = 0; s < GGML_MAX_SRC; ++s)
        x->src[s] = tensorOf(t.src[s]);
    x->view_src = tensorOf(t.view_src);
    x->view_offs = static_cast<size_t>(t.view_offs);
    if (t.in_split) {
      split.nodes.push_back(x);
      split.exported.push_back(t.exported != 0);
    }
    if (t.role == HIPEP_BACKEND_ROLE_WEIGHT)
      tensors.weights.insert(x);
    if (t.role == HIPEP_BACKEND_ROLE_KV_CACHE)
      tensors.caches.insert(x);
  }
  // Tensors of no elements (a chunk's rows of no logits, and all made of
  // them): one row, the code of one row's. The split is computable only when
  // nothing of elements reads them.
  // (Sources first: a view's source is one row before it is.)
  std::vector<const ggml_tensor *> emptyTensors;
  for (ggml_tensor *x : all) {
    if (ggml_nelements(x) != 0)
      continue;
    if (ggml_blck_size(x->type) != 1 || x->view_offs != 0)
      throw std::runtime_error(
          "a quantized tensor, or a view inside another, of no elements");
    int first = -1;
    for (int d = 0; d < 4; ++d)
      if (x->ne[d] == 0) {
        first = first < 0 ? d : first;
        x->ne[d] = 1;
      }
    if (x->op == GGML_OP_PERMUTE || x->op == GGML_OP_TRANSPOSE) {
      // Its source's strides, relabelled (result.nb[axis[i]] = src.nb[i]).
      const int32_t transpose[4] = {1, 0, 2, 3};
      const int32_t *axes =
          x->op == GGML_OP_TRANSPOSE ? transpose : x->op_params;
      for (int i = 0; i < 4; ++i)
        x->nb[axes[i]] = x->src[0]->nb[i];
    } else {
      // The strides past the first extent of none are its multiples.
      for (int d = first + 1; d < 4; ++d)
        x->nb[d] = x->nb[d - 1] * static_cast<size_t>(x->ne[d - 1]);
    }
    emptyTensors.push_back(x);
  }

  std::map<const ggml_tensor *, int32_t> indexOf;
  for (size_t i = 0; i < n; ++i)
    indexOf.emplace(all[i], static_cast<int32_t>(i));
  description = capture::describeSplit(split, tensors);
  for (size_t i = 0; i < description.originals.size(); ++i) {
    auto found = indexOf.find(description.originals[i]);
    if (found == indexOf.end())
      throw std::runtime_error(
          "the split describes a tensor the caller did not (tensor " +
          std::to_string(i) + ")");
    if (found->second != static_cast<int32_t>(i))
      throw std::runtime_error(
          "the split's tensors are not numbered sources first (tensor " +
          std::to_string(i) + ")");
  }
  if (description.originals.size() != n)
    throw std::runtime_error("the caller describes tensors the split does "
                             "not read");
  for (size_t i = 0; i < description.originals.size(); ++i)
    if (std::find(emptyTensors.begin(), emptyTensors.end(),
                  description.originals[i]) != emptyTensors.end())
      padded.push_back(static_cast<int32_t>(i));
  if (!padded.empty())
    variant += "#padded";
}

Graph::~Graph() {
  if (context_ != nullptr)
    ggml_free(context_);
}

DescriptionCopy::DescriptionCopy(const hipep_backend_split_desc &desc) {
  tensors_.reserve(desc.n_tensors);
  strings_.reserve(2 * desc.n_tensors);
  for (size_t i = 0; i < desc.n_tensors; ++i) {
    hipep_backend_tensor_desc t = tensorAt(desc, i);
    strings_.emplace_back(t.op ? t.op : "");
    strings_.emplace_back(t.name ? t.name : "");
    tensors_.push_back(t);
  }
  // The strings no longer move.
  for (size_t i = 0; i < tensors_.size(); ++i) {
    tensors_[i].op = strings_[2 * i].c_str();
    tensors_[i].name = strings_[2 * i + 1].c_str();
  }
  desc_ = desc;
  desc_.tensor_size = sizeof(hipep_backend_tensor_desc);
  desc_.tensors = tensors_.data();
}

} // namespace hipep_backend
