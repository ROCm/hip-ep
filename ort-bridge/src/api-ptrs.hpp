/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once
#include "./ort-array-span.hpp"
#include "./ort-status-exception.hpp"
// bugfix for link error in Ort::ConstValueInfo::Name() method
// undefined reference to `Ort::detail::ValueInfoImpl<Ort::detail::Unowned<const
// OrtValueInfo>>::Name() const'
//
// I have to put it here because on linux there is an error, like
// cannot instantiate a template after specialization
//
namespace Ort {

namespace detail {
template <>
inline std::string
ValueInfoImpl<Ort::detail::Unowned<const OrtValueInfo>>::Name() const {
  const char* name = nullptr;
  ThrowOnError(GetApi().GetValueInfoName(this->p_, &name));
  return name;
}

template <>
inline ConstTypeInfo
ValueInfoImpl<Ort::detail::Unowned<const OrtValueInfo>>::TypeInfo() const {
  const OrtTypeInfo* type_info = nullptr;
  ThrowOnError(GetApi().GetValueInfoTypeInfo(this->p_, &type_info));
  return ConstTypeInfo{type_info};
}
} // namespace detail
} // namespace Ort
// end bugfix
#include <string>
namespace onnxruntime {
struct Model;
struct Graph;
} // namespace onnxruntime

namespace morphizen {

// Forward declaration
struct Graph;

struct ApiPtrs {
  const OrtApi& ort_api;
  const OrtEpApi& ep_api; // Method declarations
  void throw_if_error(OrtStatus* status) const;
  void throw_error(const std::string& message) const;

  // Factory method to create managed array spans
  template <typename T>
  OrtArraySpan<T> make_array_span(OrtArrayOfConstObjects* array) const {
    return OrtArraySpan<T>(this, array);
  }
};

using GraphUniquePtr =
    std::unique_ptr<onnxruntime::Graph, void (*)(onnxruntime::Graph*)>;
using ModelUniquePtr =
    std::unique_ptr<onnxruntime::Model, void (*)(onnxruntime::Model*)>;

} // namespace morphizen
#include "./ort-array-span.inl"
