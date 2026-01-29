/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once
#include "morphizen/vaip_ort.hpp"
#include <iostream>
#include <vaip/vaip_ort_api.h>
namespace vaip_core {
struct VaipOrtApi2 {
#define DECL_VAIP_ORT_OPTIONAL_API_1(ret_type, field)                          \
  template <typename T, class = void>                                          \
  struct field##_t : public std::false_type {                                  \
    template <typename... Args> static ret_type field(Args&&...) {             \
      std::cerr << "VaipOrtApi::" << #field                                    \
                << " is not implemented, please upgrade onnxruntime or apply " \
                   "patches.";                                                 \
      std::abort();                                                            \
    }                                                                          \
  };                                                                           \
  template <typename T>                                                        \
  struct field##_t<T, std::void_t<decltype(std::declval<T&>().field)>>         \
      : public std::true_type {                                                \
    template <typename... Args> static ret_type field(Args&&... args) {        \
      auto api =                                                               \
          static_cast<const T*>(static_cast<const void*>(::vaip_core::api())); \
      if (api->field == nullptr) {                                             \
        std::cerr << "VaipOrtApi::" << #field << " is nullptr";                \
        std::abort();                                                          \
      }                                                                        \
      return api->field(std::forward<Args>(args)...);                          \
    }                                                                          \
  };                                                                           \
  static constexpr auto has_##field =                                          \
      field##_t<::vaip_core::OrtApiForVaip>::value;

#define DECL_VAIP_ORT_OPTIONAL_API(ret_type, field)                            \
  DECL_VAIP_ORT_OPTIONAL_API_1(ret_type, field)                                \
  template <typename... Args> static ret_type field(Args&&... args) {          \
    return field##_t<::vaip_core::OrtApiForVaip>::field(                       \
        std::forward<Args>(args)...);                                          \
  }

#define DECL_VAIP_ORT_OPTIONAL_API_with_fallback(ret_type, field)              \
  DECL_VAIP_ORT_OPTIONAL_API_1(ret_type, field)                                \
  template <typename... Args> static ret_type field

  DECL_VAIP_ORT_OPTIONAL_API_with_fallback(void, graph_set_name)(
      Graph& graph, const std::string& name) {
    if (has_graph_set_name) {
      return graph_set_name_t<::vaip_core::OrtApiForVaip>::graph_set_name(
          graph, name.c_str());
    }
    std::cerr << "VaipOrtApi::graph_set_name is not implemented, fallback to "
                 "graph_set_name_with_default, graph name might not be set.";
    return;
  }
  DECL_VAIP_ORT_OPTIONAL_API(vaip_core::DllSafe<std::string>,
                             attr_proto_release_string)
  DECL_VAIP_ORT_OPTIONAL_API(TensorProto*, tensor_proto_new_u4)
  DECL_VAIP_ORT_OPTIONAL_API(TensorProto*, tensor_proto_new_i4)
  DECL_VAIP_ORT_OPTIONAL_API(bool, is_profiling_enabled)
  DECL_VAIP_ORT_OPTIONAL_API_with_fallback(void, cleanup_vaip)() {
    // do nothing.
  }
};
} // namespace vaip_core
