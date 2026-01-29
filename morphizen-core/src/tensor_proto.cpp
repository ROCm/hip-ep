/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//
#include <exception>
#include <glog/logging.h>
// include glog/logging.h to define CHECK before include plugin.hpp

#include "./config.hpp"
#include "./pass_imp.hpp"

#include "morphizen/util.hpp"

#include "morphizen/env_config.hpp"
#include "morphizen/morphizen_core.hpp"
#include "morphizen/ort_api_wrapper.hpp"
#include "morphizen/plugin.hpp"
#include "morphizen/tensor_proto.hpp"
#include "version_info.hpp"
#include <morphizen/custom_op.h>
#include <morphizen/my_ort.h>

#include <memory>

namespace morphizen {
MORPHIZEN_DLL_SPEC std::vector<int64_t>
tensor_proto_get_shape(const TensorProto& tensor_proto) {
  auto shape = MORPHIZEN_ORT_API(tensor_proto_get_shape_unsafe)(tensor_proto);
  CHECK(shape.get() != nullptr)
      << "tensor_proto_get_shape_unsafe should not return null shape";
  return *shape;
}

MORPHIZEN_DLL_SPEC TensorProtoPtr tensor_proto_new_floats(
    const std::string& name, const std::vector<int64_t>& shape,
    const std::vector<float>& data) {
  return TensorProtoPtr(
      MORPHIZEN_ORT_API(tensor_proto_new_floats)(name, shape, data));
}
#if MORPHIZEN_ORT_API_MAJOR >= 3

MORPHIZEN_DLL_SPEC
TensorProtoPtr tensor_proto_new_doubles(const std::string& name,
                                        const std::vector<int64_t>& shape,
                                        const std::vector<double>& data) {
  return TensorProtoPtr(
      MORPHIZEN_ORT_API(tensor_proto_new_doubles)(name, shape, data));
}

MORPHIZEN_DLL_SPEC TensorProtoPtr tensor_proto_new_bf16(
    const std::string& name, const std::vector<int64_t>& shape,
    const std::vector<int16_t>& data) {
  return TensorProtoPtr(
      MORPHIZEN_ORT_API(tensor_proto_new_bf16)(name, shape, data));
}

MORPHIZEN_DLL_SPEC TensorProtoPtr tensor_proto_new_fp16(
    const std::string& name, const std::vector<int64_t>& shape,
    const std::vector<int16_t>& data) {
  return TensorProtoPtr(
      MORPHIZEN_ORT_API(tensor_proto_new_fp16)(name, shape, data));
}
#endif

MORPHIZEN_DLL_SPEC
TensorProtoPtr tensor_proto_new_i32(const std::string& name,
                                    const std::vector<int64_t>& shape,
                                    const std::vector<int32_t>& data) {
  return TensorProtoPtr(
      MORPHIZEN_ORT_API(tensor_proto_new_i32)(name, shape, data));
}
MORPHIZEN_DLL_SPEC TensorProtoPtr
tensor_proto_new_i64(const std::string& name, const std::vector<int64_t>& shape,
                     const std::vector<int64_t>& data) {
  return TensorProtoPtr(
      MORPHIZEN_ORT_API(tensor_proto_new_i64)(name, shape, data));
}
MORPHIZEN_DLL_SPEC TensorProtoPtr
tensor_proto_new_i8(const std::string& name, const std::vector<int64_t>& shape,
                    const std::vector<int8_t>& data) {
  return TensorProtoPtr(
      MORPHIZEN_ORT_API(tensor_proto_new_i8)(name, shape, data));
}

#if MORPHIZEN_ORT_API_MAJOR >= 3
MORPHIZEN_DLL_SPEC
TensorProtoPtr tensor_proto_new_i16(const std::string& name,
                                    const std::vector<int64_t>& shape,
                                    const std::vector<int16_t>& data) {
  return TensorProtoPtr(
      MORPHIZEN_ORT_API(tensor_proto_new_i16)(name, shape, data));
}

MORPHIZEN_DLL_SPEC TensorProtoPtr
tensor_proto_new_u8(const std::string& name, const std::vector<int64_t>& shape,
                    const std::vector<uint8_t>& data) {
  return TensorProtoPtr(
      MORPHIZEN_ORT_API(tensor_proto_new_u8)(name, shape, data));
}

MORPHIZEN_DLL_SPEC TensorProtoPtr
tensor_proto_new_u16(const std::string& name, const std::vector<int64_t>& shape,
                     const std::vector<uint16_t>& data) {
  return TensorProtoPtr(
      MORPHIZEN_ORT_API(tensor_proto_new_u16)(name, shape, data));
}

MORPHIZEN_DLL_SPEC TensorProtoPtr
tensor_proto_new_u32(const std::string& name, const std::vector<int64_t>& shape,
                     const std::vector<uint32_t>& data) {
  return TensorProtoPtr(
      MORPHIZEN_ORT_API(tensor_proto_new_u32)(name, shape, data));
}

MORPHIZEN_DLL_SPEC TensorProtoPtr
tensor_proto_new_u64(const std::string& name, const std::vector<int64_t>& shape,
                     const std::vector<uint64_t>& data) {
  return TensorProtoPtr(
      MORPHIZEN_ORT_API(tensor_proto_new_u64)(name, shape, data));
}

#endif

MORPHIZEN_DLL_SPEC
TensorProtoPtr tensor_proto_new_i4(const std::string& name,
                                   const std::vector<int64_t>& shape,
                                   const std::vector<int8_t>& data) {
  return TensorProtoPtr(
      MorphizenOrtApi2::tensor_proto_new_i4(name, shape, data));
}

MORPHIZEN_DLL_SPEC TensorProtoPtr
tensor_proto_new_u4(const std::string& name, const std::vector<int64_t>& shape,
                    const std::vector<uint8_t>& data) {
  return TensorProtoPtr(
      MorphizenOrtApi2::tensor_proto_new_u4(name, shape, data));
}

#if MORPHIZEN_ORT_API_MAJOR >= 19
MORPHIZEN_DLL_SPEC
TensorProtoPtr tensor_proto_new_bool(const std::string& name,
                                     const std::vector<int64_t>& shape,
                                     const std::vector<uint8_t>& data) {
  return TensorProtoPtr(
      MORPHIZEN_ORT_API(tensor_proto_new_bool)(name, shape, data));
}
#endif

MORPHIZEN_DLL_SPEC gsl::span<const char>
tensor_proto_as_raw(const onnxruntime::Graph& graph,
                    const TensorProto& tensor_proto) {
#if MORPHIZEN_ORT_API_MAJOR >= 9
  auto raw_data = MORPHIZEN_ORT_API(tensor_proto_as_raw)(graph, tensor_proto);
#else
  auto raw_data = MORPHIZEN_ORT_API(tensor_proto_as_raw)(tensor_proto);
#endif
  return raw_data;
}

template <typename T>
static gsl::span<const T> tensor_proto_as(const onnxruntime::Graph& graph,
                                          const TensorProto& tensor_proto,
                                          int data_type) {

  auto tensor_data_type =
      MORPHIZEN_ORT_API(tensor_proto_data_type)(tensor_proto);
  CHECK_EQ(tensor_data_type, data_type);
  auto raw_data = tensor_proto_as_raw(graph, tensor_proto);
  auto p = reinterpret_cast<const T*>(raw_data.data());
  auto num_of_element = raw_data.size() / sizeof(T);
  return gsl::span<const T>(p, p + num_of_element);
}

MORPHIZEN_DLL_SPEC float tensor_proto_as_float(const onnxruntime::Graph& graph,
                                               const TensorProto& tensor) {
  auto v = tensor_proto_as_floats(graph, tensor);
  auto shape = tensor_proto_get_shape(tensor);
  // CHECK(shape.empty()) << "tensor proto is not float";
  CHECK_EQ(v.size(), 1u);
  return v[0];
}

MORPHIZEN_DLL_SPEC double
tensor_proto_as_double(const onnxruntime::Graph& graph,
                       const TensorProto& tensor) {
  auto v = tensor_proto_as_doubles(graph, tensor);
  auto shape = tensor_proto_get_shape(tensor);
  CHECK(shape.empty()) << "tensor proto is not double";
  CHECK_EQ(v.size(), 1u);
  return v[0];
}

MORPHIZEN_DLL_SPEC int16_t tensor_proto_as_bf16(const onnxruntime::Graph& graph,
                                                const TensorProto& tensor) {
  auto v = tensor_proto_as_bf16s(graph, tensor);
  auto shape = tensor_proto_get_shape(tensor);
  CHECK(shape.empty()) << "tensor proto is not bf16";
  CHECK_EQ(v.size(), 1u);
  return v[0];
}

MORPHIZEN_DLL_SPEC int16_t tensor_proto_as_fp16(const onnxruntime::Graph& graph,
                                                const TensorProto& tensor) {
  auto v = tensor_proto_as_fp16s(graph, tensor);
  auto shape = tensor_proto_get_shape(tensor);
  CHECK(shape.empty()) << "tensor proto is not fp16";
  CHECK_EQ(v.size(), 1u);
  return v[0];
}

MORPHIZEN_DLL_SPEC int8_t tensor_proto_as_i8(const onnxruntime::Graph& graph,
                                             const TensorProto& tensor) {
  auto v = tensor_proto_as_i8s(graph, tensor);
  auto shape = tensor_proto_get_shape(tensor);
  CHECK(shape.empty()) << "tensor proto is not i8s";
  CHECK_EQ(v.size(), 1u);
  return v[0];
}
MORPHIZEN_DLL_SPEC uint8_t tensor_proto_as_u8(const onnxruntime::Graph& graph,
                                              const TensorProto& tensor) {
  auto v = tensor_proto_as_u8s(graph, tensor);
  auto shape = tensor_proto_get_shape(tensor);
  CHECK(shape.empty()) << "tensor proto is not u8s";
  CHECK_EQ(v.size(), 1u);
  return v[0];
}

MORPHIZEN_DLL_SPEC int16_t tensor_proto_as_i16(const onnxruntime::Graph& graph,
                                               const TensorProto& tensor) {
  auto v = tensor_proto_as_i16s(graph, tensor);
  auto shape = tensor_proto_get_shape(tensor);
  CHECK(shape.empty()) << "tensor proto is not i16s";
  CHECK_EQ(v.size(), 1u);
  return v[0];
}

MORPHIZEN_DLL_SPEC uint16_t tensor_proto_as_u16(const onnxruntime::Graph& graph,
                                                const TensorProto& tensor) {
  auto v = tensor_proto_as_u16s(graph, tensor);
  auto shape = tensor_proto_get_shape(tensor);
  CHECK(shape.empty()) << "tensor proto is not u16s";
  CHECK_EQ(v.size(), 1u);
  return v[0];
}

MORPHIZEN_DLL_SPEC int32_t tensor_proto_as_i32(const onnxruntime::Graph& graph,
                                               const TensorProto& tensor) {
  auto v = tensor_proto_as_i32s(graph, tensor);
  auto shape = tensor_proto_get_shape(tensor);
  CHECK(shape.empty()) << "tensor proto is not i32s";
  CHECK_EQ(v.size(), 1u);
  return v[0];
}

MORPHIZEN_DLL_SPEC uint32_t tensor_proto_as_u32(const onnxruntime::Graph& graph,
                                                const TensorProto& tensor) {
  auto v = tensor_proto_as_u32s(graph, tensor);
  auto shape = tensor_proto_get_shape(tensor);
  CHECK(shape.empty()) << "tensor proto is not u32s";
  CHECK_EQ(v.size(), 1u);
  return v[0];
}

MORPHIZEN_DLL_SPEC int64_t tensor_proto_as_i64(const onnxruntime::Graph& graph,
                                               const TensorProto& tensor) {
  auto v = tensor_proto_as_i64s(graph, tensor);
  auto shape = tensor_proto_get_shape(tensor);
  CHECK(shape.empty()) << "tensor proto is not i64";
  CHECK_EQ(v.size(), 1u);
  return v[0];
}

MORPHIZEN_DLL_SPEC uint64_t tensor_proto_as_u64(const onnxruntime::Graph& graph,
                                                const TensorProto& tensor) {
  auto v = tensor_proto_as_u64s(graph, tensor);
  auto shape = tensor_proto_get_shape(tensor);
  CHECK(shape.empty()) << "tensor proto is not u64";
  CHECK_EQ(v.size(), 1u);
  return v[0];
}

void TensorProtoDeleter::operator()(TensorProto* p) const {
  MORPHIZEN_ORT_API(tensor_proto_delete)(p);
}

MORPHIZEN_DLL_SPEC
int8_t get_int4_value(gsl::span<const int8_t> data, size_t idx) {
  size_t byte_idx = idx / 2;
  int8_t value = data[byte_idx];
  if (idx & 1) { // odd, upper
    value = static_cast<int8_t>(static_cast<uint8_t>(value) >> 4);
  } else {
    value = value & 0xf;
  }
  if (value > 7) {
    value -= 16;
  }
  return value;
}

MORPHIZEN_DLL_SPEC
uint8_t get_uint4_value(gsl::span<const uint8_t> data, size_t idx) {
  size_t byte_idx = idx / 2;
  uint8_t value = data[byte_idx];
  if (idx & 1) { // odd, upper
    value = value >> 4;
  } else {
    value = value & 0xf;
  }
  return value;
}

MORPHIZEN_DLL_SPEC
gsl::span<const int8_t> tensor_proto_as_i4s(const onnxruntime::Graph& graph,
                                            const TensorProto& tensor) {
  return tensor_proto_as<int8_t>(graph, tensor,

                                 onnx::TensorProto_DataType_INT4);
}
MORPHIZEN_DLL_SPEC
gsl::span<const uint8_t> tensor_proto_as_u4s(const onnxruntime::Graph& graph,
                                             const TensorProto& tensor) {
  return tensor_proto_as<uint8_t>(graph, tensor,
                                  onnx::TensorProto_DataType_UINT4);
}

MORPHIZEN_DLL_SPEC
gsl::span<const int8_t> tensor_proto_as_i8s(const onnxruntime::Graph& graph,
                                            const TensorProto& tensor) {
  return tensor_proto_as<int8_t>(graph, tensor,
                                 onnx::TensorProto_DataType_INT8);
}
MORPHIZEN_DLL_SPEC
gsl::span<const uint8_t> tensor_proto_as_u8s(const onnxruntime::Graph& graph,
                                             const TensorProto& tensor) {
  return tensor_proto_as<uint8_t>(graph, tensor,
                                  onnx::TensorProto_DataType_UINT8);
}
MORPHIZEN_DLL_SPEC
gsl::span<const uint16_t> tensor_proto_as_u16s(const onnxruntime::Graph& graph,
                                               const TensorProto& tensor) {
  return tensor_proto_as<uint16_t>(graph, tensor,
                                   onnx::TensorProto_DataType_UINT16);
}
MORPHIZEN_DLL_SPEC
gsl::span<const int16_t> tensor_proto_as_i16s(const onnxruntime::Graph& graph,
                                              const TensorProto& tensor) {
  return tensor_proto_as<int16_t>(graph, tensor,
                                  onnx::TensorProto_DataType_INT16);
}

MORPHIZEN_DLL_SPEC
gsl::span<const int32_t> tensor_proto_as_i32s(const onnxruntime::Graph& graph,
                                              const TensorProto& tensor) {
  return tensor_proto_as<int32_t>(graph, tensor,
                                  onnx::TensorProto_DataType_INT32);
}

MORPHIZEN_DLL_SPEC
gsl::span<const uint32_t> tensor_proto_as_u32s(const onnxruntime::Graph& graph,
                                               const TensorProto& tensor) {
  return tensor_proto_as<uint32_t>(graph, tensor,
                                   onnx::TensorProto_DataType_UINT32);
}

MORPHIZEN_DLL_SPEC
gsl::span<const int64_t> tensor_proto_as_i64s(const onnxruntime::Graph& graph,
                                              const TensorProto& tensor) {
  return tensor_proto_as<int64_t>(graph, tensor,
                                  onnx::TensorProto_DataType_INT64);
}
MORPHIZEN_DLL_SPEC
gsl::span<const uint64_t> tensor_proto_as_u64s(const onnxruntime::Graph& graph,
                                               const TensorProto& tensor) {
  return tensor_proto_as<uint64_t>(graph, tensor,
                                   onnx::TensorProto_DataType_UINT64);
}

MORPHIZEN_DLL_SPEC
gsl::span<const float> tensor_proto_as_floats(const onnxruntime::Graph& graph,
                                              const TensorProto& tensor) {
  return tensor_proto_as<float>(graph, tensor,
                                onnx::TensorProto_DataType_FLOAT);
}

MORPHIZEN_DLL_SPEC
gsl::span<const double> tensor_proto_as_doubles(const onnxruntime::Graph& graph,
                                                const TensorProto& tensor) {
  return tensor_proto_as<double>(graph, tensor,
                                 onnx::TensorProto_DataType_DOUBLE);
}

MORPHIZEN_DLL_SPEC
gsl::span<const int16_t> tensor_proto_as_bf16s(const onnxruntime::Graph& graph,
                                               const TensorProto& tensor) {
  return tensor_proto_as<int16_t>(graph, tensor,
                                  onnx::TensorProto_DataType_BFLOAT16);
}

MORPHIZEN_DLL_SPEC
gsl::span<const int16_t> tensor_proto_as_fp16s(const onnxruntime::Graph& graph,
                                               const TensorProto& tensor) {
  return tensor_proto_as<int16_t>(graph, tensor,
                                  onnx::TensorProto_DataType_FLOAT16);
}
} // namespace morphizen
