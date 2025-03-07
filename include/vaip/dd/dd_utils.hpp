/*
 *     The Xilinx Vitis AI Vaip in this distribution are provided under the
 * following free and permissive binary-only license, but are not provided in
 * source code form.  While the following free and permissive license is similar
 * to the BSD open source license, it is NOT the BSD open source license nor
 * other OSI-approved open source license.
 *
 *      Copyright (C) 2022 Xilinx, Inc. All rights reserved.
 *      Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights
 * reserved.
 *
 *      Redistribution and use in binary form only, without modification, is
 * permitted provided that the following conditions are met:
 *
 *      1. Redistributions must reproduce the above copyright notice, this list
 * of conditions and the following disclaimer in the documentation and/or other
 * materials provided with the distribution.
 *
 *      2. The name of Xilinx, Inc. may not be used to endorse or promote
 * products redistributed with this software without specific prior written
 * permission.
 *
 *      THIS SOFTWARE IS PROVIDED BY XILINX, INC. "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL XILINX, INC. BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 *      PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE
 */
#pragma once
#include "vaip/vaip.hpp"
#include "vitis/ai/env_config.hpp"
#include <cmath>
#include <functional>
#include <glog/logging.h>
#include <numeric>

using namespace vaip_core;
namespace vaip::dd {
[[maybe_unused]] static std::vector<std::string>
get_node_names(onnxruntime::Graph* graph, binder_t& binder) {
  // Get nodes
  std::vector<std::string> attr_nodes;
  for (auto& ni : binder) {
    if (!(*node_arg_is_constant)(*graph, *ni.second.node_arg)) {
      attr_nodes.push_back(node_arg_get_name(*ni.second.node_arg));
    }
  }
  return attr_nodes;
}

[[maybe_unused]] static uint16_t get_zp_from_node(onnxruntime::Graph& graph,
                                                  const NodeArg& node_arg) {
  auto out_dtype = node_arg_get_element_type(node_arg);

  if (out_dtype == (int)ONNX_NAMESPACE::TensorProto_DataType_UINT8) {
    uint16_t zp =
        static_cast<uint16_t>(node_arg_get_const_data_as_u8(graph, node_arg));
    return zp;
  } else if (out_dtype == (int)ONNX_NAMESPACE::TensorProto_DataType_UINT16) {
    uint16_t zp = node_arg_get_const_data_as_u16(graph, node_arg);
    return zp;
  } else if (out_dtype == (int)ONNX_NAMESPACE::TensorProto_DataType_INT32) {
    auto zp = node_arg_get_const_data_as_i32(graph, node_arg);
    CHECK(zp <= 65535 && zp >= 0) << "Max value exceded";
    return (uint16_t)zp;
  } else {
    CHECK(out_dtype == (int)ONNX_NAMESPACE::TensorProto_DataType_UINT8 ||
          out_dtype == (int)ONNX_NAMESPACE::TensorProto_DataType_UINT16)
        << "Currently only uint8 and uint16 are supported";
    return 42;
  }
}
[[maybe_unused]] static std::vector<uint16_t>
get_const_as_uint16_t(onnxruntime::Graph& graph, const NodeArg& node_arg) {
  auto out_dtype = node_arg_get_element_type(node_arg);

  if (out_dtype == (int)ONNX_NAMESPACE::TensorProto_DataType_UINT8) {
    gsl::span<const uint8_t> uint8_data =
        node_arg_get_const_data_as_u8s(graph, node_arg);
    std::vector<uint16_t> r(uint8_data.begin(), uint8_data.end());
    // std::vector<uint16_t> r(uint8_data.size());
    // for (size_t i = 0; i < uint8_data.size(); ++i)
    //   r[i] = uint8_data[i];
    // gsl::span<const uint16_t> ret(r);
    // return ret;
    return r;
  } else if (out_dtype == (int)ONNX_NAMESPACE::TensorProto_DataType_UINT16) {
    auto uint16_data = node_arg_get_const_data_as_u16s(graph, node_arg);
    std::vector<uint16_t> r(uint16_data.begin(), uint16_data.end());
    // std::vector<uint16_t> r(uint8_data.size());
    return r;
  }
  throw std::runtime_error("Other than uint8 and uint16, format not supported");
}

[[maybe_unused]] static std::vector<int32_t>
get_const_as_int32_t(onnxruntime::Graph& graph, const NodeArg& node_arg) {
  auto out_dtype = node_arg_get_element_type(node_arg);

  if (out_dtype == (int)ONNX_NAMESPACE::TensorProto_DataType_INT32) {
    gsl::span<const int32_t> int32_data =
        node_arg_get_const_data_as_i32s(graph, node_arg);
    std::vector<int32_t> r(int32_data.begin(), int32_data.end());
    return r;
  }
  throw std::runtime_error("Other than int32, format not supported");
}
// [[maybe_unused]] static std::string nodearg_dtype_to_string(const NodeArg& a)
// {
//   auto out_dtype = node_arg_get_element_type(a);
//   if (out_dtype == (int)ONNX_NAMESPACE::TensorProto_DataType_UINT8)
//     return std::string("uint8");
//   else if (out_dtype == (int)ONNX_NAMESPACE::TensorProto_DataType_UINT16)
//     return std::string("uint16");
//   else if (out_dtype == (int)ONNX_NAMESPACE::TensorProto_DataType_UINT32)
//     return std::string("uint32");
//   else if (out_dtype == (int)ONNX_NAMESPACE::TensorProto_DataType_UINT64)
//     return std::string("uint64");
//   else if (out_dtype == (int)ONNX_NAMESPACE::TensorProto_DataType_FLOAT)
//     return std::string("float");
//   else if (out_dtype == (int)ONNX_NAMESPACE::TensorProto_DataType_INT8)
//     return std::string("int8");
//   else if (out_dtype == (int)ONNX_NAMESPACE::TensorProto_DataType_INT16)
//     return std::string("int16");
//   else if (out_dtype == (int)ONNX_NAMESPACE::TensorProto_DataType_INT32)
//     return std::string("int32");
//   else if (out_dtype == (int)ONNX_NAMESPACE::TensorProto_DataType_INT64)
//     return std::string("int64");
//   else if (out_dtype == (int)ONNX_NAMESPACE::TensorProto_DataType_BOOL)
//     return std::string("bool");
//   else if (out_dtype == (int)ONNX_NAMESPACE::TensorProto_DataType_STRING)
//     return std::string("string");
//   else if (out_dtype == (int)ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16)
//     return std::string("bfloat16");
//   return std::string("unknown");
// }

template <typename T>
static NodeArg&
insert_named_tensor_in_graph(onnxruntime::Graph* graph, std::string tensor_name,
                             std::vector<T> data,
                             const std::vector<int64_t>& shape) {
  if constexpr (std::is_same<T, int32_t>::value) {
    auto n_tensor = tensor_proto_new_i32(tensor_name, shape, data);
    VAIP_ORT_API(graph_add_initialized_tensor)(*graph, *n_tensor);
    auto& n_arg =
        VAIP_ORT_API(node_arg_new)(*graph, tensor_name, &shape,
                                   ONNX_NAMESPACE::TensorProto_DataType_INT32);
    return n_arg;
  } else if constexpr (std::is_same<T, int64_t>::value) {
    auto n_tensor = tensor_proto_new_i64(tensor_name, shape, data);
    VAIP_ORT_API(graph_add_initialized_tensor)(*graph, *n_tensor);
    auto& n_arg =
        VAIP_ORT_API(node_arg_new)(*graph, tensor_name, &shape,
                                   ONNX_NAMESPACE::TensorProto_DataType_INT64);
    return n_arg;
  } else if constexpr (std::is_same<T, uint16_t>::value) {
    auto n_tensor = tensor_proto_new_u16(tensor_name, shape, data);
    VAIP_ORT_API(graph_add_initialized_tensor)(*graph, *n_tensor);
    auto& n_arg =
        VAIP_ORT_API(node_arg_new)(*graph, tensor_name, &shape,
                                   ONNX_NAMESPACE::TensorProto_DataType_UINT16);
    return n_arg;
  } else if constexpr (std::is_same<T, uint8_t>::value) {
    auto n_tensor = tensor_proto_new_u8(tensor_name, shape, data);
    VAIP_ORT_API(graph_add_initialized_tensor)(*graph, *n_tensor);
    auto& n_arg =
        VAIP_ORT_API(node_arg_new)(*graph, tensor_name, &shape,
                                   ONNX_NAMESPACE::TensorProto_DataType_UINT8);
    return n_arg;
  } else if constexpr (std::is_same<T, int8_t>::value) {
    auto n_tensor = tensor_proto_new_i8(tensor_name, shape, data);
    VAIP_ORT_API(graph_add_initialized_tensor)(*graph, *n_tensor);
    auto& n_arg = VAIP_ORT_API(node_arg_new)(
        *graph, tensor_name, &shape, ONNX_NAMESPACE::TensorProto_DataType_INT8);
    return n_arg;
  } else if constexpr (std::is_same<T, int16_t>::value) {
    auto n_tensor = tensor_proto_new_i16(tensor_name, shape, data);
    VAIP_ORT_API(graph_add_initialized_tensor)(*graph, *n_tensor);
    auto& n_arg =
        VAIP_ORT_API(node_arg_new)(*graph, tensor_name, &shape,
                                   ONNX_NAMESPACE::TensorProto_DataType_INT16);
    return n_arg;
  } else if constexpr (std::is_same<T, float>::value) {
    auto n_tensor = tensor_proto_new_f32(tensor_name, shape, data);
    VAIP_ORT_API(graph_add_initialized_tensor)(*graph, *n_tensor);
    auto& n_arg =
        VAIP_ORT_API(node_arg_new)(*graph, tensor_name, &shape,
                                   ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    return n_arg;
  }
}
[[maybe_unused]] static std::string
shape_as_string(const std::vector<int64_t>& shape) {
  std::ostringstream str;
  str << "[";
  int c = 0;
  for (auto s : shape) {
    if (c != 0) {
      str << ",";
    }
    str << s;
    c = c + 1;
  }
  str << "]";
  return str.str();
}
static int64_t reduce(const std::vector<int64_t>& shape) {
  return std::accumulate(shape.cbegin(), shape.cend(), (int64_t)1,
                         std::multiplies<int64_t>{});
}
template <typename T>
static std::vector<std::vector<T>> fold2D(gsl::span<const T> ws,
                                          const std::vector<int64_t>& shape) {
  CHECK(ws.size() == (size_t)reduce(shape))
      << ws.size() << "!=" << (size_t)reduce(shape);
  CHECK(shape.size() == 2);
  int32_t rows = (int32_t)shape[0];
  int32_t cols = (int32_t)shape[1];
  std::vector<std::vector<T>> ret(rows);
  for (int i = 0; i < rows; ++i)
    ret[i].resize(cols);

  for (size_t i = 0; i < ws.size(); ++i) {
    int r = (int)i / cols;
    int c = (int)i % cols;
    ret[r][c] = ws[i];
  }
  return ret;
}

template <typename T>
static std::vector<std::vector<std::vector<T>>>
fold3D(gsl::span<const T> ws, const std::vector<int64_t>& shape) {
  CHECK(ws.size() == (size_t)reduce(shape))
      << ws.size() << "!=" << (size_t)reduce(shape);
  CHECK(shape.size() == 3);
  int32_t batches = (int32_t)shape[0];
  int32_t rows = (int32_t)shape[1];
  int32_t cols = (int32_t)shape[2];
  std::vector<std::vector<std::vector<T>>> ret(batches);
  for (int n = 0; n < batches; ++n) {
    ret[n].resize(rows);
    for (int m = 0; m < rows; ++m)
      ret[n][m].resize(cols);
  }

  for (size_t i = 0; i < ws.size(); ++i) {
    int b = (int)i / (cols * rows);
    int r = (int)(i - b * cols * rows) / cols;
    int c = (int)i % cols;
    ret[b][r][c] = ws[i];
  }
  return ret;
}

template <typename T>
static std::vector<T> fold1D(gsl::span<const T> ws,
                             const std::vector<int64_t>& shape,
                             bool check1d = true) {
  CHECK(ws.size() == (size_t)reduce(shape))
      << ws.size() << "!=" << (size_t)reduce(shape);
  if (check1d)
    CHECK(shape.size() == 1);
  int32_t rows = (int32_t)reduce(shape);
  std::vector<T> ret(rows);
  for (int i = 0; i < rows; ++i)
    ret[i] = ws[i];
  return ret;
}

[[maybe_unused]] static std::string nodearg_dtype_to_string(const NodeArg& a) {
  auto out_dtype = node_arg_get_element_type(a);
  if (out_dtype == (int)ONNX_NAMESPACE::TensorProto_DataType_UINT8)
    return std::string("uint8");
  else if (out_dtype == (int)ONNX_NAMESPACE::TensorProto_DataType_UINT16)
    return std::string("uint16");
  else if (out_dtype == (int)ONNX_NAMESPACE::TensorProto_DataType_UINT32)
    return std::string("uint32");
  else if (out_dtype == (int)ONNX_NAMESPACE::TensorProto_DataType_UINT64)
    return std::string("uint64");
  else if (out_dtype == (int)ONNX_NAMESPACE::TensorProto_DataType_FLOAT)
    return std::string("float");
  else if (out_dtype == (int)ONNX_NAMESPACE::TensorProto_DataType_INT8)
    return std::string("int8");
  else if (out_dtype == (int)ONNX_NAMESPACE::TensorProto_DataType_INT16)
    return std::string("int16");
  else if (out_dtype == (int)ONNX_NAMESPACE::TensorProto_DataType_INT32)
    return std::string("int32");
  else if (out_dtype == (int)ONNX_NAMESPACE::TensorProto_DataType_INT64)
    return std::string("int64");
  else if (out_dtype == (int)ONNX_NAMESPACE::TensorProto_DataType_BOOL)
    return std::string("bool");
  else if (out_dtype == (int)ONNX_NAMESPACE::TensorProto_DataType_STRING)
    return std::string("string");
  else if (out_dtype == (int)ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16)
    return std::string("bfloat16");
  return std::string("unknown");
}

} // namespace vaip::dd