/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// High-level NodeArg const data extraction functions
// These functions depend on tensor_proto which is a morphizen-core component

#include "morphizen/node_arg.hpp"
#include "morphizen/tensor_proto.hpp"
#include <glog/logging.h>

namespace morphizen {

MORPHIZEN_DLL_SPEC int8_t
node_arg_get_const_data_as_i8(const Graph& graph, const NodeArg& node_arg) {
  CHECK(node_arg_exists(node_arg)) << "node_arg doesn't exist!";
  return tensor_proto_as_i8(graph,
                            node_arg_get_const_data_as_tensor(graph, node_arg));
}
MORPHIZEN_DLL_SPEC uint8_t
node_arg_get_const_data_as_u8(const Graph& graph, const NodeArg& node_arg) {
  CHECK(node_arg_exists(node_arg)) << "node_arg doesn't exist!";
  return tensor_proto_as_u8(graph,
                            node_arg_get_const_data_as_tensor(graph, node_arg));
}
MORPHIZEN_DLL_SPEC int16_t
node_arg_get_const_data_as_i16(const Graph& graph, const NodeArg& node_arg) {
  CHECK(node_arg_exists(node_arg)) << "node_arg doesn't exist!";
  return tensor_proto_as_i16(
      graph, node_arg_get_const_data_as_tensor(graph, node_arg));
}
MORPHIZEN_DLL_SPEC uint16_t
node_arg_get_const_data_as_u16(const Graph& graph, const NodeArg& node_arg) {
  CHECK(node_arg_exists(node_arg)) << "node_arg doesn't exist!";
  return tensor_proto_as_u16(
      graph, node_arg_get_const_data_as_tensor(graph, node_arg));
}
MORPHIZEN_DLL_SPEC int32_t
node_arg_get_const_data_as_i32(const Graph& graph, const NodeArg& node_arg) {
  CHECK(node_arg_exists(node_arg)) << "node_arg doesn't exist!";
  return tensor_proto_as_i32(
      graph, node_arg_get_const_data_as_tensor(graph, node_arg));
}
MORPHIZEN_DLL_SPEC uint32_t
node_arg_get_const_data_as_u32(const Graph& graph, const NodeArg& node_arg) {
  CHECK(node_arg_exists(node_arg)) << "node_arg doesn't exist!";
  return tensor_proto_as_u32(
      graph, node_arg_get_const_data_as_tensor(graph, node_arg));
}
MORPHIZEN_DLL_SPEC int64_t
node_arg_get_const_data_as_i64(const Graph& graph, const NodeArg& node_arg) {
  CHECK(node_arg_exists(node_arg)) << "node_arg doesn't exist!";
  return tensor_proto_as_i64(
      graph, node_arg_get_const_data_as_tensor(graph, node_arg));
}
MORPHIZEN_DLL_SPEC uint64_t
node_arg_get_const_data_as_u64(const Graph& graph, const NodeArg& node_arg) {
  CHECK(node_arg_exists(node_arg)) << "node_arg doesn't exist!";
  return tensor_proto_as_u64(
      graph, node_arg_get_const_data_as_tensor(graph, node_arg));
}
MORPHIZEN_DLL_SPEC float
node_arg_get_const_data_as_float(const Graph& graph, const NodeArg& node_arg) {
  CHECK(node_arg_exists(node_arg)) << "node_arg doesn't exist!";
  return tensor_proto_as_float(
      graph, node_arg_get_const_data_as_tensor(graph, node_arg));
}
MORPHIZEN_DLL_SPEC double
node_arg_get_const_data_as_double(const Graph& graph, const NodeArg& node_arg) {
  CHECK(node_arg_exists(node_arg)) << "node_arg doesn't exist!";
  return tensor_proto_as_double(
      graph, node_arg_get_const_data_as_tensor(graph, node_arg));
}
MORPHIZEN_DLL_SPEC int16_t
node_arg_get_const_data_as_bf16(const Graph& graph, const NodeArg& node_arg) {
  CHECK(node_arg_exists(node_arg)) << "node_arg doesn't exist!";
  return tensor_proto_as_bf16(
      graph, node_arg_get_const_data_as_tensor(graph, node_arg));
}

MORPHIZEN_DLL_SPEC int16_t
node_arg_get_const_data_as_fp16(const Graph& graph, const NodeArg& node_arg) {
  CHECK(node_arg_exists(node_arg)) << "node_arg doesn't exist!";
  return tensor_proto_as_fp16(
      graph, node_arg_get_const_data_as_tensor(graph, node_arg));
}

MORPHIZEN_DLL_SPEC gsl::span<const uint8_t>
node_arg_get_const_data_as_u4s(const Graph& graph, const NodeArg& node_arg) {
  CHECK(node_arg_exists(node_arg)) << "node_arg doesn't exist!";
  return tensor_proto_as_u4s(
      graph, node_arg_get_const_data_as_tensor(graph, node_arg));
}
MORPHIZEN_DLL_SPEC gsl::span<const int8_t>
node_arg_get_const_data_as_i4s(const Graph& graph, const NodeArg& node_arg) {
  CHECK(node_arg_exists(node_arg)) << "node_arg doesn't exist!";
  return tensor_proto_as_i4s(
      graph, node_arg_get_const_data_as_tensor(graph, node_arg));
}

MORPHIZEN_DLL_SPEC gsl::span<const uint8_t>
node_arg_get_const_data_as_u8s(const Graph& graph, const NodeArg& node_arg) {
  CHECK(node_arg_exists(node_arg)) << "node_arg doesn't exist!";
  return tensor_proto_as_u8s(
      graph, node_arg_get_const_data_as_tensor(graph, node_arg));
}
MORPHIZEN_DLL_SPEC gsl::span<const int8_t>
node_arg_get_const_data_as_i8s(const Graph& graph, const NodeArg& node_arg) {
  CHECK(node_arg_exists(node_arg)) << "node_arg doesn't exist!";
  return tensor_proto_as_i8s(
      graph, node_arg_get_const_data_as_tensor(graph, node_arg));
}
MORPHIZEN_DLL_SPEC gsl::span<const uint16_t>
node_arg_get_const_data_as_u16s(const Graph& graph, const NodeArg& node_arg) {
  CHECK(node_arg_exists(node_arg)) << "node_arg doesn't exist!";
  return tensor_proto_as_u16s(
      graph, node_arg_get_const_data_as_tensor(graph, node_arg));
}
MORPHIZEN_DLL_SPEC gsl::span<const int16_t>
node_arg_get_const_data_as_i16s(const Graph& graph, const NodeArg& node_arg) {
  CHECK(node_arg_exists(node_arg)) << "node_arg doesn't exist!";
  return tensor_proto_as_i16s(
      graph, node_arg_get_const_data_as_tensor(graph, node_arg));
}
MORPHIZEN_DLL_SPEC gsl::span<const int32_t>
node_arg_get_const_data_as_i32s(const Graph& graph, const NodeArg& node_arg) {
  CHECK(node_arg_exists(node_arg)) << "node_arg doesn't exist!";
  return tensor_proto_as_i32s(
      graph, node_arg_get_const_data_as_tensor(graph, node_arg));
}
MORPHIZEN_DLL_SPEC gsl::span<const uint32_t>
node_arg_get_const_data_as_u32s(const Graph& graph, const NodeArg& node_arg) {
  CHECK(node_arg_exists(node_arg)) << "node_arg doesn't exist!";
  return tensor_proto_as_u32s(
      graph, node_arg_get_const_data_as_tensor(graph, node_arg));
}
MORPHIZEN_DLL_SPEC gsl::span<const int64_t>
node_arg_get_const_data_as_i64s(const Graph& graph, const NodeArg& node_arg) {
  CHECK(node_arg_exists(node_arg)) << "node_arg doesn't exist!";
  return tensor_proto_as_i64s(
      graph, node_arg_get_const_data_as_tensor(graph, node_arg));
}
MORPHIZEN_DLL_SPEC gsl::span<const uint64_t>
node_arg_get_const_data_as_u64s(const Graph& graph, const NodeArg& node_arg) {
  CHECK(node_arg_exists(node_arg)) << "node_arg doesn't exist!";
  return tensor_proto_as_u64s(
      graph, node_arg_get_const_data_as_tensor(graph, node_arg));
}
MORPHIZEN_DLL_SPEC gsl::span<const float>
node_arg_get_const_data_as_floats(const Graph& graph, const NodeArg& node_arg) {
  CHECK(node_arg_exists(node_arg)) << "node_arg doesn't exist!";
  return tensor_proto_as_floats(
      graph, node_arg_get_const_data_as_tensor(graph, node_arg));
}
MORPHIZEN_DLL_SPEC gsl::span<const double>
node_arg_get_const_data_as_doubles(const Graph& graph,
                                   const NodeArg& node_arg) {
  CHECK(node_arg_exists(node_arg)) << "node_arg doesn't exist!";
  return tensor_proto_as_doubles(
      graph, node_arg_get_const_data_as_tensor(graph, node_arg));
}
MORPHIZEN_DLL_SPEC gsl::span<const int16_t>
node_arg_get_const_data_as_bf16s(const Graph& graph, const NodeArg& node_arg) {
  CHECK(node_arg_exists(node_arg)) << "node_arg doesn't exist!";
  return tensor_proto_as_bf16s(
      graph, node_arg_get_const_data_as_tensor(graph, node_arg));
}

MORPHIZEN_DLL_SPEC gsl::span<const int16_t>
node_arg_get_const_data_as_fp16s(const Graph& graph, const NodeArg& node_arg) {
  CHECK(node_arg_exists(node_arg)) << "node_arg doesn't exist!";
  return tensor_proto_as_fp16s(
      graph, node_arg_get_const_data_as_tensor(graph, node_arg));
}

} // namespace morphizen

// NodeArgConstRef member functions that depend on const data extraction
namespace morphizen_cxx {

int8_t NodeArgConstRef::const_data_as_i8() const {
  return morphizen::node_arg_get_const_data_as_i8(graph_, self_);
}

uint8_t NodeArgConstRef::const_data_as_u8() const {
  return morphizen::node_arg_get_const_data_as_u8(graph_, self_);
}

int16_t NodeArgConstRef::const_data_as_i16() const {
  return morphizen::node_arg_get_const_data_as_i16(graph_, self_);
}

uint16_t NodeArgConstRef::const_data_as_u16() const {
  return morphizen::node_arg_get_const_data_as_u16(graph_, self_);
}

int32_t NodeArgConstRef::const_data_as_i32() const {
  return morphizen::node_arg_get_const_data_as_i32(graph_, self_);
}

uint32_t NodeArgConstRef::const_data_as_u32() const {
  return morphizen::node_arg_get_const_data_as_u32(graph_, self_);
}

int64_t NodeArgConstRef::const_data_as_i64() const {
  return morphizen::node_arg_get_const_data_as_i64(graph_, self_);
}

uint64_t NodeArgConstRef::const_data_as_u64() const {
  return morphizen::node_arg_get_const_data_as_u64(graph_, self_);
}

float NodeArgConstRef::const_data_as_f32() const {
  return morphizen::node_arg_get_const_data_as_float(graph_, self_);
}

double NodeArgConstRef::const_data_as_f64() const {
  return morphizen::node_arg_get_const_data_as_double(graph_, self_);
}

bf16_t NodeArgConstRef::const_data_as_bf16() const {
  return morphizen::node_arg_get_const_data_as_bf16(graph_, self_);
}

fp16_t NodeArgConstRef::const_data_as_fp16() const {
  return morphizen::node_arg_get_const_data_as_fp16(graph_, self_);
}

gsl::span<const uint8_t> NodeArgConstRef::const_data_as_u8_span() const {
  return morphizen::node_arg_get_const_data_as_u8s(graph_, self_);
}

gsl::span<const int8_t> NodeArgConstRef::const_data_as_i8_span() const {
  return morphizen::node_arg_get_const_data_as_i8s(graph_, self_);
}

gsl::span<const uint16_t> NodeArgConstRef::const_data_as_u16_span() const {
  return morphizen::node_arg_get_const_data_as_u16s(graph_, self_);
}

gsl::span<const int16_t> NodeArgConstRef::const_data_as_i16_span() const {
  return morphizen::node_arg_get_const_data_as_i16s(graph_, self_);
}

gsl::span<const uint32_t> NodeArgConstRef::const_data_as_u32_span() const {
  return morphizen::node_arg_get_const_data_as_u32s(graph_, self_);
}

gsl::span<const int32_t> NodeArgConstRef::const_data_as_i32_span() const {
  return morphizen::node_arg_get_const_data_as_i32s(graph_, self_);
}

gsl::span<const uint64_t> NodeArgConstRef::const_data_as_u64_span() const {
  return morphizen::node_arg_get_const_data_as_u64s(graph_, self_);
}

gsl::span<const int64_t> NodeArgConstRef::const_data_as_i64_span() const {
  return morphizen::node_arg_get_const_data_as_i64s(graph_, self_);
}

gsl::span<const float> NodeArgConstRef::const_data_as_f32_span() const {
  return morphizen::node_arg_get_const_data_as_floats(graph_, self_);
}

gsl::span<const double> NodeArgConstRef::const_data_as_f64_span() const {
  return morphizen::node_arg_get_const_data_as_doubles(graph_, self_);
}

gsl::span<const bf16_t> NodeArgConstRef::const_data_as_bf16_span() const {
  return morphizen::node_arg_get_const_data_as_bf16s(graph_, self_);
}

gsl::span<const fp16_t> NodeArgConstRef::const_data_as_fp16_span() const {
  return morphizen::node_arg_get_const_data_as_fp16s(graph_, self_);
}

gsl::span<const char> NodeArgConstRef::const_data_as_raw() const {
  return morphizen::tensor_proto_as_raw(
      graph_, morphizen::node_arg_get_const_data_as_tensor(graph_, self_));
}

} // namespace morphizen_cxx
