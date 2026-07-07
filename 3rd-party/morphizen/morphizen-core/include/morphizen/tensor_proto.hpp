/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

/**
 * @file tensor_proto.hpp
 * @brief This file contains functions declarations related to tensor_proto
 * creation and manipulation.
 */

#pragma once
#include "morphizen/_sanity_check.hpp"
#include <filesystem>
#include <morphizen/morphizen_gsl.h>
#include <morphizen/my_ort.h>
#include <sstream>
namespace morphizen {

MORPHIZEN_DLL_SPEC std::vector<int64_t>
tensor_proto_get_shape(const TensorProto &tensor_proto);

/**
 * @brief Creates a new TensorProto with float data type.
 * @param name The name of the tensor.
 * @param shape The shape of the tensor.
 * @param data The float data of the tensor.
 * @return A pointer to the created TensorProto.
 */
MORPHIZEN_DLL_SPEC TensorProtoPtr tensor_proto_new_floats(
    const std::string &name, const std::vector<int64_t> &shape,
    const std::vector<float> &data);

/** @brief tensor_proto_new_f32 is an alias of of tensor_proto_new_floats
 */
inline TensorProtoPtr tensor_proto_new_f32(const std::string &name,
                                           const std::vector<int64_t> &shape,
                                           const std::vector<float> &data) {
  return tensor_proto_new_floats(name, shape, data);
}

#if MORPHIZEN_ORT_API_MAJOR >= 3

/**
 * @brief Creates a new TensorProto with double data type.
 * @param name The name of the tensor.
 * @param shape The shape of the tensor.
 * @param data The double data of the tensor.
 * @return A pointer to the created TensorProto.
 */
MORPHIZEN_DLL_SPEC TensorProtoPtr tensor_proto_new_doubles(
    const std::string &name, const std::vector<int64_t> &shape,
    const std::vector<double> &data);
/** @brief tensor_proto_new_f64 is an alias of of tensor_proto_new_double
 */
inline TensorProtoPtr tensor_proto_new_f64(const std::string &name,
                                           const std::vector<int64_t> &shape,
                                           const std::vector<double> &data) {
  return tensor_proto_new_doubles(name, shape, data);
}
/**
 * @brief Creates a new TensorProto with bfloat16 data type.
 * @param name The name of the tensor.
 * @param shape The shape of the tensor.
 * @param data The bfloat16 data of the tensor.
 * @return A pointer to the created TensorProto.
 */
MORPHIZEN_DLL_SPEC TensorProtoPtr tensor_proto_new_bf16(
    const std::string &name, const std::vector<int64_t> &shape,
    const std::vector<int16_t> &data);

/**
 * @brief Creates a new TensorProto with float16 data type.
 * @param name The name of the tensor.
 * @param shape The shape of the tensor.
 * @param data The float16 data of the tensor.
 * @return A pointer to the created TensorProto.
 */
MORPHIZEN_DLL_SPEC TensorProtoPtr tensor_proto_new_fp16(
    const std::string &name, const std::vector<int64_t> &shape,
    const std::vector<int16_t> &data);
#endif

#if MORPHIZEN_ORT_API_MAJOR >= 13
/**
 * @brief Creates a new TensorProto with int4 data type.
 * @param name The name of the tensor.
 * @param shape The shape of the tensor.
 * @param data The int4 data of the tensor.
 * @return A pointer to the created TensorProto.
 */
MORPHIZEN_DLL_SPEC TensorProtoPtr
tensor_proto_new_i4(const std::string &name, const std::vector<int64_t> &shape,
                    const std::vector<int8_t> &data);

/**
 * @brief Creates a new TensorProto with uint4 data type.
 * @param name The name of the tensor.
 * @param shape The shape of the tensor.
 * @param data The uint4 data of the tensor.
 * @return A pointer to the created TensorProto.
 */
MORPHIZEN_DLL_SPEC TensorProtoPtr
tensor_proto_new_u4(const std::string &name, const std::vector<int64_t> &shape,
                    const std::vector<uint8_t> &data);

#endif
#if MORPHIZEN_ORT_API_MAJOR >= 19
/**
 * @brief Creates a new TensorProto with bool data type.
 * @param name The name of the tensor.
 * @param shape The shape of the tensor.
 * @param data The bool data of the tensor.
 * @return A pointer to the created TensorProto.
 */
MORPHIZEN_DLL_SPEC TensorProtoPtr tensor_proto_new_bool(
    const std::string &name, const std::vector<int64_t> &shape,
    const std::vector<uint8_t> &data);
#endif

/**
 * @brief Creates a new TensorProto with int32 data type.
 * @param name The name of the tensor.
 * @param shape The shape of the tensor.
 * @param data The int32 data of the tensor.
 * @return A pointer to the created TensorProto.
 */
MORPHIZEN_DLL_SPEC TensorProtoPtr
tensor_proto_new_i32(const std::string &name, const std::vector<int64_t> &shape,
                     const std::vector<int32_t> &data);

/**
 * @brief Creates a new TensorProto with int64 data type.
 * @param name The name of the tensor.
 * @param shape The shape of the tensor.
 * @param data The int64 data of the tensor.
 * @return A pointer to the created TensorProto.
 */
MORPHIZEN_DLL_SPEC TensorProtoPtr
tensor_proto_new_i64(const std::string &name, const std::vector<int64_t> &shape,
                     const std::vector<int64_t> &data);

/**
 * @brief Creates a new TensorProto with int8 data type.
 * @param name The name of the tensor.
 * @param shape The shape of the tensor.
 * @param data The int8 data of the tensor.
 * @return A pointer to the created TensorProto.
 */
MORPHIZEN_DLL_SPEC TensorProtoPtr
tensor_proto_new_i8(const std::string &name, const std::vector<int64_t> &shape,
                    const std::vector<int8_t> &data);

#if MORPHIZEN_ORT_API_MAJOR >= 3

/**
 * @brief Creates a new TensorProto with int16 data type.
 * @param name The name of the tensor.
 * @param shape The shape of the tensor.
 * @param data The int16 data of the tensor.
 * @return A pointer to the created TensorProto.
 */
MORPHIZEN_DLL_SPEC TensorProtoPtr
tensor_proto_new_i16(const std::string &name, const std::vector<int64_t> &shape,
                     const std::vector<int16_t> &data);

/**
 * @brief Creates a new TensorProto with uint8 data type.
 * @param name The name of the tensor.
 * @param shape The shape of the tensor.
 * @param data The uint8 data of the tensor.
 * @return A pointer to the created TensorProto.
 */
MORPHIZEN_DLL_SPEC TensorProtoPtr
tensor_proto_new_u8(const std::string &name, const std::vector<int64_t> &shape,
                    const std::vector<uint8_t> &data);

/**
 * @brief Creates a new TensorProto with uint16 data type.
 * @param name The name of the tensor.
 * @param shape The shape of the tensor.
 * @param data The uint16 data of the tensor.
 * @return A pointer to the created TensorProto.
 */
MORPHIZEN_DLL_SPEC TensorProtoPtr
tensor_proto_new_u16(const std::string &name, const std::vector<int64_t> &shape,
                     const std::vector<uint16_t> &data);

/**
 * @brief Creates a new TensorProto with uint32 data type.
 * @param name The name of the tensor.
 * @param shape The shape of the tensor.
 * @param data The uint32 data of the tensor.
 * @return A pointer to the created TensorProto.
 */
MORPHIZEN_DLL_SPEC TensorProtoPtr
tensor_proto_new_u32(const std::string &name, const std::vector<int64_t> &shape,
                     const std::vector<uint32_t> &data);

/**
 * @brief Creates a new TensorProto with uint64 data type.
 * @param name The name of the tensor.
 * @param shape The shape of the tensor.
 * @param data The uint64 data of the tensor.
 * @return A pointer to the created TensorProto.
 */
MORPHIZEN_DLL_SPEC TensorProtoPtr
tensor_proto_new_u64(const std::string &name, const std::vector<int64_t> &shape,
                     const std::vector<uint64_t> &data);
#endif

/**
 * @brief Extracts to a single float value from a TensorProto.
 * @param tensor The TensorProto.
 * @param graph The Graph.
 * @return The float scalar value extracted from the TensorProto.
 */
MORPHIZEN_DLL_SPEC float tensor_proto_as_float(const onnxruntime::Graph &graph,
                                               const TensorProto &tensor);

/**
 * @brief Extracts to a single double value from a TensorProto.
 * @param tensor The TensorProto.
 * @param graph The Graph.
 * @return The double scalar value extracted from the TensorProto.
 */
MORPHIZEN_DLL_SPEC double
tensor_proto_as_double(const onnxruntime::Graph &graph,
                       const TensorProto &tensor);

/**
 * @brief Extracts to a single bfloat16 value from a TensorProto.
 * @param tensor The TensorProto.
 * @param graph The Graph.
 * @return The bfloat16 scalar value extracted from the TensorProto.
 */
MORPHIZEN_DLL_SPEC int16_t tensor_proto_as_bf16(const onnxruntime::Graph &graph,
                                                const TensorProto &tensor);

/**
 * @brief Extracts to a single float16 value from a TensorProto.
 * @param tensor The TensorProto.
 * @param graph The Graph.
 * @return The float16 scalar value extracted from the TensorProto.
 */
MORPHIZEN_DLL_SPEC int16_t tensor_proto_as_fp16(const onnxruntime::Graph &graph,
                                                const TensorProto &tensor);

/**
 * @brief Extracts to a single int8 value from a TensorProto.
 * @param tensor The TensorProto.
 * @param graph The Graph.
 * @return The int8 scalar value extracted from the TensorProto.
 */
MORPHIZEN_DLL_SPEC int8_t tensor_proto_as_i8(const onnxruntime::Graph &graph,
                                             const TensorProto &tensor);

/**
 * @brief Extracts to a single uint8 value from a TensorProto.
 * @param tensor The TensorProto.
 * @param graph The Graph.
 * @return The uint8 scalar value extracted from the TensorProto.
 */
MORPHIZEN_DLL_SPEC uint8_t tensor_proto_as_u8(const onnxruntime::Graph &graph,
                                              const TensorProto &tensor);

/**
 * @brief Extracts to a single int16 value from a TensorProto.
 * @param tensor The TensorProto.
 * @param graph The Graph.
 * @return The int16 scalar value extracted from the TensorProto.
 */
MORPHIZEN_DLL_SPEC int16_t tensor_proto_as_i16(const onnxruntime::Graph &graph,
                                               const TensorProto &tensor);

/**
 * @brief Extracts to a single uint16 value from a TensorProto.
 * @param tensor The TensorProto.
 * @param graph The Graph.
 * @return The uint16 scalar value extracted from the TensorProto.
 */
MORPHIZEN_DLL_SPEC uint16_t tensor_proto_as_u16(const onnxruntime::Graph &graph,
                                                const TensorProto &tensor);

/**
 * @brief Extracts to a single int32 value from a TensorProto.
 * @param tensor The TensorProto.
 * @param graph The Graph.
 * @return The int32 scalar value extracted from the TensorProto.
 */
MORPHIZEN_DLL_SPEC int32_t tensor_proto_as_i32(const onnxruntime::Graph &graph,
                                               const TensorProto &tensor);

/**
 * @brief Extracts to a single uint32 value from a TensorProto.
 * @param tensor The TensorProto.
 * @param graph The Graph.
 * @return The uint32 scalar value extracted from the TensorProto.
 */
MORPHIZEN_DLL_SPEC uint32_t tensor_proto_as_u32(const onnxruntime::Graph &graph,
                                                const TensorProto &tensor);

/**
 * @brief Extracts to a single int64 value from a TensorProto.
 * @param tensor The TensorProto.
 * @param graph The Graph.
 * @return The int64 scalar value extracted from the TensorProto.
 */
MORPHIZEN_DLL_SPEC int64_t tensor_proto_as_i64(const onnxruntime::Graph &graph,
                                               const TensorProto &tensor);

/**
 * @brief Extracts to a single uint64 value from a TensorProto.
 * @param tensor The TensorProto.
 * @param graph The Graph.
 * @return The uint64 scalar value extracted from the TensorProto.
 */
MORPHIZEN_DLL_SPEC uint64_t tensor_proto_as_u64(const onnxruntime::Graph &graph,
                                                const TensorProto &tensor);

/**
 * @brief Extracts to int4 values from a TensorProto.
 * @param tensor The TensorProto.
 * @param graph The Graph.
 * @return The int4 values extracted from the TensorProto.
 */

MORPHIZEN_DLL_SPEC
int8_t get_int4_value(gsl::span<const int8_t> data, size_t idx);

MORPHIZEN_DLL_SPEC
uint8_t get_uint4_value(gsl::span<const uint8_t> data, size_t idx);

MORPHIZEN_DLL_SPEC
gsl::span<const int8_t> tensor_proto_as_i4s(const onnxruntime::Graph &graph,
                                            const TensorProto &tensor);

/**
 * @brief Extracts to uint4 values from a TensorProto.
 * @param tensor The TensorProto.
 * @param graph The Graph.
 * @return The uint4 values extracted from the TensorProto.
 */
MORPHIZEN_DLL_SPEC
gsl::span<const uint8_t> tensor_proto_as_u4s(const onnxruntime::Graph &graph,
                                             const TensorProto &tensor);

/**
 * @brief Extracts to int8 values from a TensorProto.
 * @param tensor The TensorProto.
 * @param graph The Graph.
 * @return The int8 values extracted from the TensorProto.
 */
MORPHIZEN_DLL_SPEC
gsl::span<const int8_t> tensor_proto_as_i8s(const onnxruntime::Graph &graph,
                                            const TensorProto &tensor);

/**
 * @brief Extracts to uint8 values from a TensorProto.
 * @param tensor The TensorProto.
 * @param graph The Graph.
 * @return The uint8 values extracted from the TensorProto.
 */
MORPHIZEN_DLL_SPEC
gsl::span<const uint8_t> tensor_proto_as_u8s(const onnxruntime::Graph &graph,
                                             const TensorProto &tensor);

/**
 * @brief Extracts to uint16 values from a TensorProto.
 * @param tensor The TensorProto.
 * @param graph The Graph.
 * @return The uint16 values extracted from the TensorProto.
 */
MORPHIZEN_DLL_SPEC
gsl::span<const uint16_t> tensor_proto_as_u16s(const onnxruntime::Graph &graph,
                                               const TensorProto &tensor);

/**
 * @brief Extracts to int16 values from a TensorProto.
 * @param tensor The TensorProto.
 * @param graph The Graph.
 * @return The int16 values extracted from the TensorProto.
 */
MORPHIZEN_DLL_SPEC
gsl::span<const int16_t> tensor_proto_as_i16s(const onnxruntime::Graph &graph,
                                              const TensorProto &tensor);

/**
 * @brief Extracts to uint32 values from a TensorProto.
 * @param tensor The TensorProto.
 * @param graph The Graph.
 * @return The uint32 values extracted from the TensorProto.
 */
MORPHIZEN_DLL_SPEC
gsl::span<const uint32_t> tensor_proto_as_u32s(const onnxruntime::Graph &graph,
                                               const TensorProto &tensor);

/**
 * @brief Extracts to int32 values from a TensorProto.
 * @param tensor The TensorProto.
 * @param graph The Graph.
 * @return The int32 values extracted from the TensorProto.
 */
MORPHIZEN_DLL_SPEC
gsl::span<const int32_t> tensor_proto_as_i32s(const onnxruntime::Graph &graph,
                                              const TensorProto &tensor);

/**
 * @brief Extracts to int64 values from a TensorProto.
 * @param tensor The TensorProto.
 * @param graph The Graph.
 * @return The int64 values extracted from the TensorProto.
 */
MORPHIZEN_DLL_SPEC
gsl::span<const int64_t> tensor_proto_as_i64s(const onnxruntime::Graph &graph,
                                              const TensorProto &tensor);

/**
 * @brief Extracts to uint64 values from a TensorProto.
 * @param tensor The TensorProto.
 * @param graph The Graph.
 * @return The uint64 values extracted from the TensorProto.
 */
MORPHIZEN_DLL_SPEC
gsl::span<const uint64_t> tensor_proto_as_u64s(const onnxruntime::Graph &graph,
                                               const TensorProto &tensor);

/**
 * @brief Extracts to float values from a TensorProto.
 * @param tensor The TensorProto.
 * @param graph The Graph.
 * @return The float values extracted from the TensorProto.
 */
MORPHIZEN_DLL_SPEC
gsl::span<const float> tensor_proto_as_floats(const onnxruntime::Graph &graph,
                                              const TensorProto &tensor);

/**
 * @brief Extracts to double values from a TensorProto.
 * @param tensor The TensorProto.
 * @param graph The Graph.
 * @return The double values extracted from the TensorProto.
 */
MORPHIZEN_DLL_SPEC
gsl::span<const double> tensor_proto_as_doubles(const onnxruntime::Graph &graph,
                                                const TensorProto &tensor);

/**
 * @brief Extracts to bloat16 values from a TensorProto.
 * @param tensor The TensorProto.
 * @param graph The Graph.
 * @return The bfloat16 values extracted from the TensorProto.
 */
MORPHIZEN_DLL_SPEC
gsl::span<const int16_t> tensor_proto_as_bf16s(const onnxruntime::Graph &graph,
                                               const TensorProto &tensor);

/**
 * @brief Extracts to float16 values from a TensorProto.
 * @param tensor The TensorProto.
 * @param graph The Graph.
 * @return The float16 values extracted from the TensorProto.
 */
MORPHIZEN_DLL_SPEC
gsl::span<const int16_t> tensor_proto_as_fp16s(const onnxruntime::Graph &graph,
                                               const TensorProto &tensor);

MORPHIZEN_DLL_SPEC
gsl::span<const char> tensor_proto_as_raw(const onnxruntime::Graph &graph,
                                          const TensorProto &tensor_proto);
} // namespace morphizen
