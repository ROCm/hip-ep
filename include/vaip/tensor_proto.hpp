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

/**
 * @file tensor_proto.hpp
 * @brief This file contains functions declarations related to tensor_proto
 * creation and manipulation.
 */

#pragma once
#include "./_sanity_check.hpp"
#include <filesystem>
#include <sstream>
#include <vaip/my_ort.h>
#include <vaip/vaip_gsl.h>
namespace vaip_core {

/**
 * @brief Creates a new TensorProto with float data type.
 * @param name The name of the tensor.
 * @param shape The shape of the tensor.
 * @param data The float data of the tensor.
 * @return A pointer to the created TensorProto.
 */
VAIP_DLL_SPEC TensorProtoPtr tensor_proto_new_floats(
    const std::string& name, const std::vector<int64_t>& shape,
    const std::vector<float>& data);

/** @brief tensor_proto_new_f32 is an alias of of tensor_proto_new_floats
 */
inline TensorProtoPtr tensor_proto_new_f32(const std::string& name,
                                           const std::vector<int64_t>& shape,
                                           const std::vector<float>& data) {
  return tensor_proto_new_floats(name, shape, data);
}

#if VAIP_ORT_API_MAJOR >= 3

/**
 * @brief Creates a new TensorProto with double data type.
 * @param name The name of the tensor.
 * @param shape The shape of the tensor.
 * @param data The double data of the tensor.
 * @return A pointer to the created TensorProto.
 */
VAIP_DLL_SPEC TensorProtoPtr tensor_proto_new_doubles(
    const std::string& name, const std::vector<int64_t>& shape,
    const std::vector<double>& data);
/** @brief tensor_proto_new_f64 is an alias of of tensor_proto_new_double
 */
inline TensorProtoPtr tensor_proto_new_f64(const std::string& name,
                                           const std::vector<int64_t>& shape,
                                           const std::vector<double>& data) {
  return tensor_proto_new_doubles(name, shape, data);
}
/**
 * @brief Creates a new TensorProto with bfloat16 data type.
 * @param name The name of the tensor.
 * @param shape The shape of the tensor.
 * @param data The bfloat16 data of the tensor.
 * @return A pointer to the created TensorProto.
 */
VAIP_DLL_SPEC TensorProtoPtr tensor_proto_new_bf16(
    const std::string& name, const std::vector<int64_t>& shape,
    const std::vector<int16_t>& data);

/**
 * @brief Creates a new TensorProto with float16 data type.
 * @param name The name of the tensor.
 * @param shape The shape of the tensor.
 * @param data The float16 data of the tensor.
 * @return A pointer to the created TensorProto.
 */
VAIP_DLL_SPEC TensorProtoPtr tensor_proto_new_fp16(
    const std::string& name, const std::vector<int64_t>& shape,
    const std::vector<int16_t>& data);
#endif

#if VAIP_ORT_API_MAJOR >= 13
/**
 * @brief Creates a new TensorProto with int4 data type.
 * @param name The name of the tensor.
 * @param shape The shape of the tensor.
 * @param data The int4 data of the tensor.
 * @return A pointer to the created TensorProto.
 */
VAIP_DLL_SPEC TensorProtoPtr
tensor_proto_new_i4(const std::string& name, const std::vector<int64_t>& shape,
                    const std::vector<int8_t>& data);

/**
 * @brief Creates a new TensorProto with uint4 data type.
 * @param name The name of the tensor.
 * @param shape The shape of the tensor.
 * @param data The uint4 data of the tensor.
 * @return A pointer to the created TensorProto.
 */
VAIP_DLL_SPEC TensorProtoPtr
tensor_proto_new_u4(const std::string& name, const std::vector<int64_t>& shape,
                    const std::vector<uint8_t>& data);

#endif
/**
 * @brief Creates a new TensorProto with int32 data type.
 * @param name The name of the tensor.
 * @param shape The shape of the tensor.
 * @param data The int32 data of the tensor.
 * @return A pointer to the created TensorProto.
 */
VAIP_DLL_SPEC TensorProtoPtr
tensor_proto_new_i32(const std::string& name, const std::vector<int64_t>& shape,
                     const std::vector<int32_t>& data);

/**
 * @brief Creates a new TensorProto with int64 data type.
 * @param name The name of the tensor.
 * @param shape The shape of the tensor.
 * @param data The int64 data of the tensor.
 * @return A pointer to the created TensorProto.
 */
VAIP_DLL_SPEC TensorProtoPtr
tensor_proto_new_i64(const std::string& name, const std::vector<int64_t>& shape,
                     const std::vector<int64_t>& data);

/**
 * @brief Creates a new TensorProto with int8 data type.
 * @param name The name of the tensor.
 * @param shape The shape of the tensor.
 * @param data The int8 data of the tensor.
 * @return A pointer to the created TensorProto.
 */
VAIP_DLL_SPEC TensorProtoPtr
tensor_proto_new_i8(const std::string& name, const std::vector<int64_t>& shape,
                    const std::vector<int8_t>& data);

#if VAIP_ORT_API_MAJOR >= 3

/**
 * @brief Creates a new TensorProto with int16 data type.
 * @param name The name of the tensor.
 * @param shape The shape of the tensor.
 * @param data The int16 data of the tensor.
 * @return A pointer to the created TensorProto.
 */
VAIP_DLL_SPEC TensorProtoPtr
tensor_proto_new_i16(const std::string& name, const std::vector<int64_t>& shape,
                     const std::vector<int16_t>& data);

/**
 * @brief Creates a new TensorProto with uint8 data type.
 * @param name The name of the tensor.
 * @param shape The shape of the tensor.
 * @param data The uint8 data of the tensor.
 * @return A pointer to the created TensorProto.
 */
VAIP_DLL_SPEC TensorProtoPtr
tensor_proto_new_u8(const std::string& name, const std::vector<int64_t>& shape,
                    const std::vector<uint8_t>& data);

/**
 * @brief Creates a new TensorProto with uint16 data type.
 * @param name The name of the tensor.
 * @param shape The shape of the tensor.
 * @param data The uint16 data of the tensor.
 * @return A pointer to the created TensorProto.
 */
VAIP_DLL_SPEC TensorProtoPtr
tensor_proto_new_u16(const std::string& name, const std::vector<int64_t>& shape,
                     const std::vector<uint16_t>& data);

/**
 * @brief Creates a new TensorProto with uint32 data type.
 * @param name The name of the tensor.
 * @param shape The shape of the tensor.
 * @param data The uint32 data of the tensor.
 * @return A pointer to the created TensorProto.
 */
VAIP_DLL_SPEC TensorProtoPtr
tensor_proto_new_u32(const std::string& name, const std::vector<int64_t>& shape,
                     const std::vector<uint32_t>& data);

/**
 * @brief Creates a new TensorProto with uint64 data type.
 * @param name The name of the tensor.
 * @param shape The shape of the tensor.
 * @param data The uint64 data of the tensor.
 * @return A pointer to the created TensorProto.
 */
VAIP_DLL_SPEC TensorProtoPtr
tensor_proto_new_u64(const std::string& name, const std::vector<int64_t>& shape,
                     const std::vector<uint64_t>& data);
#endif

/**
 * @brief Extracts to a single float value from a TensorProto.
 * @param tensor The TensorProto.
 * @param graph The Graph.
 * @return The float scalar value extracted from the TensorProto.
 */
VAIP_DLL_SPEC float tensor_proto_as_float(const onnxruntime::Graph& graph,
                                          const TensorProto& tensor);

/**
 * @brief Extracts to a single double value from a TensorProto.
 * @param tensor The TensorProto.
 * @param graph The Graph.
 * @return The double scalar value extracted from the TensorProto.
 */
VAIP_DLL_SPEC double tensor_proto_as_double(const onnxruntime::Graph& graph,
                                            const TensorProto& tensor);

/**
 * @brief Extracts to a single bfloat16 value from a TensorProto.
 * @param tensor The TensorProto.
 * @param graph The Graph.
 * @return The bfloat16 scalar value extracted from the TensorProto.
 */
VAIP_DLL_SPEC int16_t tensor_proto_as_bf16(const onnxruntime::Graph& graph,
                                           const TensorProto& tensor);

/**
 * @brief Extracts to a single float16 value from a TensorProto.
 * @param tensor The TensorProto.
 * @param graph The Graph.
 * @return The float16 scalar value extracted from the TensorProto.
 */
VAIP_DLL_SPEC int16_t tensor_proto_as_fp16(const onnxruntime::Graph& graph,
                                           const TensorProto& tensor);

/**
 * @brief Extracts to a single int8 value from a TensorProto.
 * @param tensor The TensorProto.
 * @param graph The Graph.
 * @return The int8 scalar value extracted from the TensorProto.
 */
VAIP_DLL_SPEC int8_t tensor_proto_as_i8(const onnxruntime::Graph& graph,
                                        const TensorProto& tensor);

/**
 * @brief Extracts to a single uint8 value from a TensorProto.
 * @param tensor The TensorProto.
 * @param graph The Graph.
 * @return The uint8 scalar value extracted from the TensorProto.
 */
VAIP_DLL_SPEC uint8_t tensor_proto_as_u8(const onnxruntime::Graph& graph,
                                         const TensorProto& tensor);

/**
 * @brief Extracts to a single int16 value from a TensorProto.
 * @param tensor The TensorProto.
 * @param graph The Graph.
 * @return The int16 scalar value extracted from the TensorProto.
 */
VAIP_DLL_SPEC int16_t tensor_proto_as_i16(const onnxruntime::Graph& graph,
                                          const TensorProto& tensor);

/**
 * @brief Extracts to a single uint16 value from a TensorProto.
 * @param tensor The TensorProto.
 * @param graph The Graph.
 * @return The uint16 scalar value extracted from the TensorProto.
 */
VAIP_DLL_SPEC uint16_t tensor_proto_as_u16(const onnxruntime::Graph& graph,
                                           const TensorProto& tensor);

/**
 * @brief Extracts to a single int32 value from a TensorProto.
 * @param tensor The TensorProto.
 * @param graph The Graph.
 * @return The int32 scalar value extracted from the TensorProto.
 */
VAIP_DLL_SPEC int32_t tensor_proto_as_i32(const onnxruntime::Graph& graph,
                                          const TensorProto& tensor);

/**
 * @brief Extracts to a single uint32 value from a TensorProto.
 * @param tensor The TensorProto.
 * @param graph The Graph.
 * @return The uint32 scalar value extracted from the TensorProto.
 */
VAIP_DLL_SPEC uint32_t tensor_proto_as_u32(const onnxruntime::Graph& graph,
                                           const TensorProto& tensor);

/**
 * @brief Extracts to a single int64 value from a TensorProto.
 * @param tensor The TensorProto.
 * @param graph The Graph.
 * @return The int64 scalar value extracted from the TensorProto.
 */
VAIP_DLL_SPEC int64_t tensor_proto_as_i64(const onnxruntime::Graph& graph,
                                          const TensorProto& tensor);

/**
 * @brief Extracts to a single uint64 value from a TensorProto.
 * @param tensor The TensorProto.
 * @param graph The Graph.
 * @return The uint64 scalar value extracted from the TensorProto.
 */
VAIP_DLL_SPEC uint64_t tensor_proto_as_u64(const onnxruntime::Graph& graph,
                                           const TensorProto& tensor);

/**
 * @brief Extracts to int4 values from a TensorProto.
 * @param tensor The TensorProto.
 * @param graph The Graph.
 * @return The int4 values extracted from the TensorProto.
 */

VAIP_DLL_SPEC
int8_t get_int4_value(gsl::span<const int8_t> data, size_t idx);

VAIP_DLL_SPEC
uint8_t get_uint4_value(gsl::span<const uint8_t> data, size_t idx);

VAIP_DLL_SPEC
gsl::span<const int8_t> tensor_proto_as_i4s(const onnxruntime::Graph& graph,
                                            const TensorProto& tensor);

/**
 * @brief Extracts to uint4 values from a TensorProto.
 * @param tensor The TensorProto.
 * @param graph The Graph.
 * @return The uint4 values extracted from the TensorProto.
 */
VAIP_DLL_SPEC
gsl::span<const uint8_t> tensor_proto_as_u4s(const onnxruntime::Graph& graph,
                                             const TensorProto& tensor);

/**
 * @brief Extracts to int8 values from a TensorProto.
 * @param tensor The TensorProto.
 * @param graph The Graph.
 * @return The int8 values extracted from the TensorProto.
 */
VAIP_DLL_SPEC
gsl::span<const int8_t> tensor_proto_as_i8s(const onnxruntime::Graph& graph,
                                            const TensorProto& tensor);

/**
 * @brief Extracts to uint8 values from a TensorProto.
 * @param tensor The TensorProto.
 * @param graph The Graph.
 * @return The uint8 values extracted from the TensorProto.
 */
VAIP_DLL_SPEC
gsl::span<const uint8_t> tensor_proto_as_u8s(const onnxruntime::Graph& graph,
                                             const TensorProto& tensor);

/**
 * @brief Extracts to uint16 values from a TensorProto.
 * @param tensor The TensorProto.
 * @param graph The Graph.
 * @return The uint16 values extracted from the TensorProto.
 */
VAIP_DLL_SPEC
gsl::span<const uint16_t> tensor_proto_as_u16s(const onnxruntime::Graph& graph,
                                               const TensorProto& tensor);

/**
 * @brief Extracts to int16 values from a TensorProto.
 * @param tensor The TensorProto.
 * @param graph The Graph.
 * @return The int16 values extracted from the TensorProto.
 */
VAIP_DLL_SPEC
gsl::span<const int16_t> tensor_proto_as_i16s(const onnxruntime::Graph& graph,
                                              const TensorProto& tensor);

/**
 * @brief Extracts to uint32 values from a TensorProto.
 * @param tensor The TensorProto.
 * @param graph The Graph.
 * @return The uint32 values extracted from the TensorProto.
 */
VAIP_DLL_SPEC
gsl::span<const uint32_t> tensor_proto_as_u32s(const onnxruntime::Graph& graph,
                                               const TensorProto& tensor);

/**
 * @brief Extracts to int32 values from a TensorProto.
 * @param tensor The TensorProto.
 * @param graph The Graph.
 * @return The int32 values extracted from the TensorProto.
 */
VAIP_DLL_SPEC
gsl::span<const int32_t> tensor_proto_as_i32s(const onnxruntime::Graph& graph,
                                              const TensorProto& tensor);

/**
 * @brief Extracts to int64 values from a TensorProto.
 * @param tensor The TensorProto.
 * @param graph The Graph.
 * @return The int64 values extracted from the TensorProto.
 */
VAIP_DLL_SPEC
gsl::span<const int64_t> tensor_proto_as_i64s(const onnxruntime::Graph& graph,
                                              const TensorProto& tensor);

/**
 * @brief Extracts to uint64 values from a TensorProto.
 * @param tensor The TensorProto.
 * @param graph The Graph.
 * @return The uint64 values extracted from the TensorProto.
 */
VAIP_DLL_SPEC
gsl::span<const uint64_t> tensor_proto_as_u64s(const onnxruntime::Graph& graph,
                                               const TensorProto& tensor);

/**
 * @brief Extracts to float values from a TensorProto.
 * @param tensor The TensorProto.
 * @param graph The Graph.
 * @return The float values extracted from the TensorProto.
 */
VAIP_DLL_SPEC
gsl::span<const float> tensor_proto_as_floats(const onnxruntime::Graph& graph,
                                              const TensorProto& tensor);

/**
 * @brief Extracts to double values from a TensorProto.
 * @param tensor The TensorProto.
 * @param graph The Graph.
 * @return The double values extracted from the TensorProto.
 */
VAIP_DLL_SPEC
gsl::span<const double> tensor_proto_as_doubles(const onnxruntime::Graph& graph,
                                                const TensorProto& tensor);

/**
 * @brief Extracts to bloat16 values from a TensorProto.
 * @param tensor The TensorProto.
 * @param graph The Graph.
 * @return The bfloat16 values extracted from the TensorProto.
 */
VAIP_DLL_SPEC
gsl::span<const int16_t> tensor_proto_as_bf16s(const onnxruntime::Graph& graph,
                                               const TensorProto& tensor);

/**
 * @brief Extracts to float16 values from a TensorProto.
 * @param tensor The TensorProto.
 * @param graph The Graph.
 * @return The float16 values extracted from the TensorProto.
 */
VAIP_DLL_SPEC
gsl::span<const int16_t> tensor_proto_as_fp16s(const onnxruntime::Graph& graph,
                                               const TensorProto& tensor);

VAIP_DLL_SPEC
gsl::span<const char> tensor_proto_as_raw(const onnxruntime::Graph& graph,
                                          const TensorProto& tensor_proto);
} // namespace vaip_core
