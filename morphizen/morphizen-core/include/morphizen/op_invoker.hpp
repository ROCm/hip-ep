/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once
#include "morphizen/onnxruntime_api.hpp"
#include <memory>

namespace morphizen {

class OpInvoker {
public:
  // Private tag pattern - prevents direct construction
  struct PrivateTag {
  private:
    friend OpInvoker;
    explicit PrivateTag() = default;
  };

  /**
   * @brief Create a default CPU Ort::MemoryInfo.
   */
  static Ort::MemoryInfo CreateDefaultCpuMemInfo();

  /**
   * @brief Create a `OpInvoker` object with deafult Ort::Env and default
   * Ort::SessionOptions.
   *
   * @param op_name Operator name (OpType in ONNX)
   * @param domain Operator domain
   * @param version Operator opset version
   * @param attr_values Attributes used to initialize the operator
   * @param attr_count Number of the attributes
   * @param input_type_values DataType of each input value
   * @param input_count Number of inputs
   * @param output_type_values DataType of each output value
   * @param output_count Number of outputs
   *
   * @return std::unique_ptr of an instance of OpInvoker.
   */
  static std::unique_ptr<OpInvoker>
  Create(const char *op_name, const char *domain, int version,
         const Ort::OpAttr *attr_values, size_t attr_count,
         const ONNXTensorElementDataType *input_type_values, size_t input_count,
         const ONNXTensorElementDataType *output_type_values,
         size_t output_count);

  /**
   * @brief Create a `OpInvoker` object.
   *
   * @param env Environment used to create Ort::Session
   * @param options Options used to create Ort::Session
   * @param op_name Operator name (OpType in ONNX)
   * @param domain Operator domain
   * @param version Operator opset version
   * @param attr_values Attributes used to initialize the operator
   * @param attr_count Number of the attributes
   * @param input_type_values DataType of each input value
   * @param input_count Number of inputs
   * @param output_type_values DataType of each output value
   * @param output_count Number of outputs
   *
   * @return std::unique_ptr of an instance of OpInvoker.
   */
  static std::unique_ptr<OpInvoker>
  Create(const Ort::Env &env, const Ort::SessionOptions &options,
         const char *op_name, const char *domain, int version,
         const Ort::OpAttr *attr_values, size_t attr_count,
         const ONNXTensorElementDataType *input_type_values, size_t input_count,
         const ONNXTensorElementDataType *output_type_values,
         size_t output_count);

  /**
   * @brief Invoke the OpInvoker created by OpInvoker::Create
   * The inputs and outputs must follow the order as specified in onnx
   * specification
   *
   * @param input_values Array of inputs
   * @param input_count Number of inputs
   * @param output_values Array of outputs
   * @param output_count Number of outputs
   */
  void Invoke(const Ort::Value *input_values, size_t input_count,
              Ort::Value *output_values, size_t output_count);

  /**
   * @brief Invoke the OpInvoker created by OpInvoker::Create
   * The inputs and outputs must follow the order as specified in onnx
   * specification
   *
   * @param run_options Options used to run Ort::Session
   * @param input_values Array of inputs
   * @param input_count Number of inputs
   * @param output_values Array of outputs. If output_values[i] is empty or has
   * no value, mem_info_arr[i] will be used (if provided, otherwise the default
   * CPU OrtMemoryInfo will be used) to create output value by Session.
   * @param output_count Number of outputs
   * @param mem_info_arr Array of "const OrtMemoryInfo*" for outputs or
   * nullptr. If array is provided, the length of array must equal to
   * output_count. If nullptr is provided, the return value of
   * CreateDefaultCpuMemInfo will be used.
   */
  void Invoke(const Ort::RunOptions &run_options,
              const Ort::Value *input_values, size_t input_count,
              Ort::Value *output_values, size_t output_count,
              const OrtMemoryInfo *const *mem_info_arr = nullptr);

  /**
   * @brief Constructor only accessible through Create() factory method
   */
  explicit OpInvoker(PrivateTag, const Ort::Env &env,
                     const Ort::SessionOptions &options, const char *op_name,
                     const char *domain, int version,
                     const Ort::OpAttr *attr_values, size_t attr_count,
                     const ONNXTensorElementDataType *input_type_values,
                     size_t input_count,
                     const ONNXTensorElementDataType *output_type_values,
                     size_t output_count);

  /**
   * @brief Destroys the `OpInvoker` object.
   */
  ~OpInvoker();

private:
  Ort::Session session_{nullptr};
};

} // namespace morphizen
