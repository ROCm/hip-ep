/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "./test_environment.hpp"
#include "morphizen/op_invoker.hpp"
#include <gtest/gtest.h>

static bool HasValue(const Ort::Value &value) {
  return static_cast<const OrtValue *>(value) && value.HasValue();
}

struct ValueCreater {
  std::vector<std::vector<char>> data_buffers;

  Ort::MemoryInfo mem_info{
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};

public:
  template <typename T>
  Ort::Value add_tensor_value(const std::vector<int64_t> &shape) {
    size_t total_size = 1;
    for (auto d : shape)
      total_size *= d;

    data_buffers.emplace_back(total_size * sizeof(T));

    return Ort::Value::CreateTensor<T>(mem_info,
                                       (T *)data_buffers.back().data(),
                                       total_size, shape.data(), shape.size());
  }
};

TEST(OpInvokerTest, CreateAndInvoke) {
#ifdef MORPHIZEN_ENABLE_MLIR_BACKEND
  GTEST_SKIP()
      << "Test skipped: MLIR backend shape nullptr issue (see Issue #034)";
#else
  int64_t do_rotary = 1;
  int64_t kv_num_heads = 8;
  int64_t local_window_size = -1;
  int64_t num_heads = 32;
  int64_t rotary_interleaved = 0;
  float scale = 0.125f;

  int64_t max_seq_len = 131072;
  int64_t head_size = 64;

  int64_t qkv_seq_len = 1;
  int64_t past_seq_len = 1024;
  int64_t total_seq_len = 1025;

  std::unique_ptr<morphizen::OpInvoker> op_invoker;

  {
    SCOPED_TRACE("Create");

    // attributes
    auto attr_do_rotary =
        Ort::OpAttr("do_rotary", &do_rotary, 1, OrtOpAttrType::ORT_OP_ATTR_INT);
    auto attr_kv_num_heads = Ort::OpAttr("kv_num_heads", &kv_num_heads, 1,
                                         OrtOpAttrType::ORT_OP_ATTR_INT);
    auto attr_local_window_size =
        Ort::OpAttr("local_window_size", &local_window_size, 1,
                    OrtOpAttrType::ORT_OP_ATTR_INT);
    auto attr_num_heads =
        Ort::OpAttr("num_heads", &num_heads, 1, OrtOpAttrType::ORT_OP_ATTR_INT);
    auto attr_rotary_interleave =
        Ort::OpAttr("rotary_interleaved", &rotary_interleaved, 1,
                    OrtOpAttrType::ORT_OP_ATTR_INT);
    auto attr_scale =
        Ort::OpAttr("scale", &scale, 1, OrtOpAttrType::ORT_OP_ATTR_FLOAT);

    Ort::OpAttr gqa_attrs[6] = {
        std::move(attr_do_rotary), std::move(attr_kv_num_heads),
        std::move(attr_num_heads), std::move(attr_rotary_interleave),
        std::move(attr_scale),     std::move(attr_local_window_size)};

    // input type values
    ONNXTensorElementDataType input_type_values[9] = {
        ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, // query
        ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, // key
        ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, // value
        ONNXTensorElementDataType::
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, // past_key
        ONNXTensorElementDataType::
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, // past_value
        ONNXTensorElementDataType::
            ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, // seqlens_k
        ONNXTensorElementDataType::
            ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, // total_sequence_length
        ONNXTensorElementDataType::
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, // cos_cache
        ONNXTensorElementDataType::
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT // sin_cache
    };

    // output type values
    ONNXTensorElementDataType output_type_values[3] = {
        ONNXTensorElementDataType::
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, // output
        ONNXTensorElementDataType::
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, // present_key
        ONNXTensorElementDataType::
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT // present_value
    };

    op_invoker = morphizen::OpInvoker::Create(
        "GroupQueryAttention", "com.microsoft", 1, gqa_attrs, 6,
        input_type_values, 9, output_type_values, 3);

    ASSERT_NE(op_invoker, nullptr);
  }

  {
    SCOPED_TRACE("Invoke");

    ValueCreater vc;

    // input tensors
    std::vector<Ort::Value> input_tensor_values;

    // query
    input_tensor_values.emplace_back(
        vc.add_tensor_value<float>({1, qkv_seq_len, num_heads * head_size}));

    // key
    input_tensor_values.emplace_back(
        vc.add_tensor_value<float>({1, qkv_seq_len, kv_num_heads * head_size}));

    // value
    input_tensor_values.emplace_back(
        vc.add_tensor_value<float>({1, qkv_seq_len, kv_num_heads * head_size}));

    // past_key
    input_tensor_values.emplace_back(
        vc.add_tensor_value<float>({1, kv_num_heads, past_seq_len, head_size}));

    // past_value
    input_tensor_values.emplace_back(
        vc.add_tensor_value<float>({1, kv_num_heads, past_seq_len, head_size}));

    // seqlens_k
    input_tensor_values.emplace_back(vc.add_tensor_value<int32_t>({1}));
    input_tensor_values.back().GetTensorMutableData<int32_t>()[0] =
        past_seq_len;

    // total_sequence_length
    input_tensor_values.emplace_back(vc.add_tensor_value<int32_t>({}));
    input_tensor_values.back().GetTensorMutableData<int32_t>()[0] =
        total_seq_len;

    // cos_cache
    input_tensor_values.emplace_back(
        vc.add_tensor_value<float>({max_seq_len, head_size / 2}));

    // sin_cache
    input_tensor_values.emplace_back(
        vc.add_tensor_value<float>({max_seq_len, head_size / 2}));

    {
      SCOPED_TRACE("PreAllocatedByUser");

      // output tensors
      std::vector<Ort::Value> output_tensor_values;

      // output
      output_tensor_values.emplace_back(
          vc.add_tensor_value<float>({1, qkv_seq_len, num_heads * head_size}));

      // present_key
      output_tensor_values.emplace_back(vc.add_tensor_value<float>(
          {1, kv_num_heads, total_seq_len, head_size}));

      // present_value
      output_tensor_values.emplace_back(vc.add_tensor_value<float>(
          {1, kv_num_heads, total_seq_len, head_size}));

      ASSERT_TRUE(HasValue(output_tensor_values[0]) &&
                  output_tensor_values[0].IsTensor());
      ASSERT_TRUE(HasValue(output_tensor_values[1]) &&
                  output_tensor_values[1].IsTensor());
      ASSERT_TRUE(HasValue(output_tensor_values[2]) &&
                  output_tensor_values[2].IsTensor());

      ASSERT_NO_THROW(op_invoker->Invoke(
          input_tensor_values.data(), input_tensor_values.size(),
          output_tensor_values.data(), output_tensor_values.size()));
    }

    {
      SCOPED_TRACE("AllocatedBySession");

      Ort::MemoryInfo mem_info =
          Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

      // output tensors
      std::vector<Ort::Value> output_tensor_values;

      // output
      output_tensor_values.emplace_back();

      // present_key
      output_tensor_values.emplace_back();

      // present_value
      output_tensor_values.emplace_back();

      ASSERT_FALSE(HasValue(output_tensor_values[0]));
      ASSERT_FALSE(HasValue(output_tensor_values[1]));
      ASSERT_FALSE(HasValue(output_tensor_values[2]));

      ASSERT_NO_THROW(op_invoker->Invoke(
          input_tensor_values.data(), input_tensor_values.size(),
          output_tensor_values.data(), output_tensor_values.size()));

      ASSERT_TRUE(HasValue(output_tensor_values[0]) &&
                  output_tensor_values[0].IsTensor());
      ASSERT_TRUE(HasValue(output_tensor_values[1]) &&
                  output_tensor_values[1].IsTensor());
      ASSERT_TRUE(HasValue(output_tensor_values[2]) &&
                  output_tensor_values[2].IsTensor());

      {
        Ort::TensorTypeAndShapeInfo info =
            output_tensor_values[0].GetTensorTypeAndShapeInfo();
        ASSERT_EQ(
            info.GetElementType(),
            ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

        std::vector<int64_t> shape = info.GetShape();
        ASSERT_EQ(shape.size(), 3);
        ASSERT_EQ(shape[0], 1);
        ASSERT_EQ(shape[1], qkv_seq_len);
        ASSERT_EQ(shape[2], num_heads * head_size);
      }

      {
        Ort::TensorTypeAndShapeInfo info =
            output_tensor_values[1].GetTensorTypeAndShapeInfo();
        ASSERT_EQ(
            info.GetElementType(),
            ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

        std::vector<int64_t> shape = info.GetShape();
        ASSERT_EQ(shape.size(), 4);
        ASSERT_EQ(shape[0], 1);
        ASSERT_EQ(shape[1], kv_num_heads);
        ASSERT_EQ(shape[2], total_seq_len);
        ASSERT_EQ(shape[3], head_size);
      }

      {
        Ort::TensorTypeAndShapeInfo info =
            output_tensor_values[2].GetTensorTypeAndShapeInfo();
        ASSERT_EQ(
            info.GetElementType(),
            ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

        std::vector<int64_t> shape = info.GetShape();
        ASSERT_EQ(shape.size(), 4);
        ASSERT_EQ(shape[0], 1);
        ASSERT_EQ(shape[1], kv_num_heads);
        ASSERT_EQ(shape[2], total_seq_len);
        ASSERT_EQ(shape[3], head_size);
      }
    }

    {
      SCOPED_TRACE("MixedAllocation");

      Ort::MemoryInfo mem_info =
          Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

      // output tensors
      std::vector<Ort::Value> output_tensor_values;

      // output
      output_tensor_values.emplace_back(
          vc.add_tensor_value<float>({1, qkv_seq_len, num_heads * head_size}));

      // present_key
      output_tensor_values.emplace_back();

      // present_value
      output_tensor_values.emplace_back(vc.add_tensor_value<float>(
          {1, kv_num_heads, total_seq_len, head_size}));

      ASSERT_TRUE(HasValue(output_tensor_values[0]) &&
                  output_tensor_values[0].IsTensor());
      ASSERT_FALSE(HasValue(output_tensor_values[1]));
      ASSERT_TRUE(HasValue(output_tensor_values[2]) &&
                  output_tensor_values[2].IsTensor());

      ASSERT_NO_THROW(op_invoker->Invoke(
          input_tensor_values.data(), input_tensor_values.size(),
          output_tensor_values.data(), output_tensor_values.size()));

      ASSERT_TRUE(HasValue(output_tensor_values[1]) &&
                  output_tensor_values[1].IsTensor());

      {
        Ort::TensorTypeAndShapeInfo info =
            output_tensor_values[1].GetTensorTypeAndShapeInfo();
        ASSERT_EQ(
            info.GetElementType(),
            ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

        std::vector<int64_t> shape = info.GetShape();
        ASSERT_EQ(shape.size(), 4);
        ASSERT_EQ(shape[0], 1);
        ASSERT_EQ(shape[1], kv_num_heads);
        ASSERT_EQ(shape[2], total_seq_len);
        ASSERT_EQ(shape[3], head_size);
      }
    }
  }
#endif
}
