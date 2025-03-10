/*
 *     The Xilinx Vitis AI Vaip in this distribution are provided under the
 * following free and permissive binary-only license, but are not provided in
 * source code form.  While the following free and permissive license is similar
 * to the BSD open source license, it is NOT the BSD open source license nor
 * other OSI-approved open source license.
 *
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

#include "onnxruntime_api.hpp"

#include "glog/logging.h"
namespace vaip_core {

static ONNXTensorElementDataType
convert_elem_type(const std::string& data_type) {
  auto elemType = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  if (data_type == "float32") {
    elemType = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  } else if (data_type == "int8") {
    elemType = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
  } else if (data_type == "uint8") {
    elemType = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
  } else if (data_type == "int32") {
    elemType = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
  } else if (data_type == "uint32") {
    elemType = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32;
  } else if (data_type == "int64") {
    elemType = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  } else if (data_type == "uint64") {
    elemType = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64;
  } else if (data_type == "int1") {
    elemType = ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
  } else if (data_type == "bfloat16") {
    elemType = ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
  } else if (data_type == "float16") {
    elemType = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
  } else if (data_type == "uint16") {
    elemType = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16;
  } else if (data_type == "int16") {
    elemType = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16;
  } else if (data_type == "double") {
    elemType = ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
  } else if (data_type == "string") {
    elemType = ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
  } else if (data_type == "complex64") {
    elemType = ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64;
  } else if (data_type == "complex128") {
    elemType = ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128;
  } else if (data_type == "float8e4m3fn") {
    elemType = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN;
  } else if (data_type == "float8e4m3fnuz") {
    elemType = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FNUZ;
  } else if (data_type == "float8e5m2") {
    elemType = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2;
  } else if (data_type == "float8e5m2funz") {
    elemType = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2FNUZ;
  }
  return elemType;
}

static Ort::ShapeInferContext::Shape
convert_shape(const std::vector<int64_t>& shape) {
  auto ret = Ort::ShapeInferContext::Shape();
  ret.reserve(shape.size());
  for (auto i = 0u; i < shape.size(); i++) {
    ret.push_back(shape.at(i));
  }
  return ret;
}

template <typename TKernel>
struct XilinxCustomOpBase
    : public Ort::CustomOpBase<XilinxCustomOpBase<TKernel>, TKernel> {
  std::string name_;
  size_t num_output_;
  bool is_single_output_;

  XilinxCustomOpBase(const std::string& name, size_t num_output,
                     bool is_single_output)
      : name_(name), num_output_(num_output),
        is_single_output_(is_single_output) {

    OrtCustomOp::InferOutputShapeFn =
        [](const OrtCustomOp* op,
           OrtShapeInferContext* ort_ctx) -> OrtStatusPtr {
      Ort::ShapeInferContext ctx(&Ort::GetApi(), ort_ctx);
      auto custom_op = static_cast<const XilinxCustomOpBase*>(op);
      if (custom_op->name_ != "super_layer") {
        if (custom_op->is_single_output_) {
          try {
            auto data_type = ctx.GetAttrString("data_type");
            auto shape = ctx.GetAttrInts("shape");
            ctx.SetOutputShape(0, convert_shape(shape),
                               convert_elem_type(data_type));
          } catch (const std::exception& e) {
            LOG(FATAL) << "[VitisAI] custom op shape infer get shape and "
                          "data_type failed: "
                       << e.what();
          }
        } else {
          for (auto i = 0u; i < custom_op->num_output_; i++) {
            try {
              auto data_type_attr_name =
                  std::string("data_type_") + std::to_string(i);
              auto data_type = ctx.GetAttrString(data_type_attr_name.c_str());
              auto shape_attr_name = std::string("shape_") + std::to_string(i);
              auto shape = ctx.GetAttrInts(shape_attr_name.c_str());
              ctx.SetOutputShape(i, convert_shape(shape),
                                 convert_elem_type(data_type));
            } catch (const std::exception& e) {
              LOG(FATAL) << "[VitisAI] custom op shape infer get shape_"
                         << std::to_string(i)
                         << " and "
                            "data_type_"
                         << std::to_string(i) << " failed: " << e.what();
            }
          }
        }
      }
      return nullptr;
    };
  }

  void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const {
    return std::make_unique<TKernel>(api, info).release();
  };

  const char* GetName() const { return name_.c_str(); };

  const char* GetExecutionProviderType() const {
    return "VitisAIExecutionProvider";
  };

  size_t GetInputTypeCount() const { return 1u; }

  ONNXTensorElementDataType GetInputType(size_t index) const {
    // versions of the same operator define.
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  };

  size_t GetOutputTypeCount() const { return 1u; };

  ONNXTensorElementDataType GetOutputType(size_t index) const {
    // CHECK_EQ(index, 0u)
    //   << "'com.xilinx' domain's op not support multiple outputs ";
    // If 'type' is undefined, all types are allowed regardless of what other
    // versions of the same operator define.

    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  };

  int GetVariadicInputMinArity() const { return 0; }
  int GetVariadicOutputMinArity() const { return 0; }

  OrtCustomOpInputOutputCharacteristic
  GetInputCharacteristic(size_t /*index*/) const {
    // disable number of inputs checking , export_to_xir will checking
    return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_VARIADIC;
  }
  OrtCustomOpInputOutputCharacteristic
  GetOutputCharacteristic(size_t /*index*/) const {
    return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_VARIADIC;
  }
  bool GetVariadicOutputHomogeneity() const { return false; }
  bool GetVariadicInputHomogeneity() const { return false; }
};

struct XilinxCustomKernel {
  XilinxCustomKernel(const OrtApi& api, const OrtKernelInfo* info) {}
  void Compute(OrtKernelContext* context) {
    LOG(FATAL) << "TODO : com.xilinx op not implemention";
  }
};

struct XilinxCustomOp : XilinxCustomOpBase<XilinxCustomKernel> {
  XilinxCustomOp(const std::string& name, size_t num_output = 1u,
                 bool is_single_output = true)
      : XilinxCustomOpBase(name, num_output, is_single_output) {}
};

} // namespace vaip_core
