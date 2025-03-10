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

#include <glog/logging.h>

#include "vaip/node_attr.hpp"
#include <vaip/vaip_ort_api.h>

namespace vaip_core {
void AttributeProtoDeleter::operator()(AttributeProto* p) const {
  VAIP_ORT_API(attr_proto_delete)(p);
}
void NodeAttributesDeleter::operator()(NodeAttributes* p) const {
  VAIP_ORT_API(node_attributes_delete)(p);
}
std::string attr_proto_as_string(const AttributeProto& attr) { return "TODO"; }
std::string data_type_to_string(int elem_type) {
  auto ret = std::string();
  if (elem_type ==
      ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT8) {
    ret = "int8";
  } else if (elem_type ==
             ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT8) {
    ret = "uint8";
  } else if (elem_type ==
             ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT32) {
    ret = "int32";
  } else if (elem_type ==
             ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT64) {
    ret = "int64";
  } else if (elem_type ==
             ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT) {
    ret = "float32";
  } else if (elem_type == ONNX_NAMESPACE::TensorProto_DataType::
                              TensorProto_DataType_FLOAT16) {
    // It seems that FP16 is float16, and BFLOAT16 is bf16. but xir don't
    // support it or don't care now. test case: adobe fp16 model.
    ret = "float16";
  } else if (elem_type ==
             ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_BOOL) {
    // xir does not support bool.
    // test case: /home/public/bevdet/LS_int.onnx
    ret = "int1";
  } else if (elem_type == ONNX_NAMESPACE::TensorProto_DataType::
                              TensorProto_DataType_UINT16) {
    ret = "uint16";
  } else if (elem_type ==
             ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT16) {
    ret = "int16";
  } else if (elem_type == ONNX_NAMESPACE::TensorProto_DataType::
                              TensorProto_DataType_BFLOAT16) {
    ret = "bfloat16";
  } else if (elem_type ==
             ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT4) {
    ret = "int4";
  } else if (elem_type ==
             ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT4) {
    ret = "uint4";
  } else {
    LOG(FATAL) << "unknown type:" << elem_type;
  }
  return ret;
}

NodeAttr::NodeAttr(const std::string& name, int64_t value)
    : attribute_proto_{
          AttributeProtoPtr(VAIP_ORT_API(attr_proto_new_int)(name, value))} {}

NodeAttr::NodeAttr(const std::string& name, float value)
    : attribute_proto_{VAIP_ORT_API(attr_proto_new_float)(name, value)} {}

NodeAttr::NodeAttr(const std::string& name, const std::string& value)
    : attribute_proto_{VAIP_ORT_API(attr_proto_new_string)(name, value)} {}

NodeAttr::NodeAttr(const std::string& name, const TensorProto& value)
    : attribute_proto_{VAIP_ORT_API(attr_proto_new_tensor)(name, value)} {}

NodeAttr::NodeAttr(const std::string& name, const std::vector<int64_t>& value)
    : attribute_proto_{VAIP_ORT_API(attr_proto_new_ints)(name, value)} {}

NodeAttr::NodeAttr(const std::string& name, const std::vector<float>& value)
    : attribute_proto_{VAIP_ORT_API(attr_proto_new_floats)(name, value)} {}

NodeAttr::NodeAttr(const std::string& name,
                   const std::vector<std::string>& value)
    : attribute_proto_{VAIP_ORT_API(attr_proto_new_strings)(name, value)} {}

NodeAttr::NodeAttr(const std::string& name, AttributeProtoPtr ptr)
    : attribute_proto_{std::move(ptr)} {
  VAIP_ORT_API(attr_proto_set_name)(attribute_proto_.get(), name);
}

AttributeProto& NodeAttr::get() { return *attribute_proto_; }
const AttributeProto& NodeAttr::get() const { return *attribute_proto_; }

NodeAttributesBuilder::NodeAttributesBuilder(size_t capacity) : attrs_{} {
  attrs_.reserve(capacity);
}

NodeAttributesPtr NodeAttributesBuilder::build() {
  auto ret = NodeAttributesPtr(VAIP_ORT_API(node_attributes_new)());
  for (auto& node_attr : attrs_) {
    AttributeProto& attr_proto = node_attr.get();
    VAIP_ORT_API(node_attributes_add)(*ret, std::move(attr_proto));
  }
  attrs_.clear();
  return ret;
}

void NodeAttributesBuilder::merge_into(Node& node) {
  merge_into(VAIP_ORT_API(node_get_attributes)(node));
}

void NodeAttributesBuilder::merge_into(NodeAttributes& attrs) {
  for (auto& attr : attrs_) {
    AttributeProto& attr_proto = attr.get();
    VAIP_ORT_API(node_attributes_add)(attrs, std::move(attr_proto));
  }
}
VAIP_DLL_SPEC AttributeProtoPtr attr_proto_clone(const AttributeProto& attr) {
  return AttributeProtoPtr(VAIP_ORT_API(attr_proto_clone)(attr));
}

VAIP_DLL_SPEC AttributeProtoPtr attr_proto_new_ints(
    const std::string& name, const std::vector<int64_t>& value) {
  return AttributeProtoPtr(VAIP_ORT_API(attr_proto_new_ints)(name, value));
}

} // namespace vaip_core
