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
#include "./_sanity_check.hpp"
#include <vaip/my_ort.h>
namespace vaip_core {
class NodeAttr {
public:
  VAIP_DLL_SPEC NodeAttr(const std::string& name, int64_t value);
  VAIP_DLL_SPEC NodeAttr(const std::string& name, float value);
  VAIP_DLL_SPEC NodeAttr(const std::string& name, const std::string& value);
  VAIP_DLL_SPEC NodeAttr(const std::string& name, const TensorProto& value);

  VAIP_DLL_SPEC NodeAttr(const std::string& name,
                         const std::vector<int64_t>& value);
  VAIP_DLL_SPEC NodeAttr(const std::string& name,
                         const std::vector<float>& value);
  VAIP_DLL_SPEC NodeAttr(const std::string& name,
                         const std::vector<std::string>& value);
  VAIP_DLL_SPEC NodeAttr(const std::string& name, AttributeProtoPtr ptr);

  VAIP_DLL_SPEC AttributeProto& get();
  VAIP_DLL_SPEC const AttributeProto& get() const;

private:
  AttributeProtoPtr attribute_proto_;
};

class NodeAttributesBuilder {
public:
  VAIP_DLL_SPEC explicit NodeAttributesBuilder(size_t capacity = 10);
  VAIP_DLL_SPEC
  NodeAttributesBuilder(const NodeAttributesBuilder&) = delete;
  VAIP_DLL_SPEC
  NodeAttributesBuilder(NodeAttributesBuilder&&) = default;
  /// after build, all attrs_ are cleared.
  VAIP_DLL_SPEC NodeAttributesPtr build();
  /// for efficiency reason, after merge_into, all attrs_ are
  /// moved.
  VAIP_DLL_SPEC void merge_into(Node& node);
  VAIP_DLL_SPEC void merge_into(NodeAttributes& attrs);
  template <typename T>
  NodeAttributesBuilder& add(const std::string& name, T&& value) {
    attrs_.emplace_back(name, std::forward<T>(value));
    return *this;
  }

private:
  std::vector<NodeAttr> attrs_;
};
std::string attr_proto_as_string(const AttributeProto& attr);
std::vector<int64_t> tensor_proto_get_shape(const TensorProto& tensor_proto);
std::string data_type_to_string(int elem_type);
VAIP_DLL_SPEC AttributeProtoPtr attr_proto_clone(const AttributeProto& attr);
VAIP_DLL_SPEC AttributeProtoPtr
attr_proto_new_ints(const std::string& name, const std::vector<int64_t>& attr);
} // namespace vaip_core
