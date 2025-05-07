/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
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
  /**
   * @brief Retrieves the list of node attributes.
   *
   * @return A constant reference to a vector containing the node attributes.
   */
  const std::vector<NodeAttr>& get() const { return attrs_; }

private:
  std::vector<NodeAttr> attrs_;
};
std::string data_type_to_string(int elem_type);
VAIP_DLL_SPEC AttributeProtoPtr attr_proto_clone(const AttributeProto& attr);
VAIP_DLL_SPEC AttributeProtoPtr
attr_proto_new_ints(const std::string& name, const std::vector<int64_t>& attr);
} // namespace vaip_core
