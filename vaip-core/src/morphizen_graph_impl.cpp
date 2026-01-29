/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// TEMPORARY: These implementations from morphizen-graph are duplicated here
// to resolve linker issues with static library symbol visibility.
// TODO: Fix the proper dllexport/dllimport setup and move these back.

#include "morphizen/graph.hpp"
#include "morphizen/node_attr.hpp"
#include <vaip/vaip_ort_api.h>

namespace vaip_core {

// From morphizen-graph/src/node_attr.cpp
void AttributeProtoDeleter::operator()(AttributeProto* p) const {
  VAIP_ORT_API(attr_proto_delete)(p);
}

} // namespace vaip_core

namespace vaip_cxx {

// From morphizen-graph/src/graph.cpp
NodeArgConstRef
GraphRef::new_node_arg(const std::string& name,
                       const std::vector<int64_t>& shape,
                       ONNX_NAMESPACE::TensorProto_DataType data_type) {
  return NodeArgConstRef::from_node_arg(
      self(), VAIP_ORT_API(node_arg_new)(*this, name, &shape, data_type));
}

} // namespace vaip_cxx
