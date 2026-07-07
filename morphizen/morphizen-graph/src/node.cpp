/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "morphizen/node.hpp"
#include "morphizen/graph.hpp"
#include "morphizen/node_arg.hpp"
#include <glog/logging.h>
#include <limits>
#include <morphizen/morphizen-ort-api-ext.hpp>
#include <morphizen/morphizen_ort_api.h>
#include <morphizen/my_ort.h>

namespace morphizen {

template <typename C> static std::string node_args_as_string_tmpl(const C &c) {
  int index = 0;
  std::ostringstream str;
  str << "[";
  for (auto arg : c) {
    if (index != 0) {
      str << ",";
    }
    if (arg == nullptr) { // optional output node_arg is nullptr
      str << "";
    } else {
      str << node_arg_as_string_internal(*arg);
    }
    index = index + 1;
  }
  str << "]";
  return str.str();
}

std::string node_args_as_string(const std::vector<const NodeArg *> &args) {
  return node_args_as_string_tmpl(args);
}

static std::string node_inputs_as_string(const Node &node) {
  return node_args_as_string(node_get_input_node_args(node));
}

static std::string node_outputs_as_string(const Node &node) {
  return node_args_as_string(node_get_output_node_args(node));
}

std::string node_as_string(const Node &node) {
  std::ostringstream str;
  str << "@" << MORPHIZEN_ORT_API(node_get_index)(node) << " "
      << node_outputs_as_string(node) << " ";
  auto domain = MORPHIZEN_ORT_API(node_op_domain)(node);
  auto op_type = MORPHIZEN_ORT_API(node_op_type)(node);
  if (!domain.empty()) {
    str << domain << "::";
  }
  str << op_type << " ";
  str << node_inputs_as_string(node);
  return str.str();
}

std::vector<NodeInput> node_get_inputs(const Node &node) {
  return *MORPHIZEN_ORT_API(node_get_inputs_unsafe)(node);
}

std::vector<const NodeArg *> node_get_input_node_args(const Node &node) {
  std::vector<const NodeArg *> ret;
  auto node_input = node_get_inputs(node);
  ret.reserve(node_input.size());
  for (auto ni : node_input) {
    ret.push_back(ni.node_arg);
  }
  return ret;
}

// optional output return nullptr
std::vector<const NodeArg *> node_get_output_node_args(const Node &node) {
  return *MORPHIZEN_ORT_API(node_get_output_node_args_unsafe)(node);
}
const NodeArg &node_get_output_node_arg(const Node &node) {
  auto outputs = node_get_output_node_args(node);
  CHECK_EQ(outputs.size(), 1u)
      << "only support single output: node=" << node_as_string(node);
  return *outputs[0];
}

const NodeArg &node_get_first_output_node_arg(const Node &node) {
  auto outputs = node_get_output_node_args(node);
  CHECK_GE(outputs.size(), 1u)
      << "at least 1 output needed: node=" << node_as_string(node);
  return *outputs[0];
}

std::vector<const AttributeProto *> node_get_attributes(const Node &node) {

  std::vector<const AttributeProto *> ret;
  auto &attributes = node_get_attributes_ref(node);
  auto keys = MORPHIZEN_ORT_API(node_attributes_get_keys)(
      const_cast<NodeAttributes &>(attributes));
  ret.reserve(keys->size());
  for (auto &key : *keys) {
    ret.push_back(node_attributes_get(attributes, key));
  }
  return ret;
}

const NodeAttributes &node_get_attributes_ref(const Node &node) {
  auto &ret = MORPHIZEN_ORT_API(node_get_attributes)(const_cast<Node &>(node));
  // CHECK(ret != nullptr) << node_as_string(node);
  return ret;
}

std::vector<int64_t> node_get_output_shape(const Node &node, int index) {
  auto node_args = node_get_output_node_args(node);
  CHECK_LT(index, node_args.size()) << node_as_string(node) << index;
  auto shape = node_arg_get_shape_i64(*node_args[index]);
  CHECK(shape != nullptr) << node_as_string(node) << " shape absent";
  return *shape;
}

const std::string &node_get_output_name(const Node &node) {
  const NodeArg &output = node_get_output_node_arg(node);
  return node_arg_get_name(output);
}

const std::string &node_get_first_output_name(const Node &node) {
  const NodeArg &output = node_get_first_output_node_arg(node);
  return node_arg_get_name(output);
}

bool node_is_op(const Node &node, const std::string &op_type1,
                const std::string &domain1) {
  auto domain = MORPHIZEN_ORT_API(node_op_domain)(node);
  auto op_type = MORPHIZEN_ORT_API(node_op_type)(node);
  auto ret = op_type == op_type1;
  if (domain1.empty() || domain1 == "ai.onnx") {
    ret = ret && (domain.empty() || domain == "ai.onnx");
  } else {
    ret = ret && domain == domain;
  }
  return ret;
}

int node_get_output_element_type(const Node &node) {
  const NodeArg &output = node_get_output_node_arg(node);
  return MORPHIZEN_ORT_API(node_arg_get_element_type)(output);
}

MORPHIZEN_DLL_SPEC bool node_has_attr(const Node &node,
                                      const std::string &name) {
  auto attr = node_attributes_get(node_get_attributes_ref(node), name);
  return attr != nullptr;
}

MORPHIZEN_DLL_SPEC int64_t node_get_attr_int(const Node &node,
                                             const std::string &name) {
  auto attr = node_get_attr(node, name);
  auto value = MORPHIZEN_ORT_API(attr_proto_get_int)(*attr);
  return value;
}
MORPHIZEN_DLL_SPEC int64_t node_get_attr_int_with_default(
    const Node &node, const std::string &name, int64_t default_value) {
  auto ret = default_value;
  if (node_has_attr(node, name)) {
    ret = node_get_attr_int(node, name);
  }
  return ret;
}

MORPHIZEN_DLL_SPEC float node_get_attr_float(const Node &node,
                                             const std::string &name) {
  auto attr = node_get_attr(node, name);
  return MORPHIZEN_ORT_API(attr_proto_get_float)(*attr);
}
MORPHIZEN_DLL_SPEC float
node_get_attr_float_with_default(const Node &node, const std::string &name,
                                 float default_value) {
  auto ret = default_value;
  if (node_has_attr(node, name)) {
    ret = node_get_attr_float(node, name);
  }
  return ret;
}

MORPHIZEN_DLL_SPEC gsl::span<const float>
node_get_attr_floats(const Node &node, const std::string &name) {
  auto attr = node_get_attr(node, name);
  return MORPHIZEN_ORT_API(attr_proto_get_floats)(*attr);
}

MORPHIZEN_DLL_SPEC gsl::span<const int64_t>
node_get_attr_ints(const Node &node, const std::string &name) {
  auto attr = node_get_attr(node, name);
  return MORPHIZEN_ORT_API(attr_proto_get_ints)(*attr);
}
MORPHIZEN_DLL_SPEC const std::string &
node_get_attr_string(const Node &node, const std::string &name) {
  auto attr = node_get_attr(node, name);
  return MORPHIZEN_ORT_API(attr_proto_get_string)(*attr);
}

MORPHIZEN_DLL_SPEC std::vector<std::string>
node_get_attr_strings(const Node &node, const std::string &name) {
  auto &attrs = node_get_attributes_ref(node);
  auto attr_proto = node_attributes_get(attrs, name);
  auto strs_value = MORPHIZEN_ORT_API(attr_proto_get_strings)(*attr_proto);
  return strs_value;
}

MORPHIZEN_DLL_SPEC const std::string &
node_get_attr_string_with_default(const Node &node, const std::string &name,
                                  const std::string &default_value) {
  return node_has_attr(node, name) ? node_get_attr_string(node, name)
                                   : default_value;
}

MORPHIZEN_DLL_SPEC const TensorProto &
node_get_attr_tensor(const Node &node, const std::string &name) {
  auto attr = node_get_attr(node, name);
  return MORPHIZEN_ORT_API(attr_proto_get_tensor)(*attr);
}

MORPHIZEN_DLL_SPEC const AttributeProto *
node_attributes_get(const NodeAttributes &attributes, const std::string &name) {
  return MORPHIZEN_ORT_API(node_attributes_get)(
      const_cast<NodeAttributes &>(attributes), name);
}

// TODO: use template to ensure compatibility
MORPHIZEN_DLL_SPEC morphizen::DllSafe<std::string>
node_release_attr_string(const Node &node, const std::string &name) {
  auto const_attr = node_attributes_get(node_get_attributes_ref(node), name);
  CHECK(const_attr != nullptr);
  return MORPHIZEN_ORT_API(attr_proto_release_string)(
      const_cast<AttributeProto *>(const_attr));
}

MORPHIZEN_DLL_SPEC const AttributeProto *
node_get_attr(const Node &node, const std::string &name) {
  auto attr = node_attributes_get(node_get_attributes_ref(node), name);
  CHECK(attr != nullptr);
  return attr;
}

MORPHIZEN_DLL_SPEC const std::string &node_op_type(const Node &node) {
  return MORPHIZEN_ORT_API(node_op_type)(node);
}

MORPHIZEN_DLL_SPEC const std::string &node_op_domain(const Node &node) {
  return MORPHIZEN_ORT_API(node_op_domain)(node);
}

MORPHIZEN_DLL_SPEC NodeAttributesPtr node_attributes_new() {
  return NodeAttributesPtr(MORPHIZEN_ORT_API(node_attributes_new)());
}

MORPHIZEN_DLL_SPEC NodeAttributesPtr node_clone_attributes(const Node &node) {
  auto ret = node_attributes_new();
  for (auto &attr : node_get_attributes(node)) {
    auto cloned_attr = attr_proto_clone(*attr);
    MORPHIZEN_ORT_API(node_attributes_add)(*ret, std::move(*cloned_attr));
  }
  return ret;
}

size_t node_get_index(const Node &node) {
  return MORPHIZEN_ORT_API(node_get_index)(node);
}

} // namespace morphizen

namespace morphizen_cxx {

size_t NodeConstRef::index() const {
  return MORPHIZEN_ORT_API(node_get_index)(*this);
}
std::vector<std::optional<NodeArgConstRef>> NodeConstRef::inputs() const {
  // Get input node args directly using MORPHIZEN_ORT_API
  auto node_input = *MORPHIZEN_ORT_API(node_get_inputs_unsafe)(*this);
  std::vector<std::optional<NodeArgConstRef>> ret;
  ret.reserve(node_input.size());
  for (auto ni : node_input) {
    auto arg = ni.node_arg;
    if (arg != nullptr && morphizen::node_arg_exists(*arg)) {
      ret.push_back(NodeArgConstRef(this->graph(), *arg));
    } else {
      // in ORT, input could be nullptr or a pointer to the empty node arg
      // represent optional input. we are not sure yet which way is used.
      ret.push_back(std::nullopt);
    }
  }
  return ret;
}
std::vector<morphizen::NodeInput> NodeConstRef::inputs_as_node_input() const {
  return *MORPHIZEN_ORT_API(node_get_inputs_unsafe)(*this);
}
std::vector<std::optional<NodeArgConstRef>>
NodeConstRef::implicit_inputs() const {
  auto node_input =
      *MORPHIZEN_ORT_API_EXT(node_get_implicit_inputs_unsafe)(*this);
  std::vector<std::optional<NodeArgConstRef>> ret;
  ret.reserve(node_input.size());
  for (auto ni : node_input) {
    auto arg = ni.node_arg;
    if (arg != nullptr && morphizen::node_arg_exists(*arg)) {
      ret.push_back(NodeArgConstRef(this->graph(), *arg));
    } else {
      ret.push_back(std::nullopt);
    }
  }
  return ret;
}
std::vector<std::optional<NodeArgConstRef>> NodeConstRef::all_inputs() const {
  auto ret = inputs();
  auto impl = implicit_inputs();
  // NodeArgConstRef holds a const Graph& and is not copy-assignable, so
  // optional<NodeArgConstRef> can't be assigned -- use push_back which
  // only invokes copy-construction.
  ret.reserve(ret.size() + impl.size());
  for (const auto &opt : impl) {
    ret.push_back(opt);
  }
  return ret;
}
const std::string &NodeConstRef::name() const {
  return MORPHIZEN_ORT_API(node_get_name)(*this);
}
const std::string &NodeConstRef::op_type() const {
  return MORPHIZEN_ORT_API(node_op_type)(*this);
}

const std::string &NodeConstRef::op_domain() const {
  return MORPHIZEN_ORT_API(node_op_domain)(*this);
}
std::string NodeConstRef::to_string() const {
  std::ostringstream str;
  str << "@" << MORPHIZEN_ORT_API(node_get_index)(*this) << " ";
  // outputs
  auto output_node_args =
      *MORPHIZEN_ORT_API(node_get_output_node_args_unsafe)(*this);
  str << "[";
  for (size_t i = 0; i < output_node_args.size(); ++i) {
    if (i != 0) {
      str << ",";
    }
    if (output_node_args[i] == nullptr) {
      str << "";
    } else {
      str << morphizen::node_arg_as_string_internal(*output_node_args[i]);
    }
  }
  str << "] ";

  auto domain = MORPHIZEN_ORT_API(node_op_domain)(*this);
  auto op_type = MORPHIZEN_ORT_API(node_op_type)(*this);
  if (!domain.empty()) {
    str << domain << "::";
  }
  str << op_type << " ";

  // inputs
  auto node_input = *MORPHIZEN_ORT_API(node_get_inputs_unsafe)(*this);
  str << "[";
  for (size_t i = 0; i < node_input.size(); ++i) {
    if (i != 0) {
      str << ",";
    }
    auto arg = node_input[i].node_arg;
    if (arg == nullptr) {
      str << "";
    } else {
      str << morphizen::node_arg_as_string_internal(*arg);
    }
  }
  str << "]";
  return str.str();
}
bool NodeConstRef::has_attr(const std::string &name) const {
  return morphizen::node_has_attr(*this, name);
}

int64_t NodeConstRef::get_attr_int(const std::string &name) const {
  return morphizen::node_get_attr_int(*this, name);
}
int64_t NodeConstRef::get_attr_int(const std::string &name,
                                   int64_t default_value) const {
  if (!this->has_attr(name)) {
    return default_value;
  }
  return morphizen::node_get_attr_int(*this, name);
}
gsl::span<const int64_t>
NodeConstRef::get_attr_ints(const std::string &name) const {
  return morphizen::node_get_attr_ints(*this, name);
}
gsl::span<const int64_t>
NodeConstRef::get_attr_ints(const std::string &name,
                            const std::vector<int64_t> &default_value) const {
  if (!this->has_attr(name)) {
    return default_value;
  }
  return this->get_attr_ints(name);
}

float NodeConstRef::get_attr_float(const std::string &name) const {
  return morphizen::node_get_attr_float(*this, name);
}

float NodeConstRef::get_attr_float(const std::string &name,
                                   float default_value) const {
  if (!has_attr(name)) {
    return default_value;
  }
  return morphizen::node_get_attr_float(*this, name);
}

gsl::span<const float>
NodeConstRef::get_attr_floats(const std::string &name) const {
  return morphizen::node_get_attr_floats(*this, name);
}

const std::string &
NodeConstRef::get_attr_string(const std::string &name) const {
  return morphizen::node_get_attr_string(*this, name);
}
const std::string &
NodeConstRef::get_attr_string(const std::string &name,
                              const std::string &default_value) const {
  if (!has_attr(name)) {
    return default_value;
  }
  return morphizen::node_get_attr_string(*this, name);
}

morphizen::DllSafe<std::string>
NodeConstRef::release_attr_string(const std::string &name) const {
  return morphizen::node_release_attr_string(*this, name);
}

std::vector<std::string>
NodeConstRef::get_attr_strings(const std::string &name) const {
  return morphizen::node_get_attr_strings(*this, name);
}
std::vector<std::string> NodeConstRef::get_attr_strings(
    const std::string &name,
    const std::vector<std::string> &default_value) const {
  if (!has_attr(name)) {
    return default_value;
  }
  return morphizen::node_get_attr_strings(*this, name);
}
std::vector<std::optional<NodeArgConstRef>> NodeConstRef::outputs() const {
  auto output_node_args =
      *MORPHIZEN_ORT_API(node_get_output_node_args_unsafe)(*this);
  auto ret = std::vector<std::optional<NodeArgConstRef>>();
  ret.reserve(output_node_args.size());
  for (auto &arg : output_node_args) {
    // in ORT, output could be nullptr, represent optional output.
    if (arg == nullptr) {
      ret.push_back(std::nullopt);
    } else {
      ret.push_back(NodeArgConstRef(this->graph(), *arg));
    }
  }
  return ret;
}
const morphizen::NodeArg &NodeConstRef::first_output_node_arg() const {
  auto outputs = *MORPHIZEN_ORT_API(node_get_output_node_args_unsafe)(*this);
  CHECK_GE(outputs.size(), 1u)
      << "at least 1 output needed: node=" << this->to_string();
  return *outputs[0];
}
GraphConstRef NodeConstRef::get_function_body() const {
  auto &func_body = MORPHIZEN_ORT_API(node_get_function_body)(*this);
  return GraphConstRef(func_body);
}
std::ostream &operator<<(std::ostream &os,
                         const morphizen_cxx::NodeConstRef &node) {
  return os << node.to_string();
}
} // namespace morphizen_cxx
