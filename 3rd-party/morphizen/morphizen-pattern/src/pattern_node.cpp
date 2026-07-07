/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "./pattern_node.hpp"

#include <glog/logging.h>

#include "./pattern_constant.hpp"
#include "morphizen/node.hpp"
#include "morphizen/node_arg.hpp"

#include "./pattern_log.hpp"

#ifdef _MSC_VER
#  pragma warning(push)
#  pragma warning(disable : 4946)
#endif

#include "morphizen/pattern.pb.h"

#ifdef _MSC_VER
#  pragma warning(pop)
#endif
namespace morphizen {
PatternNode::PatternNode(int id, const std::string& op_type,
                         const std::string& op_domain,
                         std::vector<std::shared_ptr<Pattern>> args,
                         std::vector<bool> is_args_optional)
    : Pattern(id), op_type_(op_type), op_domain_(normalize_domain(op_domain)),
      args_(std::move(args)), is_args_optional_(std::move(is_args_optional)) {
  LOG_IF(ERROR, (op_domain == "ai.onnx") || (op_domain == "onnx"))
      << "Please use \"\" (empty string) as default onnx domain instead of "
      << op_domain;

  CHECK(args_.size() == is_args_optional_.size());
}

PatternNode::~PatternNode() {}

BinderBuilderPtr
PatternNode::match_uncached(const onnxruntime::Graph& graph,
                            const NodeInput& node_input,
                            const BinderBuilder& binder) const {
  if (node_input.node == nullptr) {
    auto node_arg_ref = morphizen_cxx::NodeArgConstRef::from_node_arg(
        graph, *node_input.node_arg);
    MATCH_FAILED << " not a node: " << node_arg_ref.to_string();
    return nullptr;
  }
  auto& node = *node_input.node;
  auto node_ref = morphizen_cxx::NodeConstRef::from_node(graph, node);
  auto domain = normalize_domain(node_ref.op_domain());
  auto op_type = node_ref.op_type();
  if (domain != this->op_domain_ || op_type != this->op_type_) {
    MATCH_FAILED << " expect node_type is " << this->op_domain_ << ":"
                 << this->op_type_ << " actually node type is " << domain << ":"
                 << op_type << node_ref.to_string();
    return nullptr;
  }
  auto inputs = node_ref.inputs_as_node_input();
  auto inputs_size = inputs.size();
  auto args_size = args_.size();
  // you can not match a node with three inputs but two arguments
  if (inputs_size > args_size) {
    MATCH_FAILED << " too many inputs. expect num of args is " << args_size
                 << " actual input size  is " << inputs_size
                 << "; node=" << node_ref.to_string();
    return nullptr;
  }

  if (node_input.node_arg != &node_ref.first_output_node_arg()) {
    MATCH_FAILED << "  PatternNode treats Node as single output, please use "
                    "node_with_multiple_outputs to deal with multiple outputs"
                 << "; node=" << node_ref.to_string();
    return nullptr;
  }

  auto ret = binder.add(this->get_id(), node_input);
  for (auto arg_idx = 0u; arg_idx < inputs_size; ++arg_idx) {
    if (is_args_optional_[arg_idx]) {
      if (node_arg_exists(*inputs[arg_idx].node_arg)) {
        ret = args_[arg_idx]->match_cached(graph, inputs[arg_idx], *ret);
      } else {
        // it is ok if the arg does not exits, because it is optional.
      }
    } else {
      ret = args_[arg_idx]->match_cached(graph, inputs[arg_idx], *ret);
    }
    if (ret == nullptr) {
      MATCH_FAILED << " arg[" << arg_idx
                   << "] is not match, pattern_id=" << args_[arg_idx]->get_id();
      return nullptr;
    }
  }
  for (auto arg_idx = inputs_size; arg_idx < args_size; ++arg_idx) {
    if (!is_args_optional_[arg_idx]) {
      MATCH_FAILED << " arg[" << arg_idx
                   << "] is required. args_size=" << args_size;
      return nullptr;
    }
  }
  MY_LOG(1) << "MATCH OK. ID=" << get_id() << ", node=" << node_ref.to_string();
  return ret;
}

std::string PatternNode::debug_string() const {
  auto ret = std::string("#");
  ret += std::to_string(this->get_id()) + std::string("(");
  ret += this->op_domain_ + ":" + this->op_type_;
  if (!args_.empty()) {
    ret += std::string("(");
    for (auto i = 0u; i < args_.size() - 1; i++) {
      ret += args_[i]->debug_string() + ", ";
    }
    ret += args_[args_.size() - 1]->debug_string();
    ret += std::string(")");
  }
  ret += std::string(")");
  return ret;
}
void PatternNode::dump_to_proto_imp(RootPatternProto& pattern_proto,
                                    PatternProto& this_proto) const {
  auto proto = this_proto.mutable_call_node();
  proto->set_op_type(this->op_type_);
  proto->set_op_domain(this->op_domain_);
  for (auto is_optional : is_args_optional_) {
    proto->add_optional_args(is_optional);
  }
  for (auto arg : args_) {
    auto arg_pattern_proto = arg->dump_to_proto(pattern_proto);
    CHECK(arg_pattern_proto->has_id());
    proto->add_args()->set_name(arg_pattern_proto->id());
  }
}
void PatternNode::fill_ops_name(
    std::vector<std::string>& list_of_ops_name) const {
  for (auto arg : args_) {
    arg->fill_ops_name(list_of_ops_name);
  }
  list_of_ops_name.emplace_back(this->op_type_);
}
} // namespace morphizen
