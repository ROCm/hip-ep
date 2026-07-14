/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "./pattern_commutable_node.hpp"
#include "./pattern_log.hpp"
#include "morphizen/graph.hpp"
#include "morphizen/node.hpp"
#include "morphizen/node_arg.hpp"

namespace morphizen {
PatternCommutableNode::PatternCommutableNode(
    int id, const std::string &op_type, const std::string &op_domain,
    const std::shared_ptr<Pattern> &arg1, const std::shared_ptr<Pattern> &arg2)
    : Pattern(id), op_type_(op_type), op_domain_(normalize_domain(op_domain)),
      arg1_(arg1), arg2_(arg2) {
  LOG_IF(ERROR, (op_domain == "ai.onnx") || (op_domain == "onnx"))
      << "Please use \"\" (empty string) as default onnx domain instead of "
      << op_domain;

  CHECK(arg1_ != nullptr);
  CHECK(arg2_ != nullptr);
}

PatternCommutableNode::~PatternCommutableNode() {}

BinderBuilderPtr
PatternCommutableNode::match_uncached(const onnxruntime::Graph &graph,
                                      const NodeInput &node_input,
                                      const BinderBuilder &binder) const {
  if (node_input.node == nullptr) {
    auto node_arg_ref = morphizen_cxx::NodeArgConstRef::from_node_arg(
        graph, *node_input.node_arg);
    MATCH_FAILED << " not a node: " << node_arg_ref.to_string();
    return nullptr;
  }
  const auto &node = *node_input.node;
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
  if (inputs.size() != 2) {
    MATCH_FAILED << " expect 2 inputs, actually " << inputs.size()
                 << node_ref.to_string();
    return nullptr;
  }
  const auto ret0 = binder.add(this->get_id(), node_input);
  auto match = [&](const std::shared_ptr<Pattern> &arg1,
                   const std::shared_ptr<Pattern> &arg2) -> BinderBuilderPtr {
    MY_LOG(1) << " ID=" << get_id() << " try pattern arg1=" << arg1->get_id()
              << " arg2=" << arg2->get_id();
    auto match_ret = arg1->match_cached(graph, inputs[0], *ret0);
    if (match_ret) {
      match_ret = arg2->match_cached(graph, inputs[1], *match_ret);
    }
    if (match_ret == nullptr) {
      MY_LOG(1) << " ID=" << get_id() << " MATCH FAILED. "
                << "[p1=" << arg1->get_id() << " ,p2=" << arg2->get_id() << "]";
    }
    return match_ret;
  };
  auto ret = BinderBuilderPtr();

  auto ret12 = match(arg1_, arg2_);
  if (ret12 != nullptr) {
    MY_LOG(1) << " ID=" << get_id()         //
              << " match "                  //
              << " ["                       //
              << "arg1=" << arg1_->get_id() //
              << ","                        //
              << "arg2=" << arg2_->get_id() //
              << "] "
              << " OK";
    ret = std::move(ret12);
  } else {
    MY_LOG(1) << " ID=" << get_id()         //
              << " match "                  //
              << " ["                       //
              << "arg1=" << arg1_->get_id() //
              << ","                        //
              << "arg2=" << arg2_->get_id() //
              << "] "
              << " failed. try "
              << " ["                       //
              << "arg2=" << arg2_->get_id() //
              << ","                        //
              << "arg1=" << arg1_->get_id() //
              << "] ";
    auto ret21 = match(arg2_, arg1_);
    ret = std::move(ret21);
  }
  if (ret == nullptr) {
    MATCH_FAILED << " both arg match failed."
                 << " arg1=" << arg1_->get_id() << " arg2=" << arg2_->get_id();
  } else {
    MY_LOG(1) << "MATCH OK. ID=" << get_id()
              << ", node=" << node_ref.to_string();
  }
  return ret;
}
std::string PatternCommutableNode::debug_string() const {
  auto ret = std::string("#");
  ret += std::to_string(this->get_id()) + std::string("(");
  ret += this->op_domain_ + ":" + this->op_type_;
  ret += std::string("(");
  ret += arg1_->debug_string() + ", " + arg2_->debug_string();
  ret += std::string(")");
  return ret;
}
void PatternCommutableNode::fill_ops_name(
    std::vector<std::string> &list_of_ops_name) const {
  arg1_->fill_ops_name(list_of_ops_name);
  list_of_ops_name.emplace_back("CommutableNode -> " + this->op_type_);
  arg2_->fill_ops_name(list_of_ops_name);
  list_of_ops_name.emplace_back("CommutableNode -> " + this->op_type_);
}
} // namespace morphizen
