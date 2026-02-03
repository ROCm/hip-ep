/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "./pattern_node_output_arg.hpp"

#include <glog/logging.h>

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

PatternNodeOutputArg::PatternNodeOutputArg(
    int id, std::shared_ptr<Pattern> node_pattern, size_t output_arg_index)
    : Pattern(id), node_pattern_(node_pattern),
      output_arg_index_(output_arg_index) {}

PatternNodeOutputArg::~PatternNodeOutputArg() {}

BinderBuilderPtr
PatternNodeOutputArg::match_uncached(const onnxruntime::Graph& graph,
                                     const NodeInput& node_input,
                                     const BinderBuilder& binder) const {
  // Step 1: Verify this is an output of a node (has a producer)
  if (node_input.node == nullptr) {
    auto node_arg_ref = morphizen_cxx::NodeArgConstRef::from_node_arg(
        graph, *node_input.node_arg);
    MATCH_FAILED << " not a node: " << node_arg_ref.to_string();
    return nullptr;
  }

  // Step 2: Get the producer node and its outputs
  auto& node = *node_input.node;
  auto node_ref = morphizen_cxx::NodeConstRef::from_node(graph, node);
  auto output_args_vec = node_ref.outputs();
  std::vector<const morphizen::NodeArg*> output_args;
  for (auto& opt_arg : output_args_vec) {
    output_args.push_back(
        opt_arg.has_value()
            ? &(static_cast<const morphizen::NodeArg&>(opt_arg.value()))
            : nullptr);
  }
  CHECK_GE(output_args.size(), 1u)
      << "at least 1 output needed: node=" << node_ref.to_string();

  // Step 3: validate output index
  // Step 3a: Check bounds
  if (output_arg_index_ >= output_args.size()) {
    MATCH_FAILED << " output arg index " << output_arg_index_
                 << " is out of range " << output_args.size();
    return nullptr;
  }

  // Step 3b: Verify output exists (some outputs are optional)
  auto output_arg = output_args.at(output_arg_index_);
  if (!output_arg || !node_arg_exists(*output_arg)) {
    MATCH_FAILED << " output node arg at index " << output_arg_index_
                 << " not exist";
    return nullptr;
  }

  // Step 3c: Verify node_input matches this specific output
  // This is the KEY constraint - ensures we match the right output
  if (node_input.node_arg != output_arg) {
    auto node_arg_ref1 = morphizen_cxx::NodeArgConstRef::from_node_arg(
        graph, *node_input.node_arg);
    auto node_arg_ref2 =
        morphizen_cxx::NodeArgConstRef::from_node_arg(graph, *output_arg);
    MATCH_FAILED << " unmatched node args " << node_arg_ref1.to_string()
                 << " and " << node_arg_ref2.to_string();
    return nullptr;
  }

  // Step 4: Add this pattern to the binder cache
  auto ret = binder.add(this->get_id(), node_input);

  // Step 5: Continue matching the underlying node pattern
  // Always use the first output NodeArg of Node
  // This will match the producer node (e.g., LayerNormalization)
  return node_pattern_->match_cached(
      graph, {node_input.node, output_args.at(0)}, *ret);
}

std::string PatternNodeOutputArg::debug_string() const {
  auto ret = std::string("#");
  ret += std::to_string(this->get_id()) + std::string("(");
  ret += std::string("OUTPUT_ARG_INDEX(") + std::to_string(output_arg_index_) +
         std::string(")");
  ret += std::string("(") + node_pattern_->debug_string() + std::string(")");
  ret += std::string(")");
  return ret;
}

void PatternNodeOutputArg::dump_to_proto_imp(RootPatternProto& pattern_proto,
                                             PatternProto& this_proto) const {
  auto proto = this_proto.mutable_node_output_arg();
  auto node_pattern_proto = node_pattern_->dump_to_proto(pattern_proto);

  CHECK(node_pattern_proto->has_id());
  proto->mutable_call_node()->set_name(node_pattern_proto->id());

  proto->set_output_arg_index(output_arg_index_);
}

} // namespace morphizen
