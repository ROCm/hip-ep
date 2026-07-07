/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "./pattern_graph_output.hpp"

#include <algorithm>
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

PatternGraphOutput::PatternGraphOutput(int id, std::shared_ptr<Pattern> arg)
    : Pattern(id), arg_(arg) {}

PatternGraphOutput::PatternGraphOutput(int id, std::shared_ptr<Pattern> arg,
                                       size_t graph_output_index)
    : Pattern(id), arg_(arg), graph_output_index_(graph_output_index) {}

PatternGraphOutput::PatternGraphOutput(int id, std::shared_ptr<Pattern> arg,
                                       const std::string& graph_output_name)
    : Pattern(id), arg_(arg), graph_output_name_(graph_output_name) {}

PatternGraphOutput::~PatternGraphOutput() {}

BinderBuilderPtr
PatternGraphOutput::match_uncached(const onnxruntime::Graph& graph,
                                   const NodeInput& node_input,
                                   const BinderBuilder& binder) const {
  // Get the list of graph outputs (model outputs)
  auto graph_output_args = graph_get_outputs(graph);

  // Mode 1: Match by specific index
  if (graph_output_index_.has_value()) {
    auto graph_output_index = graph_output_index_.value();

    // Validate index is within bounds
    if (graph_output_index >= graph_output_args.size()) {
      MATCH_FAILED << " graph output index " << graph_output_index
                   << " is out of range " << graph_output_args.size();
      return nullptr;
    }

    // Get the graph output at specified index
    auto graph_output_arg = graph_output_args[graph_output_index];
    if (!graph_output_arg || !node_arg_exists(*graph_output_arg)) {
      MATCH_FAILED << " graph output at index " << graph_output_index
                   << " not exist";
      return nullptr;
    }

    // Verify node_input matches this specific graph output
    if (node_input.node_arg != graph_output_arg) {
      MATCH_FAILED << " unmatched node args "
                   << morphizen_cxx::NodeArgConstRef::from_node_arg(
                          graph, *node_input.node_arg)
                          .to_string()
                   << " and "
                   << morphizen_cxx::NodeArgConstRef::from_node_arg(
                          graph, *graph_output_arg)
                          .to_string();
      return nullptr;
    }
  }
  // Mode 2: Match by name
  else if (graph_output_name_.has_value()) {
    std::string graph_output_name = graph_output_name_.value();

    // Search for graph output with matching name
    auto it = std::find_if(
        graph_output_args.begin(), graph_output_args.end(),
        [&](const NodeArg* arg) -> bool {
          return (arg && node_arg_exists(*arg) &&
                  morphizen_cxx::NodeArgConstRef::from_node_arg(graph, *arg)
                          .name() == graph_output_name);
        });

    // Verify output with this name exists
    if (it == graph_output_args.end()) {
      MATCH_FAILED << " graph output with name " << graph_output_name
                   << " not exist";
      return nullptr;
    }

    // Verify node_input matches the found output
    if (node_input.node_arg != *it) {
      MATCH_FAILED << " unmatched node args "
                   << morphizen_cxx::NodeArgConstRef::from_node_arg(
                          graph, *node_input.node_arg)
                          .to_string()
                   << " and "
                   << morphizen_cxx::NodeArgConstRef::from_node_arg(graph, **it)
                          .to_string();
      return nullptr;
    }
  }
  // Mode 3: Wildcard - match any graph output
  else {
    // Search if node_input is any of the graph outputs
    auto it = std::find_if(graph_output_args.begin(), graph_output_args.end(),
                           [&](const NodeArg* arg) -> bool {
                             return (arg && node_arg_exists(*arg) &&
                                     node_input.node_arg == arg);
                           });

    // Verify node_input is in the graph outputs list
    if (it == graph_output_args.end()) {
      MATCH_FAILED << morphizen_cxx::NodeArgConstRef::from_node_arg(
                          graph, *node_input.node_arg)
                          .to_string()
                   << " unmatched with any graph output ";
      return nullptr;
    }
  }

  // Add this pattern to the binder cache
  auto ret = binder.add(this->get_id(), node_input);

  // Continue matching the underlying pattern
  // This allows chaining: is_graph_output(node2("Relu", {...}))
  return arg_->match_cached(graph, node_input, *ret);
}

std::string PatternGraphOutput::debug_string() const {
  auto ret = std::string("#");
  ret += std::to_string(this->get_id()) + std::string("(");
  if (graph_output_index_.has_value()) {
    ret += std::string("GRAPH_OUTPUT_INDEX(") +
           std::to_string(graph_output_index_.value()) + std::string(")");
  } else if (graph_output_name_.has_value()) {
    ret += std::string("GRAPH_OUTPUT_NAME(") + graph_output_name_.value() +
           std::string(")");
  } else {
    ret += std::string("GRAPH_OUTPUT");
  }
  ret += std::string("(") + arg_->debug_string() + std::string(")");
  ret += std::string(")");
  return ret;
}

void PatternGraphOutput::dump_to_proto_imp(RootPatternProto& pattern_proto,
                                           PatternProto& this_proto) const {
  auto proto = this_proto.mutable_graph_output();
  auto arg_proto = arg_->dump_to_proto(pattern_proto);

  CHECK(arg_proto->has_id());
  proto->mutable_node_arg()->set_name(arg_proto->id());

  if (graph_output_index_.has_value()) {
    proto->set_graph_output_index(graph_output_index_.value());
  } else if (graph_output_name_.has_value()) {
    proto->set_graph_output_name(graph_output_name_.value());
  }
}

} // namespace morphizen
