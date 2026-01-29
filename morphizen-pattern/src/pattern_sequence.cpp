/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "./pattern_sequence.hpp"
#include "./pattern_log.hpp"
#include "morphizen/node.hpp"
#include "morphizen/node_arg.hpp"
namespace morphizen {
PatternSequence::PatternSequence(
    int id, gsl::span<const std::shared_ptr<Pattern>> patterns)
    : Pattern(id), patterns_{patterns.begin(), patterns.end()} {
  CHECK(!patterns.empty()) << "wronge number of patterns.";
}
PatternSequence::~PatternSequence() {}
std::string PatternSequence::debug_string() const {
  auto ret = std::string("#");
  ret += std::to_string(this->get_id()) + std::string("(");
  ret += std::string("*");
  ret += std::string(")");
  return ret;
}

BinderBuilderPtr
PatternSequence::match_uncached(const onnxruntime::Graph& graph1,
                                const NodeInput& node_input,
                                const BinderBuilder& binder) const {
  auto graph = morphizen_cxx::GraphConstRef(graph1);
  auto ret = patterns_.front()->match_cached(graph, node_input, binder);
  if (ret == nullptr) {
    MY_LOG(1) << "MATCH FAIL. ID=" << get_id()
              << ", the first pattern does not matched: "
              << (node_input.node != nullptr
                      ? morphizen_cxx::NodeConstRef::from_node(graph,
                                                               *node_input.node)
                            .to_string()
                      : morphizen_cxx::NodeArgConstRef::from_node_arg(
                            graph, *node_input.node_arg)
                            .to_string());
    return nullptr;
  }
  auto nodes = graph.nodes();
  auto patter_size = patterns_.size();
  for (size_t i = 1; i < patter_size; ++i) {
    auto result_i = morphizen::BinderBuilderPtr();
    MY_LOG(1) << "PatternSequence try to match pattern"    //
              << "[" << i << " of " << patter_size << "] " //
              << "in all other nodes.\n";
    auto mached_node = std::optional<morphizen_cxx::NodeConstRef>();
    for (auto j = 0u; j < nodes.size(); ++j) {
      auto node = nodes[j];
      for (auto output_node_arg : node.outputs()) {
        if (output_node_arg == std::nullopt) {
          continue;
        }
        auto node_input_1 =
            morphizen::NodeInput{node.ptr(), output_node_arg->ptr()};
        result_i = patterns_[i]->match_cached(graph, node_input_1, *ret);
        if (result_i != nullptr) {
          mached_node = node;
          break;
        }
      }
      if (result_i != nullptr) {
        break;
      }
      MY_LOG(1) << "PatternSequence continue to try other nodes ignore above "
                   "errors"
                << std::endl;
    }
    if (result_i != nullptr) {
      MY_LOG(1) << "MATCH OK. sub-pattern"                   //
                << "[" << i << " of " << patter_size << "] " //
                << " match " << *mached_node;
      ret = std::move(result_i);
    } else {
      MY_LOG(1) << "MATCH FAIL. PatternSequence ID=" << get_id()
                << " sub-pattern"
                << "[" << i << " of " << patter_size
                << "] does not match any node: ";
      ret = nullptr;
    }
    if (ret == nullptr) {
      return nullptr;
    }
  }
  MY_LOG(1) << "MATCH OK. ID=" << get_id() << ", sequence matched: "
            << (node_input.node != nullptr
                    ? morphizen_cxx::NodeConstRef::from_node(graph,
                                                             *node_input.node)
                          .to_string()
                    : morphizen_cxx::NodeArgConstRef::from_node_arg(
                          graph, *node_input.node_arg)
                          .to_string());
  return ret;
}

} // namespace morphizen
