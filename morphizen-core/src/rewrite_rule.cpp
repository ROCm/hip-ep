/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "morphizen/rewrite_rule.hpp"

#include <glog/logging.h>

#include <memory>

#include "morphizen/env_config.hpp"
#include "morphizen/pass.hpp"
#include "morphizen/util.hpp"
DEF_ENV_PARAM(DEBUG_REWRITE_RULE, "0")
namespace morphizen {
using namespace onnxruntime;
void BaseRule::apply(Graph* graph) {
  IPass* null_pass = nullptr; // NO LINT;
  create_action_from_node_action(
      [this](IPass&, Graph& graph, const Node& node) -> bool {
        return this->apply_once(&graph, &node);
      })(*null_pass, *graph);
  LOG_IF(INFO, ENV_PARAM(DEBUG_REWRITE_RULE) >= 1) << "Rule::apply success";
}
BaseRule::~BaseRule() {}
bool Rule::apply_once(Graph* graph, const Node* node) {
  auto pattern = this->pattern();
  auto binder = pattern->match(*graph, *node); // match_node_arg ??
  if (binder) {
    LOG_IF(INFO, ENV_PARAM(DEBUG_REWRITE_RULE) >= 1)
        << "MATCH  " << node_as_string(*node) << " with pattern "
        << pattern->debug_string() << " binder=" << binder.get();
  }
  return binder && this->action(graph, *binder);
}

class RuleChain : public BaseRule {
public:
  explicit RuleChain(std::vector<std::unique_ptr<BaseRule>>&& chain);
  virtual ~RuleChain();

private:
  virtual bool apply_once(onnxruntime::Graph* graph,
                          const onnxruntime::Node* node) override;

private:
  std::vector<std::unique_ptr<BaseRule>> chain_;
};

RuleChain::RuleChain(std::vector<std::unique_ptr<BaseRule>>&& chain)
    : chain_{std::move(chain)} {}

RuleChain::~RuleChain() {}

bool RuleChain::apply_once(onnxruntime::Graph* graph,
                           const onnxruntime::Node* node) {
  auto ret = false;
  auto b = chain_.begin();
  auto e = chain_.end();
  for (auto it = b; it != e && !ret; ++it) {
    ret = it->get()->apply_once(graph, node);
  }
  return ret;
}

std::unique_ptr<BaseRule>
BaseRule::create_rule_chain(std::vector<std::unique_ptr<BaseRule>>&& chain) {
  return std::make_unique<RuleChain>(std::move(chain));
}

class BasicRule : public Rule {
public:
  BasicRule(std::shared_ptr<Pattern> pattern,
            const std::function<bool(onnxruntime::Graph* graph,
                                     binder_t& binder)>& action);

private:
  /// return true if graph is modified, false otherwise.
  virtual bool action(onnxruntime::Graph* graph,
                      binder_t& binder) const override final;
  const Pattern* pattern() const override final;

private:
  std::shared_ptr<Pattern> pattern_;
  const std::function<bool(onnxruntime::Graph* graph, binder_t& binder)>
      action_;
};

BasicRule::BasicRule(std::shared_ptr<Pattern> pattern,
                     const std::function<bool(onnxruntime::Graph* graph,
                                              binder_t& binder)>& action)
    : pattern_{pattern}, action_{action} {}

bool BasicRule::action(onnxruntime::Graph* graph, binder_t& binder) const {
  return action_(graph, binder);
}

const Pattern* BasicRule::pattern() const { return pattern_.get(); }

std::unique_ptr<Rule> Rule::create_rule(
    std::shared_ptr<Pattern> pattern,
    const std::function<bool(onnxruntime::Graph* graph, binder_t& binder)>&
        action) {
  return std::make_unique<BasicRule>(pattern, action);
}

} // namespace morphizen
