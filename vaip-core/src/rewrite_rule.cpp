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

#include "vaip/rewrite_rule.hpp"

#include <glog/logging.h>

#include <memory>

#include "vaip/pass.hpp"
#include "vaip/util.hpp"
#include "vitis/ai/env_config.hpp"
DEF_ENV_PARAM(DEBUG_REWRITE_RULE, "0")
namespace vaip_core {
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

} // namespace vaip_core
