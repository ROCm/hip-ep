/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#pragma once
#include "morphizen/_sanity_check.hpp"
#include "morphizen/pattern.hpp"

namespace vaip_core {
class BaseRule {
public:
  VAIP_DLL_SPEC static std::unique_ptr<BaseRule>
  create_rule_chain(std::vector<std::unique_ptr<BaseRule>>&& chain);
  VAIP_DLL_SPEC void apply(onnxruntime::Graph* graph);
  virtual bool apply_once(onnxruntime::Graph* graph,
                          const onnxruntime::Node* node) = 0;
  VAIP_DLL_SPEC virtual ~BaseRule();
};

class Rule : public BaseRule {
public:
  VAIP_DLL_SPEC static std::unique_ptr<Rule> create_rule(
      std::shared_ptr<Pattern> pattern,
      const std::function<bool(onnxruntime::Graph* graph, binder_t& binder)>&
          action);

public:
  explicit Rule() = default;
  virtual ~Rule() = default;

private:
  /// return true if graph is modified, false otherwise.
  virtual bool action(onnxruntime::Graph* graph, binder_t& binder) const = 0;
  virtual const Pattern* pattern() const = 0;

private:
  // it must be VAIP_DLL_SPEC because all derived classes in other DLLs need put
  // this function into vtable.
  VAIP_DLL_SPEC virtual bool
  apply_once(onnxruntime::Graph* graph,
             const onnxruntime::Node* node) override final;
};

} // namespace vaip_core
