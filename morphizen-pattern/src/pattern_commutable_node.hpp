/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "morphizen/pattern.hpp"

namespace morphizen {

class PatternCommutableNode : public Pattern {
public:
  explicit PatternCommutableNode(int id, const std::string& op_type,
                                 const std::string& op_domain,
                                 const std::shared_ptr<Pattern>& arg1,
                                 const std::shared_ptr<Pattern>& arg2);
  ~PatternCommutableNode();

private:
  virtual BinderBuilderPtr
  match_uncached(const onnxruntime::Graph& graph, const NodeInput& node_input,
                 const BinderBuilder& binder) const override final;
  virtual std::string debug_string() const final;
  virtual void fill_ops_name(
      std::vector<std::string>& list_of_ops_name) const override final;

private:
  const std::string op_type_;
  const std::string op_domain_;
  const std::shared_ptr<Pattern> arg1_;
  const std::shared_ptr<Pattern> arg2_;
};
} // namespace morphizen
