#pragma once

#include "vaip/vaiml/vaiml_config.hpp"
#include "vaip/vaip.hpp"
#include "vaip/xir_headers.hpp"

namespace vaip_vaiml {
using namespace vaip_core;

class VaimlConfigRule {
public:
  using action_t = std::function<bool(VaimlConfigRule* self, Graph& graph,
                                      VaimlConfigOptions& options)>;
  explicit VaimlConfigRule(IPass& pass);
  bool apply(Graph& graph, VaimlConfigOptions& options);
  VaimlConfigRule& action(const action_t& action) {
    actions_.push_back(action);
    return *this;
  }
  VaimlConfigRule& set_config_options();
  VaimlConfigRule& get_model_type();
  VaimlConfigRule& print_options();

public:
  IPass& pass_;

private:
  std::vector<action_t> actions_;
};

void applyVaimlConfigRule(IPass& pass, Graph& graph,
                          VaimlConfigOptions& options);

} // namespace vaip_vaiml