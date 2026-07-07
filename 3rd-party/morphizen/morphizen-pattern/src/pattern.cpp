/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "morphizen/pattern.hpp"

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4946)
#endif

#include "morphizen/pattern.pb.h"

#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include <algorithm>
#include <bitset>
#include <glog/logging.h>
#include <google/protobuf/util/json_util.h>
#include <memory>
#include <numeric>

#include "./pattern_commutable_node.hpp"
#include "./pattern_constant.hpp"
#include "./pattern_graph_input.hpp"
#include "./pattern_graph_output.hpp"
#include "./pattern_log.hpp"
#include "./pattern_node.hpp"
#include "./pattern_node_output_arg.hpp"
#include "./pattern_or.hpp"
#include "./pattern_sequence.hpp"
#include "./pattern_where.hpp"
#include "./pattern_wildcard.hpp"
#include "morphizen/node.hpp"
#include "morphizen/node_arg.hpp"

#ifdef ENABLE_PYTHON
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;
#endif
#include "./immutable_map.hpp"
// NOTE: onnx-schema.hpp must be included last as it redefines ONNX_NAMESPACE
// to morphizen_onnx to prevent naming conflicts.
#if MORPHIZEN_HAS_ONNX_SCHEMA_SUPPORT
#include "morphizen/onnx_schema.hpp"
#endif
namespace morphizen {
std::optional<morphizen_cxx::NodeInput>
Binder::create_morphizen_cxx_node_input(NodeInput node_input) const {
  if (node_input.node_arg == nullptr) {
    return std::nullopt;
  }
  return morphizen_cxx::NodeInput{graph_, *node_input.node_arg,
                                  node_input.node};
}
std::optional<morphizen_cxx::NodeInput>
Binder::operator()(size_t pattern_id) const {
  return create_morphizen_cxx_node_input((*this)[pattern_id]);
}
std::optional<morphizen_cxx::NodeInput>
Binder::operator()(const std::string &pattern_name) const {
  return create_morphizen_cxx_node_input((*this)[pattern_name]);
}

using Map = immutable_map::ImmutableMap<int, NodeInput>;

// Design Decision: Pattern::enable_trace() is a no-op
//
// Rationale:
// During component extraction, we chose to keep morphizen-pattern independent
// with minimal dependencies. The original enable_trace() relied on env_config
// from morphizen-core, which would create a dependency we're trying to avoid.
//
// Tradeoffs Considered:
// 1. Add morphizen-utils dependency for ENV_PARAM:
//    - Pro: Restores dynamic trace level control
//    - Con: Adds dependency, contradicts extraction goals
// 2. Use std::getenv() directly:
//    - Pro: No dependencies, simple
//    - Con: Less robust than ENV_PARAM, platform-specific
// 3. Make it a no-op (CHOSEN):
//    - Pro: Zero dependencies, simpler build graph
//    - Con: Lost runtime trace control
//
// Decision: Keep as no-op because:
// - Component independence is more valuable than this debug feature
// - MY_LOG is always-on in pattern_log.hpp (users still have logging)
// - Trace control is a development/debug feature, not core functionality
// - Users can still use glog's VLOG if they need filtered logging
//
// Alternative for users who need trace control:
// - Set GLOG_v environment variable to control glog verbosity
// - Use custom build with modified pattern_log.hpp
void Pattern::enable_trace(int n) {
  (void)n; // Parameter kept for API compatibility
}
Pattern::Pattern(int id) : id_{id} {}
Pattern::~Pattern() {}

binder_ptr_t Pattern::match(const onnxruntime::Graph &graph,
                            const onnxruntime::Node &node) const {
  // if node has no output, it does not match any pattern.
  // node is useless if it has no output.
  auto node_ref = morphizen_cxx::NodeConstRef::from_node(graph, node);
  auto outputs = node_ref.outputs();
  for (auto i = 0u; i < outputs.size(); ++i) {
    if (!outputs[i].has_value())
      continue; // Skip optional outputs
    auto init = BinderBuilderPtr(new BinderBuilder(new Map(), graph));
    const morphizen::NodeArg *output_arg =
        &(static_cast<const morphizen::NodeArg &>(outputs[i].value()));
    auto ret = this->match_cached(graph, {&node, output_arg}, *init);
    if (ret != nullptr) {
      return ret->build(name_to_ids_);
    }
  }

  return nullptr;
}
binder_ptr_t Pattern::match(morphizen_cxx::NodeConstRef node) const {
  return match(node.graph(), node);
}

BinderBuilderPtr Pattern::match_cached(const onnxruntime::Graph &graph,
                                       const NodeInput &node_input,
                                       const BinderBuilder &binder) const {
  auto id = this->get_id();
  auto ret = BinderBuilderPtr();
  auto matched_node_input = binder.find(id);
  if (matched_node_input.node_arg) {
    if (matched_node_input.node == node_input.node &&
        matched_node_input.node_arg == node_input.node_arg) {
      ret = binder.clone();
    } else {
      auto matched_arg_ref = morphizen_cxx::NodeArgConstRef::from_node_arg(
          graph, *matched_node_input.node_arg);
      auto node_arg_ref = morphizen_cxx::NodeArgConstRef::from_node_arg(
          graph, *node_input.node_arg);
      MATCH_FAILED << "MATCH cache failed."
                   << "pattern[id=" << get_id() << "]"
                   << " matched node_arg{" << matched_arg_ref.to_string() << "}"
                   << " it cannot matched the other node_arg{"
                   << node_arg_ref.to_string() << "}";
      ret = nullptr;
    }
  } else {
    ret = this->match_uncached(graph, node_input, binder);
  }
  return ret;
}

std::string Pattern::to_binary() const {
  RootPatternProto root_pattern_proto;
  auto patter_proto = dump_to_proto(root_pattern_proto);
  patter_proto->set_is_root(true);
  std::reverse(root_pattern_proto.mutable_patterns()->begin(),
               root_pattern_proto.mutable_patterns()->end());

  auto ret = std::string();
  CHECK(root_pattern_proto.SerializeToString(&ret))
      << "cannot serialized to string";
  return ret;
}
std::string Pattern::to_json() const {
  RootPatternProto root_pattern_proto;
  auto patter_proto = dump_to_proto(root_pattern_proto);
  patter_proto->set_is_root(true);
  std::reverse(root_pattern_proto.mutable_patterns()->begin(),
               root_pattern_proto.mutable_patterns()->end());
  std::string ret;
  google::protobuf::util::JsonPrintOptions options;
  options.add_whitespace = true;
  // options.always_print_primitive_fields = true;
  auto status = google::protobuf::util::MessageToJsonString(root_pattern_proto,
                                                            &ret, options);
  CHECK(status.ok()) << "cannot serialized to json";
  return ret;
}
std::vector<std::string> Pattern::get_ops_list_name() const {
  std::vector<std::string> ret;
  fill_ops_name(ret);
  return ret;
}

void Pattern::fill_ops_name(
    std::vector<std::string> & /*list_of_ops_name*/) const {
  return;
}

PatternProto *Pattern::dump_to_proto(RootPatternProto &pattern_proto) const {
  PatternProto *found = nullptr;
  auto id = std::to_string(get_id());
  for (auto &name_to_id : *name_to_ids_) {
    if (name_to_id.second == get_id()) {
      id = name_to_id.first;
      break;
    }
  }
  for (auto &pat : *pattern_proto.mutable_patterns()) {
    if (pat.id() == id) {
      found = &pat;
      break;
    }
  }
  if (found) {
    return found;
  }
  auto new_pattern = pattern_proto.add_patterns();
  new_pattern->set_id(id);
  new_pattern->set_is_root(false);
  dump_to_proto_imp(pattern_proto, *new_pattern);
  return new_pattern;
}
void Pattern::dump_to_proto_imp(RootPatternProto & /*pattern_proto*/,
                                PatternProto & /*this_proto*/) const {
  LOG(FATAL) << "not implemented.";
}

std::string Pattern::debug_string() const {
  return std::string("debug_string is not implemented yet");
}

std::string Pattern::virtualize_label() const {
  return std::string("virtualize_label is not implemented yet");
}

namespace {
// Helper function to parse op_type_and_domain string
std::pair<std::string, std::string>
parse_op_type_and_domain(const std::string &op_type_and_domain) {
  auto colon_pos = op_type_and_domain.find(':');
  if (colon_pos != std::string::npos) {
    return {op_type_and_domain.substr(0, colon_pos),
            op_type_and_domain.substr(colon_pos + 1)};
  } else {
    return {"", op_type_and_domain}; // Domain is optional
  }
}
} // anonymous namespace

struct PatternBuilderHelper {
  static std::shared_ptr<Pattern> create(PatternBuilder *self,
                                         const PatternProto &pattern_proto);

  static std::shared_ptr<Pattern>
  build_arg(PatternBuilder *self,
            const morphizen::PatternCallNodeArgProto &arg);

  static std::vector<std::shared_ptr<Pattern>>
  build_args(PatternBuilder *self,
             const google::protobuf::RepeatedPtrField<
                 morphizen::PatternCallNodeArgProto> &args);
};

std::shared_ptr<Pattern>
PatternBuilderHelper::create(PatternBuilder *self,
                             const PatternProto &pattern_proto) {
  auto ret = std::shared_ptr<Pattern>();
  switch (pattern_proto.type_case()) {
  case PatternProto::kWildcard:
    ret = self->wildcard();
    break;
  case PatternProto::kConstant:
    ret = self->constant();
    break;
  case PatternProto::kGraphInput:
    // todo match name:
    ret = self->graph_input();
    break;
  case PatternProto::kCallNode: {
    auto [op_domain, op_type] =
        parse_op_type_and_domain(pattern_proto.call_node().op_type());
    if (op_domain.empty()) {
      op_domain = pattern_proto.call_node().op_domain();
    } else {
      LOG(ERROR) << "Please use op_domain field to store op domain seperately.";
    }
    auto args = build_args(self, pattern_proto.call_node().args());
    std::vector<bool> optional_args(
        pattern_proto.call_node().optional_args().begin(),
        pattern_proto.call_node().optional_args().end());
    ret = self->node3_with_optional_domain(op_type, args, optional_args,
                                           op_domain);
  } break;
  case PatternProto::kNodeOutputArg: {
    auto &node_output_arg_proto = pattern_proto.node_output_arg();
    auto call_node_pattern = build_arg(self, node_output_arg_proto.call_node());
    ret = self->get_node_output_arg_by_index(
        call_node_pattern, node_output_arg_proto.output_arg_index());
  } break;
  case PatternProto::kGraphOutput: {
    auto &graph_output_proto = pattern_proto.graph_output();
    auto node_arg_pattern = build_arg(self, graph_output_proto.node_arg());
    if (graph_output_proto.has_graph_output_index()) {
      ret = self->is_graph_output(node_arg_pattern,
                                  graph_output_proto.graph_output_index());
    } else if (graph_output_proto.has_graph_output_name()) {
      ret = self->is_graph_output(node_arg_pattern,
                                  graph_output_proto.graph_output_name());
    } else {
      ret = self->is_graph_output(node_arg_pattern);
    }
  } break;
  default:
    ret = nullptr;
  }
  if (ret && pattern_proto.has_id()) {
    self->bind(pattern_proto.id(), ret);
  }
  return ret;
}

std::shared_ptr<Pattern>
PatternBuilderHelper::build_arg(PatternBuilder *self,
                                const morphizen::PatternCallNodeArgProto &arg) {
  auto ret = std::shared_ptr<Pattern>();
  switch (arg.arg_case()) {
  case PatternCallNodeArgProto::kName:
    ret = self->get_pattern(arg.name());
    break;
  case PatternCallNodeArgProto::kPattern:
    ret = create(self, arg.pattern());
    break;
  default:
    ret = nullptr;
  }
  CHECK(ret != nullptr) << arg.DebugString();
  return ret;
}

std::vector<std::shared_ptr<Pattern>> PatternBuilderHelper::build_args(
    PatternBuilder *self,
    const google::protobuf::RepeatedPtrField<morphizen::PatternCallNodeArgProto>
        &args) {
  auto ret = std::vector<std::shared_ptr<Pattern>>{};
  ret.reserve(args.size());
  for (auto &arg : args) {
    ret.push_back(build_arg(self, arg));
  }
  return ret;
}

PatternBuilder::PatternBuilder()
    : id_map_{std::make_shared<std::unordered_map<std::string, int>>()} {}

std::shared_ptr<Pattern>
PatternBuilder::create_by_json(const std::string &pattern_json) {
  RootPatternProto pattern_proto;
  auto status =
      google::protobuf::util::JsonStringToMessage(pattern_json, &pattern_proto);
  if (!status.ok()) {
    LOG(WARNING) << "cannot parse json string:" << pattern_json;
    return nullptr;
  }
  auto ret = std::shared_ptr<Pattern>{};
  auto last = std::shared_ptr<Pattern>{};
  for (auto &p : pattern_proto.patterns()) {
    last = PatternBuilderHelper::create(this, p);
    if (p.is_root()) {
      ret = last;
    }
  }
  if (ret == nullptr) {
    ret = last;
  }
  return ret;
}

std::shared_ptr<Pattern> PatternBuilder::create_from_binary(const char *data,
                                                            size_t size) {
  RootPatternProto pattern_proto;
  auto ok = pattern_proto.ParseFromArray(data, (int)size);
  CHECK(ok) << "cannot parse  protobuf data";
  auto ret = std::shared_ptr<Pattern>{};
  auto last = std::shared_ptr<Pattern>{};
  for (auto &p : pattern_proto.patterns()) {
    last = PatternBuilderHelper::create(this, p);
    if (p.is_root()) {
      ret = last;
    }
  }
  if (ret == nullptr) {
    ret = last;
  }
  return ret;
}

#ifdef ENABLE_PYTHON
std::shared_ptr<Pattern>
PatternBuilder::create_by_py(const std::string &pattern) {
  auto inter = init_interpreter();
  try {
    py::gil_scoped_acquire acquire;
    auto locals = py::globals();
    auto m = py::module::import("voe.pattern");
    locals["wildcard"] = m.attr("wildcard");
    locals["graph_input"] = m.attr("graph_input");
    locals["node"] = m.attr("node");

    py::exec(pattern, locals, locals);
    auto has_pattern = locals.contains("pattern");
    CHECK(has_pattern) << "python code need define a pattern function";

    auto py_pattern = locals["pattern"]();
    auto is_pattern_f = m.attr("is_pattern");
    bool is_pattern = py::cast<bool>(is_pattern_f(py_pattern));
    CHECK(is_pattern) << "python pattern code has error";

    std::string json_string = py::cast<std::string>(py_pattern);
    return create_by_json(json_string);
  } catch (py::error_already_set &e) {
    LOG(FATAL) << e.what();
  }
  return nullptr;
}
#endif

std::shared_ptr<Pattern> PatternBuilder::wildcard() {
  return create_internal([](int id) { return new PatternWildcard(id); });
}

std::shared_ptr<Pattern>
PatternBuilder::node2(const std::string &op_type_and_domain,
                      const std::vector<std::shared_ptr<Pattern>> &args) {
  auto [op_domain, op_type] = parse_op_type_and_domain(op_type_and_domain);
  return node2_with_optional_domain(op_type, args, op_domain);
}

std::shared_ptr<Pattern> PatternBuilder::node2_with_optional_domain(
    const std::string &op_type,
    const std::vector<std::shared_ptr<Pattern>> &args,
    const std::string &op_domain) {
  auto is_args_optional = std::vector<bool>(args.size(), false);
  return node3_with_optional_domain(op_type, args, is_args_optional, op_domain);
}

std::shared_ptr<Pattern>
PatternBuilder::node3(const std::string &op_type_and_domain,
                      const std::vector<std::shared_ptr<Pattern>> &args,
                      const std::vector<bool> &optional_args) {
  auto [op_domain, op_type] = parse_op_type_and_domain(op_type_and_domain);
  return node3_with_optional_domain(op_type, args, optional_args, op_domain);
}

std::shared_ptr<Pattern> PatternBuilder::node3_with_optional_domain(
    const std::string &op_type,
    const std::vector<std::shared_ptr<Pattern>> &args,
    const std::vector<bool> &optional_args, const std::string &op_domain) {
  return create_internal([=](int id) {
    return new PatternNode(id, op_type, op_domain, std::move(args),
                           std::move(optional_args));
  });
}

#if MORPHIZEN_HAS_ONNX_SCHEMA_SUPPORT
std::shared_ptr<Pattern> PatternBuilder::node_with_named_args(
    const std::string &op_type,
    const std::map<std::string, std::shared_ptr<Pattern>> &named_args,
    const std::string &op_domain) {
  // Get the OpSchema to understand input names and their positions
  const auto *schema = morphizen::GetOpSchema(
      op_type, op_domain); // Use high version number to get latest schema

  if (!schema) {
    LOG(FATAL) << "No schema found for op_type: " << op_type
               << " in domain: " << op_domain;
    return nullptr;
  }

  // Build ordered argument list based on schema input definitions
  std::vector<std::shared_ptr<Pattern>> args;
  std::vector<bool> optional_args;

  // Get schema inputs
  auto inputs = schema->inputs();
  args.resize(inputs.size());
  optional_args.resize(inputs.size(), true); // Default to optional

  // Fill arguments based on schema input order
  for (size_t i = 0; i < inputs.size(); ++i) {
    const auto &input = inputs[i];
    std::string input_name = input.GetName();

    // Look for exact match first
    auto it = named_args.find(input_name);
    bool user_marked_optional = false;

    if (it == named_args.end()) {
      // Look for user-provided optional version (name with '*' suffix)
      it = named_args.find(input_name + "*");
      if (it != named_args.end()) {
        user_marked_optional = true;
      }
    }

    if (it != named_args.end() && it->second != nullptr) {
      args[i] = it->second;
      // Argument is required unless:
      // 1. It's explicitly nullptr
      // 2. User marked it optional with '*' suffix
      // 3. Schema defines it as optional
      optional_args[i] =
          user_marked_optional ||
          (input.GetOption() !=
           morphizen_onnx::OpSchema::FormalParameterOption::Single);
    } else {
      // Argument not specified - use wildcard pattern to match any input
      args[i] = wildcard();
      optional_args[i] = true;
    }
  }

  // Handle any extra named arguments that don't match schema inputs
  for (const auto &[name, pattern] : named_args) {
    std::string clean_name = name;
    if (!clean_name.empty() && clean_name.back() == '*') {
      clean_name.pop_back();
    }

    // Check if this argument was already handled
    bool found = false;
    for (size_t i = 0; i < inputs.size(); ++i) {
      if (inputs[i].GetName() == clean_name) {
        found = true;
        break;
      }
    }

    if (!found) {
      LOG(WARNING) << "Named argument '" << name << "' not found in schema for "
                   << op_type << " in domain " << op_domain;
    }
  }

  return node3_with_optional_domain(op_type, args, optional_args, op_domain);
}
#endif // MORPHIZEN_HAS_ONNX_SCHEMA_SUPPORT

std::vector<std::shared_ptr<Pattern>>
PatternBuilder::node_with_multiple_outputs(
    const std::string &op_type,
    const std::vector<std::shared_ptr<Pattern>> &args,
    const std::vector<bool> &optional_args, const std::string &op_domain,
    const size_t num_of_outputs) {
  auto node =
      node3_with_optional_domain(op_type, args, optional_args, op_domain);

  std::vector<std::shared_ptr<Pattern>> ret;
  for (size_t i = 0; i < num_of_outputs; ++i) {
    ret.push_back(get_node_output_arg_by_index(node, i));
  }

  return ret;
}

std::shared_ptr<Pattern>
PatternBuilder::commutable_node(const std::string &op_type_and_domain,
                                std::shared_ptr<Pattern> arg1,
                                std::shared_ptr<Pattern> arg2) {
  auto [op_domain, op_type] = parse_op_type_and_domain(op_type_and_domain);
  return create_internal([=](int id) {
    return new PatternCommutableNode(id, op_type, op_domain, arg1, arg2);
  });
}

std::shared_ptr<Pattern>
PatternBuilder::sequence(gsl::span<const std::shared_ptr<Pattern>> patterns) {
  return create_internal(
      [=](int id) { return new PatternSequence(id, patterns); });
}

std::shared_ptr<Pattern>
PatternBuilder::Or(const std::vector<std::shared_ptr<Pattern>> &args) {
  return create_internal([=](int id) { return new PatternOr(id, args); });
}

std::shared_ptr<Pattern> PatternBuilder::constant() {
  return create_internal([](int id) { return new PatternConstant(id); });
}

std::shared_ptr<Pattern> PatternBuilder::graph_input() {
  return create_internal([](int id) { return new PatternGraphInput(id); });
}

std::shared_ptr<Pattern>
PatternBuilder::is_graph_output(const std::shared_ptr<Pattern> &arg) {
  return create_internal(
      [=](int id) { return new PatternGraphOutput(id, arg); });
}

std::shared_ptr<Pattern>
PatternBuilder::is_graph_output(const std::shared_ptr<Pattern> &arg,
                                size_t graph_output_index) {
  return create_internal([=](int id) {
    return new PatternGraphOutput(id, arg, graph_output_index);
  });
}

std::shared_ptr<Pattern>
PatternBuilder::is_graph_output(const std::shared_ptr<Pattern> &arg,
                                const std::string &graph_output_name) {
  return create_internal([=](int id) {
    return new PatternGraphOutput(id, arg, graph_output_name);
  });
}

std::shared_ptr<Pattern> PatternBuilder::xir_const_op() {
  return node2("com.xilinx:const", {});
}

void PatternBuilder::bind(const std::string &name,
                          const std::shared_ptr<Pattern> &pat) {
  (*id_map_)[name] = pat->get_id();
}

int PatternBuilder::get_id(const std::string &name) const {
  auto it = id_map_->find(name);
  auto ret = -1;
  if (it != id_map_->end()) {
    ret = it->second;
  }
  return ret;
}

std::shared_ptr<Pattern>
PatternBuilder::get_pattern(const std::string &name) const {
  auto it = id_map_->find(name);
  auto ret = std::shared_ptr<Pattern>{};
  if (it != id_map_->end()) {
    ret = patterns_[it->second];
  }
  return ret;
}

std::shared_ptr<Pattern>
PatternBuilder::create_internal(const std::function<Pattern *(int id)> &f) {
  auto id = (int)patterns_.size();
  auto ret = std::shared_ptr<Pattern>(f(id));
  patterns_.push_back(ret);
  ret->name_to_ids_ = id_map_;
  return ret;
}

std::shared_ptr<Pattern> PatternBuilder::get_node_output_arg_by_index(
    const std::shared_ptr<Pattern> &arg, size_t output_arg_index) {
  CHECK(std::dynamic_pointer_cast<PatternNode>(arg))
      << " get_node_output_arg_by_index only accepts parameter of type "
         "PatternNode";
  return create_internal([=](int id) {
    return new PatternNodeOutputArg(id, arg, output_arg_index);
  });
}

std::unordered_map<std::string, int> PatternBuilder::bindings() const {
  return *id_map_;
}

BinderBuilder::~BinderBuilder() {
  auto p = (Map *)map_;
  CHECK(p != nullptr);
  delete p;
}

binder_ptr_t BinderBuilder::build(
    const std::shared_ptr<std::unordered_map<std::string, int>> &name_to_ids)
    const {
  const auto &map = *(Map *)map_;
  MY_LOG(1) << "build binder results: " << map;
  auto store = std::map<int, NodeInput>();
  for (auto &x : map) {
    store.emplace(x);
  }
  return std::unique_ptr<Binder>(
      new Binder(std::move(store), name_to_ids, graph_));
}

BinderBuilderPtr BinderBuilder::add(int id, const NodeInput &node_input) const {
  const auto &map = *(Map *)map_;
  return BinderBuilderPtr(
      new BinderBuilder(new Map(map.insert({id, node_input})), graph_));
}

NodeInput BinderBuilder::find(int id) const {
  const auto &map = *(Map *)map_;
  auto ret = NodeInput{nullptr, nullptr};
  auto it = map.find(id);
  MY_LOG(3) << "build binder results: " << map;
  if (it != nullptr) {
    ret = *it;
  }
  return ret;
}

BinderBuilderPtr BinderBuilder::clone() const {
  const auto &map = *(Map *)map_;
  return BinderBuilderPtr(new BinderBuilder(new Map(map), graph_));
}

} // namespace morphizen
