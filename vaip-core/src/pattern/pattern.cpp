/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "morphizen/pattern.hpp"
#include "morphizen/pattern.pb.h"
#include <algorithm>
#include <bitset>
#include <glog/logging.h>
#include <google/protobuf/util/json_util.h>
#include <memory>
#include <numeric>
#include <vaip/vaip_ort_api.h>

#include "./pattern_commutable_node.hpp"
#include "./pattern_constant.hpp"
#include "./pattern_graph_input.hpp"
#include "./pattern_log.hpp"
#include "./pattern_node.hpp"
#include "./pattern_or.hpp"
#include "./pattern_sequence.hpp"
#include "./pattern_where.hpp"
#include "./pattern_wildcard.hpp"
#include "morphizen/node.hpp"
#include "morphizen/node_arg.hpp"
#include "morphizen/util.hpp"
#ifdef ENABLE_PYTHON
#  include <pybind11/embed.h>
#  include <pybind11/pybind11.h>
namespace py = pybind11;
#endif
#include "./immutable_map.hpp"
namespace vaip_core {
std::optional<vaip_cxx::NodeInput>
Binder::create_vaip_cxx_node_input(NodeInput node_input) const {
  if (node_input.node_arg == nullptr) {
    return std::nullopt;
  }
  return vaip_cxx::NodeInput{graph_, *node_input.node_arg, node_input.node};
}
std::optional<vaip_cxx::NodeInput> Binder::operator()(size_t pattern_id) const {
  return create_vaip_cxx_node_input((*this)[pattern_id]);
}
std::optional<vaip_cxx::NodeInput>
Binder::operator()(const std::string& pattern_name) const {
  return create_vaip_cxx_node_input((*this)[pattern_name]);
}

using Map = immutable_map::ImmutableMap<int, NodeInput>;
void Pattern::enable_trace(int n) { ENV_PARAM(DEBUG_VAIP_PATTERN) = n; }
Pattern::Pattern(int id) : id_{id} {}
Pattern::~Pattern() {}

binder_ptr_t Pattern::match(const onnxruntime::Graph& graph,
                            const onnxruntime::Node& node) const {
  // if node has no output, it does not match any pattern.
  // node is useless if it has no output.
  auto outputs = node_get_output_node_args(node);
  auto size = (size_t)get_id() + 1;
  auto store = std::make_shared<std::vector<NodeInput>>();
  store->resize(size);
  // outputs[i] is only used if it is a graph input or constant
  for (auto i = 0u; i < outputs.size(); ++i) {
    auto init = BinderBuilderPtr(new BinderBuilder(new Map(), graph));
    auto ret = this->match_cached(graph, {&node, outputs[i]}, *init);
    if (ret != nullptr) {
      return ret->build(name_to_ids_);
    }
  }
  return nullptr;
}
binder_ptr_t Pattern::match(vaip_cxx::NodeConstRef node) const {
  return match(node.graph(), node);
}

BinderBuilderPtr Pattern::match_cached(const onnxruntime::Graph& graph,
                                       const NodeInput& node_input,
                                       const BinderBuilder& binder) const {
  auto id = this->get_id();
  auto ret = BinderBuilderPtr();
  auto matched_node_input = binder.find(id);
  if (matched_node_input.node_arg) {
    if (matched_node_input.node == node_input.node &&
        matched_node_input.node_arg == node_input.node_arg) {
      ret = binder.clone();
    } else {
      MATCH_FAILED << "MATCH cache failed."
                   << "pattern[id=" << get_id() << "]"
                   << " matched node_arg{"
                   << node_arg_as_string(*matched_node_input.node_arg) << "}"
                   << " it cannot matched the other node_arg{"
                   << node_arg_as_string(*node_input.node_arg) << "}";
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
  options.always_print_primitive_fields = true;
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
    std::vector<std::string>& /*list_of_ops_name*/) const {
  return;
}

PatternProto* Pattern::dump_to_proto(RootPatternProto& pattern_proto) const {
  PatternProto* found = nullptr;
  auto id = std::to_string(get_id());
  for (auto& name_to_id : *name_to_ids_) {
    if (name_to_id.second == get_id()) {
      id = name_to_id.first;
      break;
    }
  }
  for (auto& pat : *pattern_proto.mutable_patterns()) {
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
void Pattern::dump_to_proto_imp(RootPatternProto& /*pattern_proto*/,
                                PatternProto& /*this_proto*/) const {
  LOG(FATAL) << "not implemented.";
}

std::string Pattern::debug_string() const {
  return std::string("debug_string is not implemented yet");
}

std::string Pattern::virtualize_label() const {
  return std::string("virtualize_label is not implemented yet");
}

PatternBuilder::PatternBuilder()
    : id_map_{std::make_shared<std::unordered_map<std::string, int>>()} {}
static std::shared_ptr<Pattern>
PatternBuilder_create(PatternBuilder* self, const PatternProto& pattern_proto);

static std::shared_ptr<Pattern>
PatternBuilder_build_arg(PatternBuilder* self,
                         const vaip_core::PatternCallNodeArgProto& arg) {
  auto ret = std::shared_ptr<Pattern>();
  switch (arg.arg_case()) {
  case PatternCallNodeArgProto::kName:
    ret = self->get_pattern(arg.name());
    break;
  case PatternCallNodeArgProto::kPattern:
    ret = PatternBuilder_create(self, arg.pattern());
    break;
  default:
    ret = nullptr;
  }
  CHECK(ret != nullptr) << arg.DebugString();
  return ret;
}

static std::vector<std::shared_ptr<Pattern>>
PatternBuilder_build_args(PatternBuilder* self,
                          const google::protobuf::RepeatedPtrField<
                              vaip_core::PatternCallNodeArgProto>& args) {
  auto ret = std::vector<std::shared_ptr<Pattern>>{};
  ret.reserve(args.size());
  for (auto& arg : args) {
    ret.push_back(PatternBuilder_build_arg(self, arg));
  }
  return ret;
}

static std::shared_ptr<Pattern>
PatternBuilder_create(PatternBuilder* self, const PatternProto& pattern_proto) {
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
    auto op_type = pattern_proto.call_node().op_type();
    auto args =
        PatternBuilder_build_args(self, pattern_proto.call_node().args());
    std::vector<bool> optional_args(
        pattern_proto.call_node().optional_args().begin(),
        pattern_proto.call_node().optional_args().end());
    ret = self->node3(op_type, args, optional_args);
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
PatternBuilder::create_by_json(const std::string& pattern_json) {
  RootPatternProto pattern_proto;
  auto status =
      google::protobuf::util::JsonStringToMessage(pattern_json, &pattern_proto);
  if (!status.ok()) {
    LOG(WARNING) << "cannot parse json string:" << pattern_json;
    return nullptr;
  }
  auto ret = std::shared_ptr<Pattern>{};
  auto last = std::shared_ptr<Pattern>{};
  for (auto& p : pattern_proto.patterns()) {
    last = PatternBuilder_create(this, p);
    if (p.is_root()) {
      ret = last;
    }
  }
  if (ret == nullptr) {
    ret = last;
  }
  return ret;
}

std::shared_ptr<Pattern> PatternBuilder::create_from_binary(const char* data,
                                                            size_t size) {
  RootPatternProto pattern_proto;
  auto ok = pattern_proto.ParseFromArray(data, (int)size);
  CHECK(ok) << "cannot parse  protobuf data";
  auto ret = std::shared_ptr<Pattern>{};
  auto last = std::shared_ptr<Pattern>{};
  for (auto& p : pattern_proto.patterns()) {
    last = PatternBuilder_create(this, p);
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
PatternBuilder::create_by_py(const std::string& pattern) {
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
  } catch (py::error_already_set& e) {
    LOG(FATAL) << e.what();
  }
  return nullptr;
}
#endif

std::shared_ptr<Pattern> PatternBuilder::wildcard() {
  return create_internal([](int id) { return new PatternWildcard(id); });
}

std::shared_ptr<Pattern>
PatternBuilder::node2(const std::string& op_type,
                      const std::vector<std::shared_ptr<Pattern>>& args) {
  return create_internal([=](int id) {
    auto is_args_optional = std::vector<bool>(args.size(), false);
    return new PatternNode(id, op_type, std::move(args),
                           std::move(is_args_optional));
  });
}

std::shared_ptr<Pattern>
PatternBuilder::node3(const std::string& op_type,
                      const std::vector<std::shared_ptr<Pattern>>& args,
                      const std::vector<bool>& optional_args) {
  return create_internal([=](int id) {
    return new PatternNode(id, op_type, std::move(args),
                           std::move(optional_args));
  });
}

std::shared_ptr<Pattern>
PatternBuilder::commutable_node(const std::string& op_type,
                                std::shared_ptr<Pattern> arg1,
                                std::shared_ptr<Pattern> arg2) {
  return create_internal([=](int id) {
    return new PatternCommutableNode(id, op_type, arg1, arg2);
  });
}

std::shared_ptr<Pattern>
PatternBuilder::sequence(gsl::span<const std::shared_ptr<Pattern>> patterns) {
  return create_internal(
      [=](int id) { return new PatternSequence(id, patterns); });
}

std::shared_ptr<Pattern>
PatternBuilder::Or(const std::vector<std::shared_ptr<Pattern>>& args) {
  return create_internal([=](int id) { return new PatternOr(id, args); });
}

std::shared_ptr<Pattern> PatternBuilder::constant() {
  return create_internal([](int id) { return new PatternConstant(id); });
}

std::shared_ptr<Pattern> PatternBuilder::graph_input() {
  return create_internal([](int id) { return new PatternGraphInput(id); });
}

std::shared_ptr<Pattern> PatternBuilder::xir_const_op() {
  return node2("com.xilinx:const", {});
}

void PatternBuilder::bind(const std::string& name,
                          const std::shared_ptr<Pattern>& pat) {

  (*id_map_)[name] = pat->get_id();
}

int PatternBuilder::get_id(const std::string& name) const {
  auto it = id_map_->find(name);
  auto ret = -1;
  if (it != id_map_->end()) {
    ret = it->second;
  }
  return ret;
}

std::shared_ptr<Pattern>
PatternBuilder::get_pattern(const std::string& name) const {
  auto it = id_map_->find(name);
  auto ret = std::shared_ptr<Pattern>{};
  if (it != id_map_->end()) {
    ret = patterns_[it->second];
  }
  return ret;
}

std::shared_ptr<Pattern>
PatternBuilder::create_internal(const std::function<Pattern*(int id)>& f) {
  auto id = (int)patterns_.size();
  auto ret = std::shared_ptr<Pattern>(f(id));
  patterns_.push_back(ret);
  ret->name_to_ids_ = id_map_;
  return ret;
}

std::unordered_map<std::string, int> PatternBuilder::bindings() const {
  return *id_map_;
}

BinderBuilder::~BinderBuilder() {
  auto p = (Map*)map_;
  CHECK(p != nullptr);
  delete p;
}

binder_ptr_t BinderBuilder::build(
    const std::shared_ptr<std::unordered_map<std::string, int>>& name_to_ids)
    const {
  const auto& map = *(Map*)map_;
  MY_LOG(1) << "build binder results: " << map;
  auto store = std::map<int, NodeInput>();
  for (auto& x : map) {
    store.emplace(x);
  }
  return std::unique_ptr<Binder>(
      new Binder(std::move(store), name_to_ids, graph_));
}

BinderBuilderPtr BinderBuilder::add(int id, const NodeInput& node_input) const {
  const auto& map = *(Map*)map_;
  return BinderBuilderPtr(
      new BinderBuilder(new Map(map.insert({id, node_input})), graph_));
}

NodeInput BinderBuilder::find(int id) const {
  const auto& map = *(Map*)map_;
  auto ret = NodeInput{nullptr, nullptr};
  auto it = map.find(id);
  MY_LOG(3) << "build binder results: " << map;
  if (it != nullptr) {
    ret = *it;
  }
  return ret;
}

BinderBuilderPtr BinderBuilder::clone() const {
  const auto& map = *(Map*)map_;
  return BinderBuilderPtr(new BinderBuilder(new Map(map), graph_));
}

} // namespace vaip_core
