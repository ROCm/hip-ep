// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <morphizen-utils/json.hpp>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

namespace morphizen {

struct PatternProto;

struct PatternWildcardProto {
  json::Json ToJson() const { return json::Json::object(); }
  static PatternWildcardProto FromJson(const json::Json &) {
    return PatternWildcardProto();
  }
};

struct PatternGraphInputProto {
  json::Json ToJson() const { return json::Json::object(); }
  static PatternGraphInputProto FromJson(const json::Json &) {
    return PatternGraphInputProto();
  }
};

struct PatternConstantProto {
  json::Json ToJson() const { return json::Json::object(); }
  static PatternConstantProto FromJson(const json::Json &) {
    return PatternConstantProto();
  }
};

struct PatternCallNodeArgProto {
  PatternCallNodeArgProto();
  PatternCallNodeArgProto(const PatternCallNodeArgProto &other);
  PatternCallNodeArgProto &operator=(const PatternCallNodeArgProto &other);
  PatternCallNodeArgProto(PatternCallNodeArgProto &&) noexcept;
  PatternCallNodeArgProto &operator=(PatternCallNodeArgProto &&) noexcept;
  ~PatternCallNodeArgProto();

  enum ArgCase { ARG_NOT_SET = 0, kName = 1, kPattern = 2 };
  ArgCase arg_case_ = ARG_NOT_SET;
  std::string name_;
  std::unique_ptr<PatternProto> pattern_;

  ArgCase arg_case() const { return arg_case_; }
  const std::string &name() const { return name_; }
  void set_name(std::string value);
  const PatternProto &pattern() const;
  PatternProto *mutable_pattern();
  json::Json ToJson() const;
  static PatternCallNodeArgProto FromJson(const json::Json &json);
  std::string DebugString() const;
};

struct PatternCallNodeProto {
  std::string op_type_;
  std::string op_domain_;
  ProtoList<PatternCallNodeArgProto> args_;
  ProtoList<bool> optional_args_;

  const std::string &op_type() const { return op_type_; }
  const std::string &op_domain() const { return op_domain_; }
  void set_op_type(std::string value) { op_type_ = std::move(value); }
  void set_op_domain(std::string value) { op_domain_ = std::move(value); }
  const ProtoList<PatternCallNodeArgProto> &args() const { return args_; }
  PatternCallNodeArgProto *add_args() { return args_.Add(); }
  const ProtoList<bool> &optional_args() const { return optional_args_; }
  void add_optional_args(bool value) { optional_args_.Add(value); }
  json::Json ToJson() const;
  static PatternCallNodeProto FromJson(const json::Json &json);
};

struct PatternNodeOutputArgProto {
  PatternCallNodeArgProto call_node_;
  uint64_t output_arg_index_ = 0;
  const PatternCallNodeArgProto &call_node() const { return call_node_; }
  PatternCallNodeArgProto *mutable_call_node() { return &call_node_; }
  uint64_t output_arg_index() const { return output_arg_index_; }
  void set_output_arg_index(uint64_t value) { output_arg_index_ = value; }
  json::Json ToJson() const;
  static PatternNodeOutputArgProto FromJson(const json::Json &json);
};

struct PatternGraphOutputProto {
  PatternCallNodeArgProto node_arg_;
  std::optional<uint64_t> graph_output_index_;
  std::optional<std::string> graph_output_name_;
  const PatternCallNodeArgProto &node_arg() const { return node_arg_; }
  PatternCallNodeArgProto *mutable_node_arg() { return &node_arg_; }
  bool has_graph_output_index() const { return graph_output_index_.has_value(); }
  uint64_t graph_output_index() const {
    return graph_output_index_ ? *graph_output_index_ : 0;
  }
  void set_graph_output_index(uint64_t value) { graph_output_index_ = value; }
  bool has_graph_output_name() const { return graph_output_name_.has_value(); }
  const std::string &graph_output_name() const {
    if (!graph_output_name_) {
      static const std::string empty;
      return empty;
    }
    return *graph_output_name_;
  }
  void set_graph_output_name(std::string value) {
    graph_output_name_ = std::move(value);
  }
  json::Json ToJson() const;
  static PatternGraphOutputProto FromJson(const json::Json &json);
};

struct PatternProto {
  enum TypeCase {
    TYPE_NOT_SET = 0,
    kWildcard = 10,
    kGraphInput = 11,
    kConstant = 12,
    kCallNode = 13,
    kNodeOutputArg = 14,
    kGraphOutput = 15,
  };

  std::optional<std::string> id_;
  bool is_root_ = false;
  TypeCase type_case_ = TYPE_NOT_SET;
  PatternWildcardProto wildcard_;
  PatternGraphInputProto graph_input_;
  PatternConstantProto constant_;
  PatternCallNodeProto call_node_;
  PatternNodeOutputArgProto node_output_arg_;
  PatternGraphOutputProto graph_output_;

  bool has_id() const { return id_.has_value(); }
  const std::string &id() const {
    if (!id_) {
      static const std::string empty;
      return empty;
    }
    return *id_;
  }
  void set_id(std::string value) { id_ = std::move(value); }
  bool is_root() const { return is_root_; }
  void set_is_root(bool value) { is_root_ = value; }
  TypeCase type_case() const { return type_case_; }
  PatternWildcardProto *mutable_wildcard() {
    type_case_ = kWildcard;
    return &wildcard_;
  }
  PatternGraphInputProto *mutable_graph_input() {
    type_case_ = kGraphInput;
    return &graph_input_;
  }
  PatternConstantProto *mutable_constant() {
    type_case_ = kConstant;
    return &constant_;
  }
  const PatternCallNodeProto &call_node() const { return call_node_; }
  PatternCallNodeProto *mutable_call_node() {
    type_case_ = kCallNode;
    return &call_node_;
  }
  const PatternNodeOutputArgProto &node_output_arg() const {
    return node_output_arg_;
  }
  PatternNodeOutputArgProto *mutable_node_output_arg() {
    type_case_ = kNodeOutputArg;
    return &node_output_arg_;
  }
  const PatternGraphOutputProto &graph_output() const { return graph_output_; }
  PatternGraphOutputProto *mutable_graph_output() {
    type_case_ = kGraphOutput;
    return &graph_output_;
  }
  json::Json ToJson() const;
  static PatternProto FromJson(const json::Json &json);
  std::string DebugString() const { return json::dump(ToJson(), true); }
};

struct RootPatternProto {
  ProtoList<PatternProto> patterns_;
  const ProtoList<PatternProto> &patterns() const { return patterns_; }
  ProtoList<PatternProto> *mutable_patterns() { return &patterns_; }
  PatternProto *add_patterns() { return patterns_.Add(); }
  json::Json ToJson() const;
  static RootPatternProto FromJson(const json::Json &json);
  static RootPatternProto FromJsonString(std::string_view text) {
    return FromJson(json::parse(text));
  }
  std::string DebugString() const { return json::dump(ToJson(), true); }
};

inline PatternCallNodeArgProto::PatternCallNodeArgProto() = default;
inline PatternCallNodeArgProto::PatternCallNodeArgProto(
    const PatternCallNodeArgProto &other)
    : arg_case_(other.arg_case_), name_(other.name_),
      pattern_(other.pattern_ ? std::make_unique<PatternProto>(*other.pattern_)
                              : nullptr) {}
inline PatternCallNodeArgProto &
PatternCallNodeArgProto::operator=(const PatternCallNodeArgProto &other) {
  if (this != &other) {
    PatternCallNodeArgProto copy(other);
    *this = std::move(copy);
  }
  return *this;
}
inline PatternCallNodeArgProto::PatternCallNodeArgProto(
    PatternCallNodeArgProto &&) noexcept = default;
inline PatternCallNodeArgProto &PatternCallNodeArgProto::operator=(
    PatternCallNodeArgProto &&) noexcept = default;
inline PatternCallNodeArgProto::~PatternCallNodeArgProto() = default;

inline void PatternCallNodeArgProto::set_name(std::string value) {
  arg_case_ = kName;
  name_ = std::move(value);
  pattern_.reset();
}
inline const PatternProto &PatternCallNodeArgProto::pattern() const {
  return *pattern_;
}
inline PatternProto *PatternCallNodeArgProto::mutable_pattern() {
  if (!pattern_) {
    pattern_ = std::make_unique<PatternProto>();
  }
  arg_case_ = kPattern;
  name_.clear();
  return pattern_.get();
}

inline json::Json PatternCallNodeArgProto::ToJson() const {
  json::Json::Object object;
  if (arg_case_ == kName) {
    object.emplace("name", json::Json::str(name_));
  } else if (arg_case_ == kPattern && pattern_) {
    object.emplace("pattern", pattern_->ToJson());
  }
  return json::Json::object(std::move(object));
}
inline PatternCallNodeArgProto
PatternCallNodeArgProto::FromJson(const json::Json &json) {
  PatternCallNodeArgProto value;
  if (const json::Json *name = json::object_field(json, "name")) {
    value.set_name(name->as_string());
  } else if (const json::Json *pattern = json::object_field(json, "pattern")) {
    *value.mutable_pattern() = PatternProto::FromJson(*pattern);
  }
  return value;
}
inline std::string PatternCallNodeArgProto::DebugString() const {
  return json::dump(ToJson(), true);
}

inline json::Json PatternCallNodeProto::ToJson() const {
  json::Json::Array args;
  for (const auto &arg : args_) {
    args.push_back(arg.ToJson());
  }
  json::Json::Array optional_args;
  for (bool value : optional_args_) {
    optional_args.push_back(json::Json::boolean(value));
  }
  json::Json::Object object;
  if (!op_type_.empty()) {
    object.emplace("op_type", json::Json::str(op_type_));
  }
  if (!op_domain_.empty()) {
    object.emplace("op_domain", json::Json::str(op_domain_));
  }
  if (!args.empty()) {
    object.emplace("args", json::Json::array(std::move(args)));
  }
  if (!optional_args.empty()) {
    object.emplace("optional_args", json::Json::array(std::move(optional_args)));
  }
  return json::Json::object(std::move(object));
}
inline PatternCallNodeProto
PatternCallNodeProto::FromJson(const json::Json &json) {
  PatternCallNodeProto value;
  if (const json::Json *field = json::object_field(json, "op_type")) {
    value.op_type_ = field->as_string();
  }
  if (const json::Json *field = json::object_field(json, "op_domain")) {
    value.op_domain_ = field->as_string();
  }
  if (const json::Json *field = json::object_field(json, "args")) {
    for (const auto &item : field->as_array()) {
      *value.args_.Add() = PatternCallNodeArgProto::FromJson(item);
    }
  }
  if (const json::Json *field = json::object_field(json, "optional_args")) {
    for (const auto &item : field->as_array()) {
      value.optional_args_.Add(item.as_bool());
    }
  }
  return value;
}

inline json::Json PatternNodeOutputArgProto::ToJson() const {
  json::Json::Object object;
  object.emplace("call_node", call_node_.ToJson());
  if (output_arg_index_ != 0) {
    object.emplace("output_arg_index",
                   json::Json::integer(static_cast<int64_t>(output_arg_index_)));
  }
  return json::Json::object(std::move(object));
}
inline PatternNodeOutputArgProto
PatternNodeOutputArgProto::FromJson(const json::Json &json) {
  PatternNodeOutputArgProto value;
  if (const json::Json *field = json::object_field(json, "call_node")) {
    value.call_node_ = PatternCallNodeArgProto::FromJson(*field);
  }
  if (const json::Json *field = json::object_field(json, "output_arg_index")) {
    value.output_arg_index_ = static_cast<uint64_t>(field->as_int());
  }
  return value;
}

inline json::Json PatternGraphOutputProto::ToJson() const {
  json::Json::Object object;
  object.emplace("node_arg", node_arg_.ToJson());
  if (graph_output_index_) {
    object.emplace("graph_output_index",
                   json::Json::integer(static_cast<int64_t>(*graph_output_index_)));
  }
  if (graph_output_name_) {
    object.emplace("graph_output_name", json::Json::str(*graph_output_name_));
  }
  return json::Json::object(std::move(object));
}
inline PatternGraphOutputProto
PatternGraphOutputProto::FromJson(const json::Json &json) {
  PatternGraphOutputProto value;
  if (const json::Json *field = json::object_field(json, "node_arg")) {
    value.node_arg_ = PatternCallNodeArgProto::FromJson(*field);
  }
  if (const json::Json *field = json::object_field(json, "graph_output_index")) {
    value.graph_output_index_ = static_cast<uint64_t>(field->as_int());
  }
  if (const json::Json *field = json::object_field(json, "graph_output_name")) {
    value.graph_output_name_ = field->as_string();
  }
  return value;
}

inline json::Json PatternProto::ToJson() const {
  json::Json::Object object;
  if (id_) {
    object.emplace("id", json::Json::str(*id_));
  }
  if (is_root_) {
    object.emplace("is_root", json::Json::boolean(true));
  }
  switch (type_case_) {
  case kWildcard:
    object.emplace("wildcard", wildcard_.ToJson());
    break;
  case kGraphInput:
    object.emplace("graph_input", graph_input_.ToJson());
    break;
  case kConstant:
    object.emplace("constant", constant_.ToJson());
    break;
  case kCallNode:
    object.emplace("call_node", call_node_.ToJson());
    break;
  case kNodeOutputArg:
    object.emplace("node_output_arg", node_output_arg_.ToJson());
    break;
  case kGraphOutput:
    object.emplace("graph_output", graph_output_.ToJson());
    break;
  case TYPE_NOT_SET:
    break;
  }
  return json::Json::object(std::move(object));
}
inline PatternProto PatternProto::FromJson(const json::Json &json) {
  PatternProto value;
  if (const json::Json *id = json::object_field(json, "id")) {
    value.id_ = id->as_string();
  }
  if (const json::Json *is_root = json::object_field(json, "is_root")) {
    value.is_root_ = is_root->as_bool();
  }
  if (const json::Json *wildcard = json::object_field(json, "wildcard")) {
    value.type_case_ = kWildcard;
    value.wildcard_ = PatternWildcardProto::FromJson(*wildcard);
  } else if (const json::Json *graph_input =
                 json::object_field(json, "graph_input")) {
    value.type_case_ = kGraphInput;
    value.graph_input_ = PatternGraphInputProto::FromJson(*graph_input);
  } else if (const json::Json *constant = json::object_field(json, "constant")) {
    value.type_case_ = kConstant;
    value.constant_ = PatternConstantProto::FromJson(*constant);
  } else if (const json::Json *call_node =
                 json::object_field(json, "call_node")) {
    value.type_case_ = kCallNode;
    value.call_node_ = PatternCallNodeProto::FromJson(*call_node);
  } else if (const json::Json *node_output_arg =
                 json::object_field(json, "node_output_arg")) {
    value.type_case_ = kNodeOutputArg;
    value.node_output_arg_ = PatternNodeOutputArgProto::FromJson(*node_output_arg);
  } else if (const json::Json *graph_output =
                 json::object_field(json, "graph_output")) {
    value.type_case_ = kGraphOutput;
    value.graph_output_ = PatternGraphOutputProto::FromJson(*graph_output);
  }
  return value;
}

inline json::Json RootPatternProto::ToJson() const {
  json::Json::Array patterns;
  for (const auto &pattern : patterns_) {
    patterns.push_back(pattern.ToJson());
  }
  json::Json::Object object;
  if (!patterns.empty()) {
    object.emplace("patterns", json::Json::array(std::move(patterns)));
  }
  return json::Json::object(std::move(object));
}
inline RootPatternProto RootPatternProto::FromJson(const json::Json &json) {
  RootPatternProto value;
  if (const json::Json *field = json::object_field(json, "patterns")) {
    for (const auto &item : field->as_array()) {
      *value.patterns_.Add() = PatternProto::FromJson(item);
    }
  }
  return value;
}

} // namespace morphizen
