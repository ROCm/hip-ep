// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

#pragma once

// In-process replacements for MorphiZen's generated protobuf messages.
// Field names in JSON are the proto field names (snake_case). Unknown object
// keys are ignored. Accessors follow the protobuf C++ names the call sites
// already use.

#include <morphizen-utils/json.hpp>

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>

namespace morphizen {

inline json::Json object_or_empty(const json::Json *field) {
  if (field == nullptr) {
    return json::Json::object();
  }
  if (!field->is_object()) {
    throw json::ParseError("expected JSON object");
  }
  return *field;
}

inline void put_string(json::Json::Object &object, const char *key,
                       const std::string &value) {
  if (!value.empty()) {
    object.emplace(key, json::Json::str(value));
  }
}
inline void put_bool(json::Json::Object &object, const char *key, bool value) {
  if (value) {
    object.emplace(key, json::Json::boolean(true));
  }
}
inline void put_int(json::Json::Object &object, const char *key, int64_t value) {
  if (value != 0) {
    object.emplace(key, json::Json::integer(value));
  }
}

inline std::string read_string(const json::Json *field) {
  return field == nullptr ? std::string() : field->as_string();
}
inline bool read_bool(const json::Json *field) {
  return field != nullptr && field->as_bool();
}
inline int read_int32(const json::Json *field) {
  return field == nullptr ? 0 : field->as_int32();
}
inline int64_t read_int64(const json::Json *field) {
  return field == nullptr ? 0 : field->as_int();
}
inline float read_float(const json::Json *field) {
  return field == nullptr ? 0.f : static_cast<float>(field->as_float());
}

inline ProtoList<std::string> read_string_list(const json::Json *field) {
  ProtoList<std::string> values;
  if (field == nullptr) {
    return values;
  }
  for (const auto &item : field->as_array()) {
    values.Add(item.as_string());
  }
  return values;
}
inline ProtoList<int32_t> read_int32_list(const json::Json *field) {
  ProtoList<int32_t> values;
  if (field == nullptr) {
    return values;
  }
  for (const auto &item : field->as_array()) {
    values.Add(item.as_int32());
  }
  return values;
}
inline ProtoList<int64_t> read_int64_list(const json::Json *field) {
  ProtoList<int64_t> values;
  if (field == nullptr) {
    return values;
  }
  for (const auto &item : field->as_array()) {
    values.Add(item.as_int());
  }
  return values;
}
inline ProtoList<bool> read_bool_list(const json::Json *field) {
  ProtoList<bool> values;
  if (field == nullptr) {
    return values;
  }
  for (const auto &item : field->as_array()) {
    values.Add(item.as_bool());
  }
  return values;
}

inline json::Json string_list_json(const ProtoList<std::string> &values) {
  json::Json::Array array;
  array.reserve(static_cast<size_t>(values.size()));
  for (const auto &value : values) {
    array.push_back(json::Json::str(value));
  }
  return json::Json::array(std::move(array));
}
inline json::Json int32_list_json(const ProtoList<int32_t> &values) {
  json::Json::Array array;
  array.reserve(static_cast<size_t>(values.size()));
  for (int32_t value : values) {
    array.push_back(json::Json::integer(value));
  }
  return json::Json::array(std::move(array));
}
inline json::Json int64_list_json(const ProtoList<int64_t> &values) {
  json::Json::Array array;
  array.reserve(static_cast<size_t>(values.size()));
  for (int64_t value : values) {
    array.push_back(json::Json::integer(value));
  }
  return json::Json::array(std::move(array));
}
inline json::Json bool_list_json(const ProtoList<bool> &values) {
  json::Json::Array array;
  array.reserve(static_cast<size_t>(values.size()));
  for (bool value : values) {
    array.push_back(json::Json::boolean(value));
  }
  return json::Json::array(std::move(array));
}

inline void put_list(json::Json::Object &object, const char *key,
                     const json::Json &list) {
  if (!list.as_array().empty()) {
    object.emplace(key, list);
  }
}

inline std::map<std::string, std::string>
read_string_map(const json::Json *field) {
  std::map<std::string, std::string> values;
  if (field == nullptr) {
    return values;
  }
  for (const auto &entry : field->as_object()) {
    values.emplace(entry.first, entry.second.as_string());
  }
  return values;
}
inline std::map<std::string, int32_t> read_int_map(const json::Json *field) {
  std::map<std::string, int32_t> values;
  if (field == nullptr) {
    return values;
  }
  for (const auto &entry : field->as_object()) {
    values.emplace(entry.first, entry.second.as_int32());
  }
  return values;
}
inline json::Json string_map_json(const std::map<std::string, std::string> &values) {
  json::Json::Object object;
  for (const auto &entry : values) {
    object.emplace(entry.first, json::Json::str(entry.second));
  }
  return json::Json::object(std::move(object));
}
inline json::Json int_map_json(const std::map<std::string, int32_t> &values) {
  json::Json::Object object;
  for (const auto &entry : values) {
    object.emplace(entry.first, json::Json::integer(entry.second));
  }
  return json::Json::object(std::move(object));
}
inline void put_object(json::Json::Object &object, const char *key,
                       const json::Json &value) {
  if (value.is_object() && !value.as_object().empty()) {
    object.emplace(key, value);
  }
}

struct VersionInfoProto {
  std::string package_name_;
  std::string commit_;
  std::string version_;
  const std::string &package_name() const { return package_name_; }
  const std::string &commit() const { return commit_; }
  const std::string &version() const { return version_; }
  void set_package_name(std::string value) { package_name_ = std::move(value); }
  void set_commit(std::string value) { commit_ = std::move(value); }
  void set_version(std::string value) { version_ = std::move(value); }

  json::Json ToJson() const {
    json::Json::Object object;
    put_string(object, "package_name", package_name_);
    put_string(object, "commit", commit_);
    put_string(object, "version", version_);
    return json::Json::object(std::move(object));
  }
  static VersionInfoProto FromJson(const json::Json &json) {
    VersionInfoProto value;
    value.package_name_ = read_string(json::object_field(json, "package_name"));
    value.commit_ = read_string(json::object_field(json, "commit"));
    value.version_ = read_string(json::object_field(json, "version"));
    return value;
  }
  std::string DebugString() const { return json::dump(ToJson(), true); }
};

struct AllVersionInfoProto {
  ProtoList<VersionInfoProto> version_infos_;
  const ProtoList<VersionInfoProto> &version_infos() const {
    return version_infos_;
  }
  ProtoList<VersionInfoProto> *mutable_version_infos() { return &version_infos_; }
  VersionInfoProto *add_version_infos() { return version_infos_.Add(); }

  json::Json ToJson() const {
    json::Json::Array array;
    for (const auto &info : version_infos_) {
      array.push_back(info.ToJson());
    }
    json::Json::Object object;
    put_list(object, "version_infos", json::Json::array(std::move(array)));
    return json::Json::object(std::move(object));
  }
  static AllVersionInfoProto FromJson(const json::Json &json) {
    AllVersionInfoProto value;
    if (const json::Json *field = json::object_field(json, "version_infos")) {
      for (const auto &item : field->as_array()) {
        *value.version_infos_.Add() = VersionInfoProto::FromJson(item);
      }
    }
    return value;
  }
  std::string DebugString() const { return json::dump(ToJson(), true); }
};

struct PassProto {
  std::string name_;
  std::string plugin_;
  ProtoList<std::string> args_;
  bool enable_log_ = false;
  int32_t log_verbosity_ = 0;
  bool enable_gc_ = false;
  bool disabled_ = false;
  std::optional<json::Json> pass_generic_param_;

  const std::string &name() const { return name_; }
  const std::string &plugin() const { return plugin_; }
  bool enable_log() const { return enable_log_; }
  int log_verbosity() const { return log_verbosity_; }
  bool enable_gc() const { return enable_gc_; }
  bool disabled() const { return disabled_; }
  void set_name(std::string value) { name_ = std::move(value); }
  void set_plugin(std::string value) { plugin_ = std::move(value); }
  void set_enable_log(bool value) { enable_log_ = value; }
  void set_log_verbosity(int value) { log_verbosity_ = value; }
  void set_enable_gc(bool value) { enable_gc_ = value; }
  void set_disabled(bool value) { disabled_ = value; }
  const ProtoList<std::string> &args() const { return args_; }
  void add_args(const std::string &value) { args_.Add(value); }
  bool has_pass_generic_param() const { return pass_generic_param_.has_value(); }
  const json::Json &pass_generic_param() const {
    if (!pass_generic_param_) {
      static const json::Json empty = json::Json::object();
      return empty;
    }
    return *pass_generic_param_;
  }
  json::Json *mutable_pass_generic_param() {
    if (!pass_generic_param_) {
      pass_generic_param_ = json::Json::object();
    }
    return &(*pass_generic_param_);
  }

  json::Json ToJson() const {
    json::Json::Object object;
    put_string(object, "name", name_);
    put_string(object, "plugin", plugin_);
    put_list(object, "args", string_list_json(args_));
    put_bool(object, "enable_log", enable_log_);
    put_int(object, "log_verbosity", log_verbosity_);
    put_bool(object, "enable_gc", enable_gc_);
    put_bool(object, "disabled", disabled_);
    if (pass_generic_param_) {
      object.emplace("pass_generic_param", *pass_generic_param_);
    }
    return json::Json::object(std::move(object));
  }
  static PassProto FromJson(const json::Json &json) {
    PassProto value;
    value.name_ = read_string(json::object_field(json, "name"));
    value.plugin_ = read_string(json::object_field(json, "plugin"));
    value.args_ = read_string_list(json::object_field(json, "args"));
    value.enable_log_ = read_bool(json::object_field(json, "enable_log"));
    value.log_verbosity_ = read_int32(json::object_field(json, "log_verbosity"));
    value.enable_gc_ = read_bool(json::object_field(json, "enable_gc"));
    value.disabled_ = read_bool(json::object_field(json, "disabled"));
    if (const json::Json *field = json::object_field(json, "pass_generic_param")) {
      value.pass_generic_param_ = *field;
    }
    return value;
  }
  std::string DebugString() const { return json::dump(ToJson(), true); }
};

struct TargetProto {
  std::string name_;
  ProtoList<std::string> pass_;
  std::map<std::string, std::string> provider_options_;

  const std::string &name() const { return name_; }
  void set_name(std::string value) { name_ = std::move(value); }
  const ProtoList<std::string> &pass() const { return pass_; }
  ProtoList<std::string> *mutable_pass() { return &pass_; }
  const std::map<std::string, std::string> &provider_options() const {
    return provider_options_;
  }
  std::map<std::string, std::string> *mutable_provider_options() {
    return &provider_options_;
  }

  json::Json ToJson() const {
    json::Json::Object object;
    put_string(object, "name", name_);
    put_list(object, "pass", string_list_json(pass_));
    put_object(object, "provider_options", string_map_json(provider_options_));
    return json::Json::object(std::move(object));
  }
  static TargetProto FromJson(const json::Json &json) {
    TargetProto value;
    value.name_ = read_string(json::object_field(json, "name"));
    value.pass_ = read_string_list(json::object_field(json, "pass"));
    value.provider_options_ =
        read_string_map(json::object_field(json, "provider_options"));
    return value;
  }
  std::string DebugString() const { return json::dump(ToJson(), true); }
};

struct ConfigProto {
  ProtoList<PassProto> passes_;
  std::map<std::string, std::string> provider_options_;
  ProtoList<TargetProto> targets_;
  std::string target_;

  const ProtoList<PassProto> &passes() const { return passes_; }
  ProtoList<PassProto> *mutable_passes() { return &passes_; }
  const std::map<std::string, std::string> &provider_options() const {
    return provider_options_;
  }
  std::map<std::string, std::string> *mutable_provider_options() {
    return &provider_options_;
  }
  const ProtoList<TargetProto> &targets() const { return targets_; }
  const std::string &target() const { return target_; }
  void set_target(std::string value) { target_ = std::move(value); }

  // Protobuf MergeFrom: overwrite scalars, append repeated fields, merge maps.
  void MergeFrom(const ConfigProto &other) {
    for (const auto &pass : other.passes_) {
      passes_.Add(pass);
    }
    for (const auto &target : other.targets_) {
      targets_.Add(target);
    }
    for (const auto &entry : other.provider_options_) {
      provider_options_[entry.first] = entry.second;
    }
    target_ = other.target_;
  }
  void CopyFrom(const ConfigProto &other) { *this = other; }
  void Swap(ConfigProto *other) { std::swap(*this, *other); }

  json::Json ToJson() const {
    json::Json::Array passes;
    for (const auto &pass : passes_) {
      passes.push_back(pass.ToJson());
    }
    json::Json::Array targets;
    for (const auto &target : targets_) {
      targets.push_back(target.ToJson());
    }
    json::Json::Object object;
    put_list(object, "passes", json::Json::array(std::move(passes)));
    put_object(object, "provider_options", string_map_json(provider_options_));
    put_list(object, "targets", json::Json::array(std::move(targets)));
    put_string(object, "target", target_);
    return json::Json::object(std::move(object));
  }
  static ConfigProto FromJson(const json::Json &json) {
    ConfigProto value;
    if (const json::Json *field = json::object_field(json, "passes")) {
      for (const auto &item : field->as_array()) {
        *value.passes_.Add() = PassProto::FromJson(item);
      }
    }
    value.provider_options_ =
        read_string_map(json::object_field(json, "provider_options"));
    if (const json::Json *field = json::object_field(json, "targets")) {
      for (const auto &item : field->as_array()) {
        *value.targets_.Add() = TargetProto::FromJson(item);
      }
    }
    value.target_ = read_string(json::object_field(json, "target"));
    return value;
  }
  static ConfigProto FromJsonString(std::string_view text) {
    return FromJson(json::parse(text));
  }
  std::string DebugString() const { return json::dump(ToJson(), true); }
};

struct MetaDefProto {
  std::string id_;
  ProtoList<std::string> inputs_;
  ProtoList<std::string> outputs_;
  ProtoList<std::string> nodes_;
  ProtoList<std::string> constant_initializers_;
  std::string device_;
  std::map<std::string, std::string> generic_param_;
  bool fallback_cpu_ = false;
  json::Json param_ = json::Json::object();
  ProtoList<int32_t> input_argument_indice_;
  ProtoList<int32_t> output_argument_indice_;

  const std::string &id() const { return id_; }
  void set_id(std::string value) { id_ = std::move(value); }
  const std::string &device() const { return device_; }
  void set_device(std::string value) { device_ = std::move(value); }
  bool fallback_cpu() const { return fallback_cpu_; }
  void set_fallback_cpu(bool value) { fallback_cpu_ = value; }
  const ProtoList<std::string> &inputs() const { return inputs_; }
  ProtoList<std::string> *mutable_inputs() { return &inputs_; }
  int inputs_size() const { return inputs_.size(); }
  void add_inputs(const std::string &value) { inputs_.Add(value); }
  const std::string &inputs(int index) const { return inputs_[index]; }
  const ProtoList<std::string> &outputs() const { return outputs_; }
  ProtoList<std::string> *mutable_outputs() { return &outputs_; }
  int outputs_size() const { return outputs_.size(); }
  void add_outputs(const std::string &value) { outputs_.Add(value); }
  const std::string &outputs(int index) const { return outputs_[index]; }
  const ProtoList<std::string> &nodes() const { return nodes_; }
  ProtoList<std::string> *mutable_nodes() { return &nodes_; }
  int nodes_size() const { return nodes_.size(); }
  void add_nodes(const std::string &value) { nodes_.Add(value); }
  const ProtoList<std::string> &constant_initializers() const {
    return constant_initializers_;
  }
  ProtoList<std::string> *mutable_constant_initializers() {
    return &constant_initializers_;
  }
  int constant_initializers_size() const { return constant_initializers_.size(); }
  void add_constant_initializers(const std::string &value) {
    constant_initializers_.Add(value);
  }
  const std::map<std::string, std::string> &generic_param() const {
    return generic_param_;
  }
  std::map<std::string, std::string> *mutable_generic_param() {
    return &generic_param_;
  }
  const json::Json &param() const { return param_; }
  json::Json *mutable_param() { return &param_; }
  const ProtoList<int32_t> &input_argument_indice() const {
    return input_argument_indice_;
  }
  ProtoList<int32_t> *mutable_input_argument_indice() {
    return &input_argument_indice_;
  }
  int32_t input_argument_indice(int index) const {
    return input_argument_indice_[index];
  }
  const ProtoList<int32_t> &output_argument_indice() const {
    return output_argument_indice_;
  }
  ProtoList<int32_t> *mutable_output_argument_indice() {
    return &output_argument_indice_;
  }
  int32_t output_argument_indice(int index) const {
    return output_argument_indice_[index];
  }

  json::Json ToJson() const {
    json::Json::Object object;
    put_string(object, "id", id_);
    put_list(object, "inputs", string_list_json(inputs_));
    put_list(object, "outputs", string_list_json(outputs_));
    put_list(object, "nodes", string_list_json(nodes_));
    put_list(object, "constant_initializers",
             string_list_json(constant_initializers_));
    put_string(object, "device", device_);
    put_object(object, "generic_param", string_map_json(generic_param_));
    put_bool(object, "fallback_CPU", fallback_cpu_);
    if (param_.is_object() && !param_.as_object().empty()) {
      object.emplace("param", param_);
    }
    put_list(object, "input_argument_indice",
             int32_list_json(input_argument_indice_));
    put_list(object, "output_argument_indice",
             int32_list_json(output_argument_indice_));
    return json::Json::object(std::move(object));
  }
  static MetaDefProto FromJson(const json::Json &json) {
    MetaDefProto value;
    value.id_ = read_string(json::object_field(json, "id"));
    value.inputs_ = read_string_list(json::object_field(json, "inputs"));
    value.outputs_ = read_string_list(json::object_field(json, "outputs"));
    value.nodes_ = read_string_list(json::object_field(json, "nodes"));
    value.constant_initializers_ =
        read_string_list(json::object_field(json, "constant_initializers"));
    value.device_ = read_string(json::object_field(json, "device"));
    value.generic_param_ =
        read_string_map(json::object_field(json, "generic_param"));
    value.fallback_cpu_ = read_bool(json::object_field(json, "fallback_CPU"));
    value.param_ = object_or_empty(json::object_field(json, "param"));
    value.input_argument_indice_ =
        read_int32_list(json::object_field(json, "input_argument_indice"));
    value.output_argument_indice_ =
        read_int32_list(json::object_field(json, "output_argument_indice"));
    return value;
  }
  std::string DebugString() const { return json::dump(ToJson(), true); }
};

struct ShapeInfoProto {
  std::string name_;
  ProtoList<int64_t> shape_;
  bool is_scalar_ = false;
  bool is_unknown_ = false;
  json::Json ToJson() const {
    json::Json::Object object;
    put_string(object, "name", name_);
    put_list(object, "shape", int64_list_json(shape_));
    put_bool(object, "is_scalar", is_scalar_);
    put_bool(object, "is_unknown", is_unknown_);
    return json::Json::object(std::move(object));
  }
  static ShapeInfoProto FromJson(const json::Json &json) {
    ShapeInfoProto value;
    value.name_ = read_string(json::object_field(json, "name"));
    value.shape_ = read_int64_list(json::object_field(json, "shape"));
    value.is_scalar_ = read_bool(json::object_field(json, "is_scalar"));
    value.is_unknown_ = read_bool(json::object_field(json, "is_unknown"));
    return value;
  }
};

struct SubgraphInfoProto {
  std::string device_;
  int32_t count_ = 0;
  json::Json ToJson() const {
    json::Json::Object object;
    put_string(object, "device", device_);
    put_int(object, "count", count_);
    return json::Json::object(std::move(object));
  }
  static SubgraphInfoProto FromJson(const json::Json &json) {
    SubgraphInfoProto value;
    value.device_ = read_string(json::object_field(json, "device"));
    value.count_ = read_int32(json::object_field(json, "count"));
    return value;
  }
};

struct NodeStatProto {
  ProtoList<std::string> output_;
  ProtoList<std::string> input_;
  std::string op_domain_;
  std::string op_type_;
  std::string comment_;
  std::string device_;
  json::Json ToJson() const {
    json::Json::Object object;
    put_list(object, "output", string_list_json(output_));
    put_list(object, "input", string_list_json(input_));
    put_string(object, "op_domain", op_domain_);
    put_string(object, "op_type", op_type_);
    put_string(object, "comment", comment_);
    put_string(object, "device", device_);
    return json::Json::object(std::move(object));
  }
  static NodeStatProto FromJson(const json::Json &json) {
    NodeStatProto value;
    value.output_ = read_string_list(json::object_field(json, "output"));
    value.input_ = read_string_list(json::object_field(json, "input"));
    value.op_domain_ = read_string(json::object_field(json, "op_domain"));
    value.op_type_ = read_string(json::object_field(json, "op_type"));
    value.comment_ = read_string(json::object_field(json, "comment"));
    value.device_ = read_string(json::object_field(json, "device"));
    return value;
  }
};

struct DeviceStatProto {
  std::string name_;
  int32_t node_num_ = 0;
  ProtoList<std::string> supported_op_type_;
  json::Json ToJson() const {
    json::Json::Object object;
    put_string(object, "name", name_);
    put_int(object, "node_num", node_num_);
    put_list(object, "supported_op_type", string_list_json(supported_op_type_));
    return json::Json::object(std::move(object));
  }
  static DeviceStatProto FromJson(const json::Json &json) {
    DeviceStatProto value;
    value.name_ = read_string(json::object_field(json, "name"));
    value.node_num_ = read_int32(json::object_field(json, "node_num"));
    value.supported_op_type_ =
        read_string_list(json::object_field(json, "supported_op_type"));
    return value;
  }
};

struct StatProto {
  ProtoList<DeviceStatProto> device_stat_;
  ProtoList<NodeStatProto> node_stat_;
  ProtoList<ShapeInfoProto> shape_info_;
  ProtoList<SubgraphInfoProto> subgraph_stat_;
  json::Json ToJson() const {
    json::Json::Object object;
    auto dump_list = [](auto const &list) {
      json::Json::Array array;
      for (const auto &item : list) {
        array.push_back(item.ToJson());
      }
      return json::Json::array(std::move(array));
    };
    put_list(object, "device_stat", dump_list(device_stat_));
    put_list(object, "node_stat", dump_list(node_stat_));
    put_list(object, "shape_info", dump_list(shape_info_));
    put_list(object, "subgraph_stat", dump_list(subgraph_stat_));
    return json::Json::object(std::move(object));
  }
  static StatProto FromJson(const json::Json &json) {
    StatProto value;
    auto read = [&](const char *key, auto &list, auto decode) {
      if (const json::Json *field = json::object_field(json, key)) {
        for (const auto &item : field->as_array()) {
          *list.Add() = decode(item);
        }
      }
    };
    read("device_stat", value.device_stat_, DeviceStatProto::FromJson);
    read("node_stat", value.node_stat_, NodeStatProto::FromJson);
    read("shape_info", value.shape_info_, ShapeInfoProto::FromJson);
    read("subgraph_stat", value.subgraph_stat_, SubgraphInfoProto::FromJson);
    return value;
  }
};

struct AnchorPointTransposeOpAttr {
  ProtoList<int64_t> order_;
  const ProtoList<int64_t> &order() const { return order_; }
  void add_order(int64_t value) { order_.Add(value); }
  json::Json ToJson() const {
    json::Json::Object object;
    put_list(object, "order", int64_list_json(order_));
    return json::Json::object(std::move(object));
  }
  static AnchorPointTransposeOpAttr FromJson(const json::Json &json) {
    AnchorPointTransposeOpAttr value;
    value.order_ = read_int64_list(json::object_field(json, "order"));
    return value;
  }
};

struct AnchorPointPadOpAttr {
  ProtoList<int64_t> paddings_;
  const ProtoList<int64_t> &paddings() const { return paddings_; }
  json::Json ToJson() const {
    json::Json::Object object;
    put_list(object, "paddings", int64_list_json(paddings_));
    return json::Json::object(std::move(object));
  }
  static AnchorPointPadOpAttr FromJson(const json::Json &json) {
    AnchorPointPadOpAttr value;
    value.paddings_ = read_int64_list(json::object_field(json, "paddings"));
    return value;
  }
};

struct AnchorPointFixAttr {
  int64_t fix_point_ = 0;
  int64_t fix_point() const { return fix_point_; }
  void set_fix_point(int64_t value) { fix_point_ = value; }
  json::Json ToJson() const {
    json::Json::Object object;
    put_int(object, "fix_point", fix_point_);
    return json::Json::object(std::move(object));
  }
  static AnchorPointFixAttr FromJson(const json::Json &json) {
    AnchorPointFixAttr value;
    value.fix_point_ = read_int64(json::object_field(json, "fix_point"));
    return value;
  }
};

struct AnchorPointQdqAttr {
  float scale_ = 0.f;
  int64_t zero_point_ = 0;
  float scale() const { return scale_; }
  int64_t zero_point() const { return zero_point_; }
  void set_scale(float value) { scale_ = value; }
  void set_zero_point(int64_t value) { zero_point_ = value; }
  json::Json ToJson() const {
    json::Json::Object object;
    if (scale_ != 0.f) {
      object.emplace("scale", json::Json::number(scale_));
    }
    put_int(object, "zero_point", zero_point_);
    return json::Json::object(std::move(object));
  }
  static AnchorPointQdqAttr FromJson(const json::Json &json) {
    AnchorPointQdqAttr value;
    value.scale_ = read_float(json::object_field(json, "scale"));
    value.zero_point_ = read_int64(json::object_field(json, "zero_point"));
    return value;
  }
};

struct AnchorPointAttributeProto {
  enum class Kind { None, Unknown, Transpose, Pad, Fix, Qdq };
  Kind kind_ = Kind::None;
  std::string unknown_attr_;
  AnchorPointTransposeOpAttr transpose_attr_;
  AnchorPointPadOpAttr pad_attr_;
  AnchorPointFixAttr fix_attr_;
  AnchorPointQdqAttr qdq_attr_;

  bool has_unknown_attr() const { return kind_ == Kind::Unknown; }
  bool has_transpose_attr() const { return kind_ == Kind::Transpose; }
  bool has_pad_attr() const { return kind_ == Kind::Pad; }
  bool has_fix_attr() const { return kind_ == Kind::Fix; }
  bool has_qdq_attr() const { return kind_ == Kind::Qdq; }
  const std::string &unknown_attr() const { return unknown_attr_; }
  const AnchorPointTransposeOpAttr &transpose_attr() const {
    return transpose_attr_;
  }
  AnchorPointTransposeOpAttr *mutable_transpose_attr() {
    kind_ = Kind::Transpose;
    return &transpose_attr_;
  }
  AnchorPointPadOpAttr *mutable_pad_attr() {
    kind_ = Kind::Pad;
    return &pad_attr_;
  }
  const AnchorPointFixAttr &fix_attr() const { return fix_attr_; }
  AnchorPointFixAttr *mutable_fix_attr() {
    kind_ = Kind::Fix;
    return &fix_attr_;
  }
  AnchorPointQdqAttr *mutable_qdq_attr() {
    kind_ = Kind::Qdq;
    return &qdq_attr_;
  }

  json::Json ToJson() const {
    json::Json::Object object;
    switch (kind_) {
    case Kind::Unknown:
      object.emplace("unknown_attr", json::Json::str(unknown_attr_));
      break;
    case Kind::Transpose:
      object.emplace("transpose_attr", transpose_attr_.ToJson());
      break;
    case Kind::Pad:
      object.emplace("pad_attr", pad_attr_.ToJson());
      break;
    case Kind::Fix:
      object.emplace("fix_attr", fix_attr_.ToJson());
      break;
    case Kind::Qdq:
      object.emplace("qdq_attr", qdq_attr_.ToJson());
      break;
    case Kind::None:
      break;
    }
    return json::Json::object(std::move(object));
  }
  static AnchorPointAttributeProto FromJson(const json::Json &json) {
    AnchorPointAttributeProto value;
    if (const json::Json *unknown_attr =
            json::object_field(json, "unknown_attr")) {
      value.kind_ = Kind::Unknown;
      value.unknown_attr_ = unknown_attr->as_string();
    } else if (const json::Json *transpose_attr =
                   json::object_field(json, "transpose_attr")) {
      value.kind_ = Kind::Transpose;
      value.transpose_attr_ = AnchorPointTransposeOpAttr::FromJson(*transpose_attr);
    } else if (const json::Json *pad_attr =
                   json::object_field(json, "pad_attr")) {
      value.kind_ = Kind::Pad;
      value.pad_attr_ = AnchorPointPadOpAttr::FromJson(*pad_attr);
    } else if (const json::Json *fix_attr =
                   json::object_field(json, "fix_attr")) {
      value.kind_ = Kind::Fix;
      value.fix_attr_ = AnchorPointFixAttr::FromJson(*fix_attr);
    } else if (const json::Json *qdq_attr =
                   json::object_field(json, "qdq_attr")) {
      value.kind_ = Kind::Qdq;
      value.qdq_attr_ = AnchorPointQdqAttr::FromJson(*qdq_attr);
    }
    return value;
  }
};

struct AnchorPointProto {
  AnchorPointProto() = default;
  AnchorPointProto(const AnchorPointProto &other);
  AnchorPointProto &operator=(const AnchorPointProto &other);
  AnchorPointProto(AnchorPointProto &&) noexcept = default;
  AnchorPointProto &operator=(AnchorPointProto &&) noexcept = default;
  ~AnchorPointProto();

  std::string name_;
  std::string op_type_;
  std::string pass_;
  enum class Operand { None, Origin, Next };
  Operand operand_ = Operand::None;
  std::string origin_node_;
  std::unique_ptr<AnchorPointProto> next_;
  AnchorPointAttributeProto attribute_;

  const std::string &name() const { return name_; }
  const std::string &op_type() const { return op_type_; }
  const std::string &pass() const { return pass_; }
  void set_name(std::string value) { name_ = std::move(value); }
  void set_op_type(std::string value) { op_type_ = std::move(value); }
  void set_pass(std::string value) { pass_ = std::move(value); }
  bool has_origin_node() const { return operand_ == Operand::Origin; }
  bool has_next() const { return operand_ == Operand::Next; }
  const std::string &origin_node() const { return origin_node_; }
  void set_origin_node(std::string value) {
    operand_ = Operand::Origin;
    origin_node_ = std::move(value);
    next_.reset();
  }
  const AnchorPointProto &next() const {
    if (!next_) {
      static const AnchorPointProto empty;
      return empty;
    }
    return *next_;
  }
  AnchorPointProto *mutable_next() {
    if (!next_) {
      next_ = std::make_unique<AnchorPointProto>();
    }
    operand_ = Operand::Next;
    origin_node_.clear();
    return next_.get();
  }
  void clear_next() {
    next_.reset();
    if (operand_ == Operand::Next) {
      operand_ = Operand::None;
    }
  }
  const AnchorPointAttributeProto &attribute() const { return attribute_; }
  AnchorPointAttributeProto *mutable_attribute() { return &attribute_; }

  json::Json ToJson() const;
  static AnchorPointProto FromJson(const json::Json &json);
  static AnchorPointProto FromJsonString(std::string_view text) {
    return FromJson(json::parse(text));
  }
  std::string DebugString() const { return json::dump(ToJson(), true); }
};

inline AnchorPointProto::AnchorPointProto(const AnchorPointProto &other)
    : name_(other.name_), op_type_(other.op_type_), pass_(other.pass_),
      operand_(other.operand_), origin_node_(other.origin_node_),
      next_(other.next_ ? std::make_unique<AnchorPointProto>(*other.next_)
                        : nullptr),
      attribute_(other.attribute_) {}

inline AnchorPointProto &
AnchorPointProto::operator=(const AnchorPointProto &other) {
  if (this != &other) {
    AnchorPointProto copy(other);
    *this = std::move(copy);
  }
  return *this;
}

inline AnchorPointProto::~AnchorPointProto() = default;

inline json::Json AnchorPointProto::ToJson() const {
  json::Json::Object object;
  put_string(object, "name", name_);
  put_string(object, "op_type", op_type_);
  put_string(object, "pass", pass_);
  if (operand_ == Operand::Origin) {
    object.emplace("origin_node", json::Json::str(origin_node_));
  } else if (operand_ == Operand::Next && next_) {
    object.emplace("next", next_->ToJson());
  }
  json::Json attribute = attribute_.ToJson();
  if (!attribute.as_object().empty()) {
    object.emplace("attribute", std::move(attribute));
  }
  return json::Json::object(std::move(object));
}

inline AnchorPointProto AnchorPointProto::FromJson(const json::Json &json) {
  AnchorPointProto value;
  value.name_ = read_string(json::object_field(json, "name"));
  value.op_type_ = read_string(json::object_field(json, "op_type"));
  value.pass_ = read_string(json::object_field(json, "pass"));
  if (const json::Json *origin_node = json::object_field(json, "origin_node")) {
    value.set_origin_node(origin_node->as_string());
  } else if (const json::Json *next = json::object_field(json, "next")) {
    *value.mutable_next() = FromJson(*next);
  }
  if (const json::Json *field = json::object_field(json, "attribute")) {
    value.attribute_ = AnchorPointAttributeProto::FromJson(*field);
  }
  return value;
}

struct SubgraphProto {
  std::string op_name_;
  MetaDefProto metadef_;
  const std::string &op_name() const { return op_name_; }
  const MetaDefProto &metadef() const { return metadef_; }
  json::Json ToJson() const {
    json::Json::Object object;
    put_string(object, "op_name", op_name_);
    json::Json metadef = metadef_.ToJson();
    if (!metadef.as_object().empty()) {
      object.emplace("metadef", std::move(metadef));
    }
    return json::Json::object(std::move(object));
  }
  static SubgraphProto FromJson(const json::Json &json) {
    SubgraphProto value;
    value.op_name_ = read_string(json::object_field(json, "op_name"));
    if (const json::Json *field = json::object_field(json, "metadef")) {
      value.metadef_ = MetaDefProto::FromJson(*field);
    }
    return value;
  }
};

struct MemUsageProto {
  int64_t peak_memory_in_bytes_ = 0;
  std::string peak_memory_;
  int64_t current_memory_in_bytes_ = 0;
  std::string current_memory_;
  int64_t peak_memory_in_bytes() const { return peak_memory_in_bytes_; }
  int64_t current_memory_in_bytes() const { return current_memory_in_bytes_; }
  const std::string &peak_memory() const { return peak_memory_; }
  const std::string &current_memory() const { return current_memory_; }
  void set_peak_memory_in_bytes(int64_t value) { peak_memory_in_bytes_ = value; }
  void set_current_memory_in_bytes(int64_t value) {
    current_memory_in_bytes_ = value;
  }
  void set_peak_memory(std::string value) { peak_memory_ = std::move(value); }
  void set_current_memory(std::string value) {
    current_memory_ = std::move(value);
  }
  json::Json ToJson() const {
    json::Json::Object object;
    put_int(object, "peak_memory_in_bytes", peak_memory_in_bytes_);
    put_string(object, "peak_memory", peak_memory_);
    put_int(object, "current_memory_in_bytes", current_memory_in_bytes_);
    put_string(object, "current_memory", current_memory_);
    return json::Json::object(std::move(object));
  }
  static MemUsageProto FromJson(const json::Json &json) {
    MemUsageProto value;
    value.peak_memory_in_bytes_ =
        read_int64(json::object_field(json, "peak_memory_in_bytes"));
    value.peak_memory_ = read_string(json::object_field(json, "peak_memory"));
    value.current_memory_in_bytes_ =
        read_int64(json::object_field(json, "current_memory_in_bytes"));
    value.current_memory_ =
        read_string(json::object_field(json, "current_memory"));
    return value;
  }
};

struct MemDumpProto {
  MemUsageProto process_totals_;
  const MemUsageProto &process_totals() const { return process_totals_; }
  MemUsageProto *mutable_process_totals() { return &process_totals_; }
  json::Json ToJson() const {
    json::Json::Object object;
    json::Json totals = process_totals_.ToJson();
    if (!totals.as_object().empty()) {
      object.emplace("process_totals", std::move(totals));
    }
    return json::Json::object(std::move(object));
  }
  static MemDumpProto FromJson(const json::Json &json) {
    MemDumpProto value;
    if (const json::Json *field = json::object_field(json, "process_totals")) {
      value.process_totals_ = MemUsageProto::FromJson(*field);
    }
    return value;
  }
};

struct EventArgProto {
  MemDumpProto dumps_;
  MemUsageProto mem_usage_;
  const MemDumpProto &dumps() const { return dumps_; }
  MemDumpProto *mutable_dumps() { return &dumps_; }
  const MemUsageProto &mem_usage() const { return mem_usage_; }
  MemUsageProto *mutable_mem_usage() { return &mem_usage_; }
  json::Json ToJson() const {
    json::Json::Object object;
    json::Json dumps = dumps_.ToJson();
    json::Json mem = mem_usage_.ToJson();
    if (!dumps.as_object().empty()) {
      object.emplace("dumps", std::move(dumps));
    }
    if (!mem.as_object().empty()) {
      object.emplace("mem_usage", std::move(mem));
    }
    return json::Json::object(std::move(object));
  }
  static EventArgProto FromJson(const json::Json &json) {
    EventArgProto value;
    if (const json::Json *field = json::object_field(json, "dumps")) {
      value.dumps_ = MemDumpProto::FromJson(*field);
    }
    if (const json::Json *field = json::object_field(json, "mem_usage")) {
      value.mem_usage_ = MemUsageProto::FromJson(*field);
    }
    return value;
  }
};

struct EventProto {
  std::string id_;
  std::string name_;
  ProtoList<std::string> cat_;
  std::string ph_;
  int64_t ts_ = 0;
  int64_t pid_ = 0;
  int64_t tid_ = 0;
  EventArgProto args_;
  std::string cname_;
  int64_t dur_ = 0;
  const std::string &id() const { return id_; }
  const std::string &name() const { return name_; }
  const std::string &ph() const { return ph_; }
  void set_id(std::string value) { id_ = std::move(value); }
  void set_name(std::string value) { name_ = std::move(value); }
  void set_ph(std::string value) { ph_ = std::move(value); }
  void set_ts(int64_t value) { ts_ = value; }
  void set_pid(int64_t value) { pid_ = value; }
  void set_tid(int64_t value) { tid_ = value; }
  void set_dur(int64_t value) { dur_ = value; }
  const EventArgProto &args() const { return args_; }
  EventArgProto *mutable_args() { return &args_; }
  json::Json ToJson() const {
    json::Json::Object object;
    put_string(object, "id", id_);
    put_string(object, "name", name_);
    put_list(object, "cat", string_list_json(cat_));
    put_string(object, "ph", ph_);
    put_int(object, "ts", ts_);
    put_int(object, "pid", pid_);
    put_int(object, "tid", tid_);
    json::Json args = args_.ToJson();
    if (!args.as_object().empty()) {
      object.emplace("args", std::move(args));
    }
    put_string(object, "cname", cname_);
    put_int(object, "dur", dur_);
    return json::Json::object(std::move(object));
  }
  static EventProto FromJson(const json::Json &json) {
    EventProto value;
    value.id_ = read_string(json::object_field(json, "id"));
    value.name_ = read_string(json::object_field(json, "name"));
    value.cat_ = read_string_list(json::object_field(json, "cat"));
    value.ph_ = read_string(json::object_field(json, "ph"));
    value.ts_ = read_int64(json::object_field(json, "ts"));
    value.pid_ = read_int64(json::object_field(json, "pid"));
    value.tid_ = read_int64(json::object_field(json, "tid"));
    if (const json::Json *field = json::object_field(json, "args")) {
      value.args_ = EventArgProto::FromJson(*field);
    }
    value.cname_ = read_string(json::object_field(json, "cname"));
    value.dur_ = read_int64(json::object_field(json, "dur"));
    return value;
  }
};

struct CPUUsageProto {
  float avg_cpu_util_ = 0.f;
  float mem_peak_working_set_size_ = 0.f;
  float avg_cpu_util() const { return avg_cpu_util_; }
  float mem_peak_working_set_size() const { return mem_peak_working_set_size_; }
  void set_avg_cpu_util(float value) { avg_cpu_util_ = value; }
  void set_mem_peak_working_set_size(float value) {
    mem_peak_working_set_size_ = value;
  }
  json::Json ToJson() const {
    json::Json::Object object;
    if (avg_cpu_util_ != 0.f) {
      object.emplace("avg_cpu_util", json::Json::number(avg_cpu_util_));
    }
    if (mem_peak_working_set_size_ != 0.f) {
      object.emplace("mem_peak_working_set_size",
                     json::Json::number(mem_peak_working_set_size_));
    }
    return json::Json::object(std::move(object));
  }
  static CPUUsageProto FromJson(const json::Json &json) {
    CPUUsageProto value;
    value.avg_cpu_util_ = read_float(json::object_field(json, "avg_cpu_util"));
    value.mem_peak_working_set_size_ =
        read_float(json::object_field(json, "mem_peak_working_set_size"));
    return value;
  }
};

struct ContextProto {
  ProtoList<MetaDefProto> meta_def_;
  std::map<std::string, AnchorPointProto> origin_nodes_;
  std::map<std::string, int32_t> device_subgraph_count_;
  ProtoList<std::string> stacks_;
  ProtoList<EventProto> events_;
  ProtoList<CPUUsageProto> cpu_usage_;
  std::map<std::string, SubgraphProto> subgraph_metadefs_;
  std::string cache_key_;
  AllVersionInfoProto version_;

  const ProtoList<MetaDefProto> &meta_def() const { return meta_def_; }
  ProtoList<MetaDefProto> *mutable_meta_def() { return &meta_def_; }
  MetaDefProto *mutable_meta_def(int index) { return &meta_def_[index]; }
  int meta_def_size() const { return meta_def_.size(); }
  const std::map<std::string, AnchorPointProto> &origin_nodes() const {
    return origin_nodes_;
  }
  std::map<std::string, AnchorPointProto> *mutable_origin_nodes() {
    return &origin_nodes_;
  }
  std::map<std::string, int32_t> *mutable_device_subgraph_count() {
    return &device_subgraph_count_;
  }
  void add_stacks(const std::string &value) { stacks_.Add(value); }
  const ProtoList<EventProto> &events() const { return events_; }
  ProtoList<EventProto> *mutable_events() { return &events_; }
  void clear_cpu_usage() { cpu_usage_.Clear(); }
  CPUUsageProto *add_cpu_usage() { return cpu_usage_.Add(); }
  const std::string &cache_key() const { return cache_key_; }
  void set_cache_key(std::string value) { cache_key_ = std::move(value); }
  std::string *mutable_cache_key() { return &cache_key_; }
  const AllVersionInfoProto &version() const { return version_; }
  AllVersionInfoProto *mutable_version() { return &version_; }

  void CopyFrom(const ContextProto &other) { *this = other; }
  void Swap(ContextProto *other) { std::swap(*this, *other); }

  json::Json ToJson() const {
    json::Json::Array meta_defs;
    for (const auto &meta_def : meta_def_) {
      meta_defs.push_back(meta_def.ToJson());
    }
    json::Json::Object origins;
    for (const auto &entry : origin_nodes_) {
      origins.emplace(entry.first, entry.second.ToJson());
    }
    json::Json::Array events;
    for (const auto &event : events_) {
      events.push_back(event.ToJson());
    }
    json::Json::Array cpu;
    for (const auto &usage : cpu_usage_) {
      cpu.push_back(usage.ToJson());
    }
    json::Json::Object subgraphs;
    for (const auto &entry : subgraph_metadefs_) {
      subgraphs.emplace(entry.first, entry.second.ToJson());
    }
    json::Json::Object object;
    put_list(object, "meta_def", json::Json::array(std::move(meta_defs)));
    put_object(object, "origin_nodes", json::Json::object(std::move(origins)));
    put_object(object, "device_subgraph_count",
               int_map_json(device_subgraph_count_));
    put_list(object, "stacks", string_list_json(stacks_));
    put_list(object, "events", json::Json::array(std::move(events)));
    put_list(object, "cpu_usage", json::Json::array(std::move(cpu)));
    put_object(object, "subgraph_metadefs",
               json::Json::object(std::move(subgraphs)));
    put_string(object, "cache_key", cache_key_);
    json::Json version = version_.ToJson();
    if (!version.as_object().empty()) {
      object.emplace("version", std::move(version));
    }
    return json::Json::object(std::move(object));
  }
  static ContextProto FromJson(const json::Json &json) {
    ContextProto value;
    if (const json::Json *field = json::object_field(json, "meta_def")) {
      for (const auto &item : field->as_array()) {
        *value.meta_def_.Add() = MetaDefProto::FromJson(item);
      }
    }
    if (const json::Json *field = json::object_field(json, "origin_nodes")) {
      for (const auto &entry : field->as_object()) {
        value.origin_nodes_.emplace(entry.first,
                                    AnchorPointProto::FromJson(entry.second));
      }
    }
    value.device_subgraph_count_ =
        read_int_map(json::object_field(json, "device_subgraph_count"));
    value.stacks_ = read_string_list(json::object_field(json, "stacks"));
    if (const json::Json *field = json::object_field(json, "events")) {
      for (const auto &item : field->as_array()) {
        *value.events_.Add() = EventProto::FromJson(item);
      }
    }
    if (const json::Json *field = json::object_field(json, "cpu_usage")) {
      for (const auto &item : field->as_array()) {
        *value.cpu_usage_.Add() = CPUUsageProto::FromJson(item);
      }
    }
    if (const json::Json *field = json::object_field(json, "subgraph_metadefs")) {
      for (const auto &entry : field->as_object()) {
        value.subgraph_metadefs_.emplace(entry.first,
                                         SubgraphProto::FromJson(entry.second));
      }
    }
    value.cache_key_ = read_string(json::object_field(json, "cache_key"));
    if (const json::Json *field = json::object_field(json, "version")) {
      value.version_ = AllVersionInfoProto::FromJson(*field);
    }
    return value;
  }
  static ContextProto FromJsonString(std::string_view text) {
    return FromJson(json::parse(text));
  }
  std::string DebugString() const { return json::dump(ToJson(), true); }
};

struct VersionProto {
  int32_t major_ = 0;
  int32_t minor_ = 0;
  int32_t patch_ = 0;
  int major() const { return major_; }
  int minor() const { return minor_; }
  int patch() const { return patch_; }
  void set_major(int value) { major_ = value; }
  void set_minor(int value) { minor_ = value; }
  void set_patch(int value) { patch_ = value; }
  json::Json ToJson() const {
    json::Json::Object object;
    put_int(object, "major", major_);
    put_int(object, "minor", minor_);
    put_int(object, "patch", patch_);
    return json::Json::object(std::move(object));
  }
  static VersionProto FromJson(const json::Json &json) {
    VersionProto value;
    value.major_ = read_int32(json::object_field(json, "major"));
    value.minor_ = read_int32(json::object_field(json, "minor"));
    value.patch_ = read_int32(json::object_field(json, "patch"));
    return value;
  }
};

struct ModelCompatibilityProto {
  VersionProto version_;
  std::optional<bool> base64_encoding_;
  std::map<std::string, std::string> backend_compatibility_;

  VersionProto *mutable_version() { return &version_; }
  const VersionProto &version() const { return version_; }
  void set_base64_encoding(bool value) { base64_encoding_ = value; }
  bool has_base64_encoding() const { return base64_encoding_.has_value(); }
  bool base64_encoding() const {
    return base64_encoding_ ? *base64_encoding_ : false;
  }
  std::map<std::string, std::string> *mutable_backend_compatibility() {
    return &backend_compatibility_;
  }
  const std::map<std::string, std::string> &backend_compatibility() const {
    return backend_compatibility_;
  }

  json::Json ToJson() const {
    json::Json::Object object;
    json::Json version = version_.ToJson();
    if (!version.as_object().empty()) {
      object.emplace("version", std::move(version));
    }
    if (base64_encoding_) {
      object.emplace("base64_encoding", json::Json::boolean(*base64_encoding_));
    }
    put_object(object, "backend_compatibility",
               string_map_json(backend_compatibility_));
    return json::Json::object(std::move(object));
  }
  static ModelCompatibilityProto FromJson(const json::Json &json) {
    ModelCompatibilityProto value;
    if (const json::Json *field = json::object_field(json, "version")) {
      value.version_ = VersionProto::FromJson(*field);
    }
    if (const json::Json *field = json::object_field(json, "base64_encoding")) {
      value.base64_encoding_ = field->as_bool();
    }
    value.backend_compatibility_ =
        read_string_map(json::object_field(json, "backend_compatibility"));
    return value;
  }
  static ModelCompatibilityProto FromJsonString(std::string_view text) {
    return FromJson(json::parse(text));
  }
  std::string DebugString() const { return json::dump(ToJson(), true); }
};

} // namespace morphizen
