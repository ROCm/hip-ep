// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <morphizen-utils/json.hpp>

#include <cstdint>
#include <string>

namespace relu_dq {

struct ReluDqParamProto {
  std::string sample_string_;
  int32_t sample_int_ = 0;
  morphizen::ProtoList<std::string> sample_strings_;
  morphizen::ProtoList<int32_t> sample_ints_;
  std::string ep_context_file_name_;
  int32_t ep_context_file_size_ = 0;

  std::string *mutable_sample_string() { return &sample_string_; }
  void set_sample_int(int value) { sample_int_ = value; }
  void add_sample_ints(int value) { sample_ints_.Add(value); }
  void add_sample_strings(const std::string &value) { sample_strings_.Add(value); }
  void set_ep_context_file_name(std::string value) {
    ep_context_file_name_ = std::move(value);
  }
  void set_ep_context_file_size(int value) { ep_context_file_size_ = value; }
  const std::string &ep_context_file_name() const { return ep_context_file_name_; }
  int ep_context_file_size() const { return ep_context_file_size_; }

  morphizen::json::Json ToJson() const {
    morphizen::json::Json::Object object;
    if (!sample_string_.empty()) {
      object.emplace("sample_string", morphizen::json::Json::str(sample_string_));
    }
    if (sample_int_ != 0) {
      object.emplace("sample_int", morphizen::json::Json::integer(sample_int_));
    }
    if (!sample_strings_.empty()) {
      morphizen::json::Json::Array values;
      for (const auto &value : sample_strings_) {
        values.push_back(morphizen::json::Json::str(value));
      }
      object.emplace("sample_strings", morphizen::json::Json::array(std::move(values)));
    }
    if (!sample_ints_.empty()) {
      morphizen::json::Json::Array values;
      for (int value : sample_ints_) {
        values.push_back(morphizen::json::Json::integer(value));
      }
      object.emplace("sample_ints", morphizen::json::Json::array(std::move(values)));
    }
    if (!ep_context_file_name_.empty()) {
      object.emplace("ep_context_file_name",
                     morphizen::json::Json::str(ep_context_file_name_));
    }
    if (ep_context_file_size_ != 0) {
      object.emplace("ep_context_file_size",
                     morphizen::json::Json::integer(ep_context_file_size_));
    }
    return morphizen::json::Json::object(std::move(object));
  }
  static ReluDqParamProto FromJsonString(std::string_view text) {
    const morphizen::json::Json json = morphizen::json::parse(text);
    ReluDqParamProto value;
    if (const morphizen::json::Json *field =
            morphizen::json::object_field(json, "sample_string")) {
      value.sample_string_ = field->as_string();
    }
    if (const morphizen::json::Json *field =
            morphizen::json::object_field(json, "sample_int")) {
      value.sample_int_ = field->as_int32();
    }
    if (const morphizen::json::Json *field =
            morphizen::json::object_field(json, "sample_strings")) {
      for (const auto &item : field->as_array()) {
        value.sample_strings_.Add(item.as_string());
      }
    }
    if (const morphizen::json::Json *field =
            morphizen::json::object_field(json, "sample_ints")) {
      for (const auto &item : field->as_array()) {
        value.sample_ints_.Add(item.as_int32());
      }
    }
    if (const morphizen::json::Json *field =
            morphizen::json::object_field(json, "ep_context_file_name")) {
      value.ep_context_file_name_ = field->as_string();
    }
    if (const morphizen::json::Json *field =
            morphizen::json::object_field(json, "ep_context_file_size")) {
      value.ep_context_file_size_ = field->as_int32();
    }
    return value;
  }
  std::string ToJsonString() const { return morphizen::json::dump(ToJson()); }
};

} // namespace relu_dq
