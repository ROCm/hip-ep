/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#pragma once

#include <morphizen-utils/json.hpp>

#include <cstdint>
#include <string>

namespace mlir_metadata {

struct Input {
  std::string name_;
  int32_t rank_ = 0;
  int32_t elem_type_ = 0;
  morphizen::ProtoList<int64_t> shape_;

  const std::string &name() const { return name_; }
  int rank() const { return rank_; }
  int elem_type() const { return elem_type_; }
  const morphizen::ProtoList<int64_t> &shape() const { return shape_; }
  void set_name(std::string value) { name_ = std::move(value); }
  void set_rank(int value) { rank_ = value; }
  void set_elem_type(int value) { elem_type_ = value; }
  void add_shape(int64_t value) { shape_.Add(value); }

  morphizen::json::Json ToJson() const;
  static Input FromJson(const morphizen::json::Json &json);
};

struct Output {
  std::string name_;
  int32_t rank_ = 0;
  int32_t elem_type_ = 0;
  morphizen::ProtoList<int64_t> shape_;

  const std::string &name() const { return name_; }
  int rank() const { return rank_; }
  int elem_type() const { return elem_type_; }
  const morphizen::ProtoList<int64_t> &shape() const { return shape_; }
  void set_name(std::string value) { name_ = std::move(value); }
  void set_rank(int value) { rank_ = value; }
  void set_elem_type(int value) { elem_type_ = value; }
  void add_shape(int64_t value) { shape_.Add(value); }

  morphizen::json::Json ToJson() const;
  static Output FromJson(const morphizen::json::Json &json);
};

struct Metadata {
  std::string artifact_filename_;
  morphizen::ProtoList<Output> outputs_;
  morphizen::ProtoList<Input> inputs_;
  std::string artifact_format_;

  const std::string &artifact_filename() const { return artifact_filename_; }
  const std::string &artifact_format() const { return artifact_format_; }
  void set_artifact_filename(std::string value) {
    artifact_filename_ = std::move(value);
  }
  void set_artifact_format(std::string value) {
    artifact_format_ = std::move(value);
  }
  const morphizen::ProtoList<Output> &outputs() const { return outputs_; }
  morphizen::ProtoList<Output> *mutable_outputs() { return &outputs_; }
  Output *add_outputs() { return outputs_.Add(); }
  const morphizen::ProtoList<Input> &inputs() const { return inputs_; }
  Input *add_inputs() { return inputs_.Add(); }

  morphizen::json::Json ToJson() const;
  static Metadata FromJson(const morphizen::json::Json &json);
  static Metadata FromJsonString(std::string_view text) {
    return FromJson(morphizen::json::parse(text));
  }
  std::string ToJsonString() const { return morphizen::json::dump(ToJson()); }
};

inline morphizen::json::Json
tensor_json(const std::string &name, int rank, int elem_type,
            const morphizen::ProtoList<int64_t> &shape) {
  morphizen::json::Json::Object object;
  if (!name.empty()) {
    object.emplace("name", morphizen::json::Json::str(name));
  }
  if (rank != 0) {
    object.emplace("rank", morphizen::json::Json::integer(rank));
  }
  if (elem_type != 0) {
    object.emplace("elem_type", morphizen::json::Json::integer(elem_type));
  }
  if (!shape.empty()) {
    morphizen::json::Json::Array dims;
    for (int64_t dim : shape) {
      dims.push_back(morphizen::json::Json::integer(dim));
    }
    object.emplace("shape", morphizen::json::Json::array(std::move(dims)));
  }
  return morphizen::json::Json::object(std::move(object));
}

inline void read_tensor(const morphizen::json::Json &json, std::string &name,
                        int32_t &rank, int32_t &elem_type,
                        morphizen::ProtoList<int64_t> &shape) {
  if (const morphizen::json::Json *field =
          morphizen::json::object_field(json, "name")) {
    name = field->as_string();
  }
  if (const morphizen::json::Json *field =
          morphizen::json::object_field(json, "rank")) {
    rank = field->as_int32();
  }
  if (const morphizen::json::Json *field =
          morphizen::json::object_field(json, "elem_type")) {
    elem_type = field->as_int32();
  }
  if (const morphizen::json::Json *field =
          morphizen::json::object_field(json, "shape")) {
    for (const auto &dim : field->as_array()) {
      shape.Add(dim.as_int());
    }
  }
}

inline morphizen::json::Json Input::ToJson() const {
  return tensor_json(name_, rank_, elem_type_, shape_);
}
inline Input Input::FromJson(const morphizen::json::Json &json) {
  Input value;
  read_tensor(json, value.name_, value.rank_, value.elem_type_, value.shape_);
  return value;
}
inline morphizen::json::Json Output::ToJson() const {
  return tensor_json(name_, rank_, elem_type_, shape_);
}
inline Output Output::FromJson(const morphizen::json::Json &json) {
  Output value;
  read_tensor(json, value.name_, value.rank_, value.elem_type_, value.shape_);
  return value;
}

inline morphizen::json::Json Metadata::ToJson() const {
  morphizen::json::Json::Object object;
  if (!artifact_filename_.empty()) {
    object.emplace("artifact_filename",
                   morphizen::json::Json::str(artifact_filename_));
  }
  if (!artifact_format_.empty()) {
    object.emplace("artifact_format",
                   morphizen::json::Json::str(artifact_format_));
  }
  if (!inputs_.empty()) {
    morphizen::json::Json::Array inputs;
    for (const auto &input : inputs_) {
      inputs.push_back(input.ToJson());
    }
    object.emplace("inputs", morphizen::json::Json::array(std::move(inputs)));
  }
  if (!outputs_.empty()) {
    morphizen::json::Json::Array outputs;
    for (const auto &output : outputs_) {
      outputs.push_back(output.ToJson());
    }
    object.emplace("outputs", morphizen::json::Json::array(std::move(outputs)));
  }
  return morphizen::json::Json::object(std::move(object));
}
inline Metadata Metadata::FromJson(const morphizen::json::Json &json) {
  Metadata value;
  if (const morphizen::json::Json *field =
          morphizen::json::object_field(json, "artifact_filename")) {
    value.artifact_filename_ = field->as_string();
  }
  if (const morphizen::json::Json *field =
          morphizen::json::object_field(json, "artifact_format")) {
    value.artifact_format_ = field->as_string();
  }
  if (const morphizen::json::Json *field =
          morphizen::json::object_field(json, "inputs")) {
    for (const auto &item : field->as_array()) {
      *value.inputs_.Add() = Input::FromJson(item);
    }
  }
  if (const morphizen::json::Json *field =
          morphizen::json::object_field(json, "outputs")) {
    for (const auto &item : field->as_array()) {
      *value.outputs_.Add() = Output::FromJson(item);
    }
  }
  return value;
}

} // namespace mlir_metadata
