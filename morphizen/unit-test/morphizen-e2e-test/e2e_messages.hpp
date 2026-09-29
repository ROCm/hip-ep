// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <morphizen-utils/json.hpp>

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <variant>

struct CPUEPParamProto {};
struct MorphiZenEPParamProto {
  std::map<std::string, std::string> provider_options_;
  const std::map<std::string, std::string> &provider_options() const {
    return provider_options_;
  }
};
struct V2ParamProto {
  std::map<std::string, std::string> provider_options_;
  const std::map<std::string, std::string> &provider_options() const {
    return provider_options_;
  }
};

struct E2ETestSessionRunProto {
  int64_t run_count_ = 0;
  std::optional<int64_t> batch_number_;
  morphizen::ProtoList<std::string> golden_input_;
  morphizen::ProtoList<std::string> golden_output_;
  bool has_batch_number() const { return batch_number_.has_value(); }
  int64_t batch_number() const { return batch_number_ ? *batch_number_ : 0; }
  std::string DebugString() const { return {}; }
};

struct E2ETestSessionProto {
  std::string model_path_;
  bool use_memory_model_ = false;
  int64_t session_count_ = 0;
  E2ETestSessionRunProto run_;
  const std::string &model_path() const { return model_path_; }
  bool use_memory_model() const { return use_memory_model_; }
  int64_t session_count() const { return session_count_; }
  const E2ETestSessionRunProto &run() const { return run_; }
  std::string DebugString() const { return {}; }
};

struct EPRegistrationProto {
  std::string name_;
  std::string library_;
  const std::string &name() const { return name_; }
  const std::string &library() const { return library_; }
  std::string DebugString() const { return name_ + " " + library_; }
};

struct E2ETestSessionOptionsProto {
  std::map<std::string, std::string> session_configs_;
  std::variant<std::monostate, CPUEPParamProto, MorphiZenEPParamProto,
               V2ParamProto>
      ep_param_;
  morphizen::ProtoList<E2ETestSessionProto> session_;

  const std::map<std::string, std::string> &session_configs() const {
    return session_configs_;
  }
  bool has_v2_param() const {
    return std::holds_alternative<V2ParamProto>(ep_param_);
  }
  const V2ParamProto &v2_param() const {
    return std::get<V2ParamProto>(ep_param_);
  }
  const morphizen::ProtoList<E2ETestSessionProto> &session() const {
    return session_;
  }
  std::string DebugString() const { return {}; }
};

struct E2ETestEnvProto {
  std::string ort_log_level_;
  std::string ort_log_id_;
  morphizen::ProtoList<EPRegistrationProto> registration_;
  morphizen::ProtoList<E2ETestSessionOptionsProto> session_options_;
  const std::string &ort_log_level() const { return ort_log_level_; }
  const std::string &ort_log_id() const { return ort_log_id_; }
  const morphizen::ProtoList<EPRegistrationProto> &registration() const {
    return registration_;
  }
  const morphizen::ProtoList<E2ETestSessionOptionsProto> &
  session_options() const {
    return session_options_;
  }
  std::string DebugString() const { return {}; }
};

struct E2ETestConfigProto {
  std::string name_;
  E2ETestEnvProto env_;
  const std::string &name() const { return name_; }
  const E2ETestEnvProto &env() const { return env_; }
  std::string DebugString() const { return name_; }
};

struct MorphizenE2ETestsProto {
  morphizen::ProtoList<E2ETestConfigProto> test_configs_;
  const morphizen::ProtoList<E2ETestConfigProto> &test_configs() const {
    return test_configs_;
  }

  static MorphizenE2ETestsProto FromJsonString(std::string_view text) {
    const morphizen::json::Json root = morphizen::json::parse(text);
    MorphizenE2ETestsProto value;
    const morphizen::json::Json *configs =
        morphizen::json::object_field(root, "test_configs");
    if (configs == nullptr) {
      return value;
    }
    for (const auto &item : configs->as_array()) {
      E2ETestConfigProto config;
      if (const morphizen::json::Json *name =
              morphizen::json::object_field(item, "name")) {
        config.name_ = name->as_string();
      }
      if (const morphizen::json::Json *env =
              morphizen::json::object_field(item, "env")) {
        if (const morphizen::json::Json *level =
                morphizen::json::object_field(*env, "ort_log_level")) {
          config.env_.ort_log_level_ = level->as_string();
        }
        if (const morphizen::json::Json *id =
                morphizen::json::object_field(*env, "ort_log_id")) {
          config.env_.ort_log_id_ = id->as_string();
        }
        if (const morphizen::json::Json *regs =
                morphizen::json::object_field(*env, "registration")) {
          for (const auto &reg_json : regs->as_array()) {
            EPRegistrationProto reg;
            if (const morphizen::json::Json *field =
                    morphizen::json::object_field(reg_json, "name")) {
              reg.name_ = field->as_string();
            }
            if (const morphizen::json::Json *field =
                    morphizen::json::object_field(reg_json, "library")) {
              reg.library_ = field->as_string();
            }
            *config.env_.registration_.Add() = std::move(reg);
          }
        }
        if (const morphizen::json::Json *options =
                morphizen::json::object_field(*env, "session_options")) {
          for (const auto &opt_json : options->as_array()) {
            *config.env_.session_options_.Add() =
                read_session_options(opt_json);
          }
        }
      }
      *value.test_configs_.Add() = std::move(config);
    }
    return value;
  }

private:
  static std::map<std::string, std::string>
  read_map(const morphizen::json::Json &json) {
    std::map<std::string, std::string> values;
    if (!json.is_object()) {
      return values;
    }
    for (const auto &entry : json.as_object()) {
      values.emplace(entry.first, entry.second.as_string());
    }
    return values;
  }
  static E2ETestSessionRunProto read_run(const morphizen::json::Json &json) {
    E2ETestSessionRunProto run;
    if (const morphizen::json::Json *field =
            morphizen::json::object_field(json, "run_count")) {
      run.run_count_ = field->as_int();
    }
    if (const morphizen::json::Json *field =
            morphizen::json::object_field(json, "batch_number")) {
      run.batch_number_ = field->as_int();
    }
    return run;
  }
  static E2ETestSessionProto read_session(const morphizen::json::Json &json) {
    E2ETestSessionProto session;
    if (const morphizen::json::Json *field =
            morphizen::json::object_field(json, "model_path")) {
      session.model_path_ = field->as_string();
    }
    if (const morphizen::json::Json *field =
            morphizen::json::object_field(json, "use_memory_model")) {
      session.use_memory_model_ = field->as_bool();
    }
    if (const morphizen::json::Json *field =
            morphizen::json::object_field(json, "session_count")) {
      session.session_count_ = field->as_int();
    }
    if (const morphizen::json::Json *field =
            morphizen::json::object_field(json, "run")) {
      session.run_ = read_run(*field);
    }
    return session;
  }
  static E2ETestSessionOptionsProto
  read_session_options(const morphizen::json::Json &json) {
    E2ETestSessionOptionsProto options;
    if (const morphizen::json::Json *field =
            morphizen::json::object_field(json, "session_configs")) {
      options.session_configs_ = read_map(*field);
    }
    if (const morphizen::json::Json *field =
            morphizen::json::object_field(json, "cpu_ep_param")) {
      options.ep_param_ = CPUEPParamProto{};
    } else if (const morphizen::json::Json *field =
                   morphizen::json::object_field(json, "morphizen_ep_param")) {
      MorphiZenEPParamProto param;
      if (const morphizen::json::Json *opts =
              morphizen::json::object_field(*field, "provider_options")) {
        param.provider_options_ = read_map(*opts);
      }
      options.ep_param_ = std::move(param);
    } else if (const morphizen::json::Json *field =
                   morphizen::json::object_field(json, "v2_param")) {
      V2ParamProto param;
      if (const morphizen::json::Json *opts =
              morphizen::json::object_field(*field, "provider_options")) {
        param.provider_options_ = read_map(*opts);
      }
      options.ep_param_ = std::move(param);
    }
    if (const morphizen::json::Json *field =
            morphizen::json::object_field(json, "session")) {
      for (const auto &item : field->as_array()) {
        *options.session_.Add() = read_session(item);
      }
    }
    return options;
  }
};
