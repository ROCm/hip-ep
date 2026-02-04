/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#pragma once
#include <fstream>
#include <morphizen/my_ort.h>
#include <string>

#ifdef _WIN32
#  pragma warning(push)
#  pragma warning(disable : 4251)
#  pragma warning(disable : 4275)
#  pragma warning(disable : 4946) // reinterpret_cast between related classes in
                                  // protobuf
#endif
#include "morphizen/config.pb.h"
#include "morphizen/pass_context.pb.h"
#ifdef _WIN32
#  pragma warning(pop)
#endif

namespace morphizen {
class PassContext;
class Config {
public:
  MORPHIZEN_DLL_SPEC
  static ConfigProto parse_from_string(const char* string);
  static void merge_config_proto(ConfigProto& config_proto,
                                 const char* json_config);
  static void add_version_info(ContextProto& context_proto);
  static void add_version_info(ContextProto& context_proto,
                               const std::string& package_name,
                               const std::string& commit_id,
                               const std::string& version_id);

public:
  explicit Config() = default;
  explicit Config(const std::string& file);
  Config(const Config&) = delete;
  Config(Config&&) = delete;

public:
  ~Config() = default;

public:
  const ConfigProto& config_proto() const;

private:
  ConfigProto config_proto_;
};
} // namespace morphizen
