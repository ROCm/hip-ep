/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "config.hpp"
#include "morphizen/config.pb.h"
#include <algorithm>
#include <set>

#include <glog/logging.h>
#ifdef _WIN32
#pragma warning(push)
#pragma warning(disable : 4251)
#endif
#include <google/protobuf/struct.pb.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/util/json_util.h>
#ifdef _WIN32
#pragma warning(pop)
#endif
#include "morphizen/env_config.hpp"
#include "morphizen/pass_context.hpp"
#include "morphizen/util.hpp"
#include "morphizen/weak.hpp"
#include <algorithm>
#include <exception>
#include <filesystem>
#include <functional>
#include <memory>
#include <type_traits>
#include <unordered_map>
// version info
#include "version_info.hpp"
#include <morphizen/morphizen_ort_api.h>

DEF_ENV_PARAM(DEBUG_MORPHIZEN_CONFIG, "0")
DEF_ENV_PARAM(MORPHIZEN_DEBUG_TARGET_DISCOVERY, "0")
DEF_ENV_PARAM(XLNX_ONNX_EP_VERBOSE, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_MORPHIZEN_CONFIG) >= n)
#define LOG_VERBOSE(n)                                                         \
  LOG_IF(INFO, ENV_PARAM(XLNX_ONNX_EP_VERBOSE) >= n)                           \
      << "[XLNX_ONNX_EP_VERBOSE] "
#define LOG_VERSION_INFO(version_info)                                         \
  LOG_VERBOSE(1) << version_info.version() << ": " << version_info.commit();

namespace morphizen {
Config::Config(const std::string &file) {
  MY_LOG(1) << "read config from : " << file;
  auto text = slurp(file.c_str());

  auto options = google::protobuf::util::JsonParseOptions();
  options.ignore_unknown_fields = true;
  auto status = google::protobuf::util::JsonStringToMessage(
      text, &config_proto_, options);
  CHECK(status.ok()) << "cannot parse config.proto: " << file << "\n" << text;
  MY_LOG(1) << "text = " << text;
}

void Config::add_version_info(ContextProto &proto,
                              const std::string &package_name,
                              const std::string &commit_id,
                              const std::string &version_id) {
  auto version_ptr = proto.mutable_version();
  auto temp_version = version_ptr->add_version_infos();
  temp_version->set_package_name(package_name);
  temp_version->set_commit(commit_id);
  temp_version->set_version(version_id);
  return;
}

void Config::add_version_info(ContextProto &proto) {
  using version_vec_tuple =
      std::vector<std::tuple<std::string, std::string, std::string>>;
  for (auto &info : version_vec_tuple{
#include "morphizen_version_info.hpp.inc"
       }) {
    add_version_info(proto, std::get<0>(info), std::get<1>(info),
                     std::get<2>(info));
  }
}

const ConfigProto &Config::config_proto() const { return config_proto_; }

// Removed: get_target_proto() - only caller (update_config_by_target) removed
// in Issue #017 Removed: remove_pass() - obsolete after Issue #014
// (compute_effective_passes) Removed: add_target_pass() - obsolete after Issue
// #014 (compute_effective_passes)

// Removed: update_target_compiler_atttr - depended on removed pass_dpu_param
// and pass_vaiml_param fields

// Removed: update_target_attr - depended on removed target_opts field

// Removed: update_hw_context_share - depended on removed share_hw_context field

// Removed: update_graph_engine_qos_priority - depended on removed
// graph_engine_qos_priority field

void Config::merge_config_proto(ConfigProto &config_proto,
                                const char *json_config) {
  std::string json_str(json_config);
  // FIXME: This var name "cache_dir_msg" is misleading.
  ConfigProto cache_dir_msg;
  auto options = google::protobuf::util::JsonParseOptions();
  //  The approach here to processing non-standard fields
  //  is conveluted.
  options.ignore_unknown_fields = true;
  auto status = google::protobuf::util::JsonStringToMessage(

      json_str, &cache_dir_msg, options);
  CHECK(status.ok()) << "cannot parse json string:" << json_str;
  MY_LOG(2) << "json_str = " << json_str
            << " cache_dir_msg = " << cache_dir_msg.DebugString();
  config_proto.MergeFrom(cache_dir_msg);
}

ConfigProto Config::parse_from_string(const char *json_config) {
  auto config_proto = ConfigProto();
  if (json_config != nullptr && !std::string(json_config).empty()) {
    Config::merge_config_proto(config_proto, json_config);
  }
  return config_proto;
}
} // namespace morphizen
