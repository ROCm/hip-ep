/*
 *     The Xilinx Vitis AI Vaip in this distribution are provided under the
 * following free and permissive binary-only license, but are not provided in
 * source code form.  While the following free and permissive license is similar
 * to the BSD open source license, it is NOT the BSD open source license nor
 * other OSI-approved open source license.
 *
 *      Copyright (C) 2022 Xilinx, Inc. All rights reserved.
 *      Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights
 * reserved.
 *
 *      Redistribution and use in binary form only, without modification, is
 * permitted provided that the following conditions are met:
 *
 *      1. Redistributions must reproduce the above copyright notice, this list
 * of conditions and the following disclaimer in the documentation and/or other
 * materials provided with the distribution.
 *
 *      2. The name of Xilinx, Inc. may not be used to endorse or promote
 * products redistributed with this software without specific prior written
 * permission.
 *
 *      THIS SOFTWARE IS PROVIDED BY XILINX, INC. "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL XILINX, INC. BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 *      PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE
 */
#include "config.hpp"
#include "morphizen/config.pb.h"
#include <algorithm>
#include <set>

#include <glog/logging.h>
#ifdef ENABLE_XRT
#  include <version.h>
#endif
#ifdef _WIN32
#  pragma warning(push)
#  pragma warning(disable : 4251)
#endif
#include <google/protobuf/struct.pb.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/util/json_util.h>
#ifdef _WIN32
#  pragma warning(pop)
#endif
#include "morphizen/env_config.hpp"
#include "morphizen/util.hpp"
#include "morphizen/weak.hpp"
#include "morphizen/xclbin_file.hpp"
#include <algorithm>
#include <exception>
#include <filesystem>
#include <functional>
#include <memory>
#include <type_traits>
#include <unordered_map>
// version info
#include "version_info.hpp"
#include <vaip/vaip_ort_api.h>
#if WITH_XCOMPILER
#  include <xcompiler/xcompiler.hpp>
#endif
#include <nlohmann/json.hpp>

DEF_ENV_PARAM(DEBUG_VAIP_CONFIG, "0")
DEF_ENV_PARAM(XLNX_ONNX_EP_VERBOSE, "0")
DEF_ENV_PARAM_2(XLNX_VART_FIRMWARE, "", std::string)
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_VAIP_CONFIG) >= n)
#define LOG_VERBOSE(n)                                                         \
  LOG_IF(INFO, ENV_PARAM(XLNX_ONNX_EP_VERBOSE) >= n)                           \
      << "[XLNX_ONNX_EP_VERBOSE] "
#define LOG_VERSION_INFO(version_info)                                         \
  LOG_VERBOSE(1) << version_info.version() << ": " << version_info.commit();

namespace vaip_core {
Config::Config(const std::string& file) {
  MY_LOG(1) << "read config from : " << file;
  auto text = slurp(file.c_str());

  auto options = google::protobuf::util::JsonParseOptions();
  options.ignore_unknown_fields = true;
  auto status = google::protobuf::util::JsonStringToMessage(
      text, &config_proto_, options);
  CHECK(status.ok()) << "cannot parse config.proto: " << file << "\n" << text;
  MY_LOG(1) << "text = " << text;
}

void Config::add_version_info(ConfigProto& proto,
                              const std::string& package_name,
                              const std::string& commit_id,
                              const std::string& version_id) {
  auto version_ptr = proto.mutable_version();
  auto temp_version = version_ptr->add_version_infos();
  temp_version->set_package_name(package_name);
  temp_version->set_commit(commit_id);
  temp_version->set_version(version_id);
  return;
}

void Config::add_version_info(ConfigProto& proto) {
  using version_vec_tuple =
      std::vector<std::tuple<std::string, std::string, std::string>>;
  for (auto& info : version_vec_tuple{
#include "vaip_version_info.hpp.inc"
       }) {
    add_version_info(proto, std::get<0>(info), std::get<1>(info),
                     std::get<2>(info));
  }
}

const ConfigProto& Config::config_proto() const { return config_proto_; }

static TargetProto* get_target_proto(ConfigProto& proto,
                                     const std::string& target_name) {
  TargetProto* ret = nullptr;
  for (auto& target : *proto.mutable_targets()) {
    if (target.name() == target_name) {
      ret = &target;
      break;
    }
  }
  return ret;
}

static void remove_pass(ConfigProto& proto,
                        std::unordered_map<std::string, PassProto>& pass_map) {
  while (!proto.passes().empty()) {
    auto& pass = proto.passes(proto.passes_size() - 1);
    std::string name = pass.name();
    pass_map[name] = pass;
    proto.mutable_passes()->RemoveLast();
  }
}

static void
add_target_pass(ConfigProto& proto,
                const std::unordered_map<std::string, PassProto>& pass_map,
                const TargetProto* target_proto) {
  for (auto pass : target_proto->pass()) {
    auto iter = pass_map.find(pass);
    CHECK(iter != pass_map.end()) << "Pass not found: " << pass;
    auto new_pass = proto.add_passes();
    new_pass->CopyFrom(iter->second);
  }
}

static std::string dump_all_targets(const ConfigProto& proto) {
  std::string targets;
  for (const TargetProto& target : proto.targets()) {
    targets += std::string(target.name());
    targets += std::string(";");
  }
  return targets;
}

static void update_target_compiler_atttr(ConfigProto& proto,
                                         const TargetAttrProto& opts) {
  // update:
  // both change pass_dpu_param and all passproto in pass_vaiml_param
  for (int i = 0; i < proto.passes_size(); ++i) {
    PassProto* pass = proto.mutable_passes(i);
    if (pass->has_pass_dpu_param()) {
      auto attrs = pass->mutable_pass_dpu_param()->mutable_xcompiler_attrs();
      for (const auto& pair : opts.xcompiler_attrs()) {
        (*attrs)[pair.first] = pair.second;
      }
    }
    if (proto.mutable_passes(i)->has_pass_vaiml_param()) {
      auto pass_vaiml_param =
          proto.mutable_passes(i)->mutable_pass_vaiml_param();
      for (auto j = 0; j < pass_vaiml_param->sub_pass_size(); j++) {

        auto vaiml_sub_pass = pass_vaiml_param->mutable_sub_pass(j);
        if (vaiml_sub_pass->has_pass_dpu_param()) {
          auto attrs = vaiml_sub_pass->mutable_pass_dpu_param()
                           ->mutable_xcompiler_attrs();
          for (const auto& pair : opts.xcompiler_attrs()) {
            (*attrs)[pair.first] = pair.second;
          }
        }
      }
    }
  }
}

static void update_target_attr(ConfigProto& proto,
                               const TargetProto* target_proto) {
  auto session_option = proto.provider_options();
  if (session_option.find("target_opts") != session_option.end()) {
    std::string target_opts = session_option.at("target_opts");
    auto options = google::protobuf::util::JsonParseOptions();
    options.ignore_unknown_fields = true;
    TargetAttrProto session_option_target_opts;
    auto status = google::protobuf::util::JsonStringToMessage(
        target_opts, &session_option_target_opts, options);
    CHECK(status.ok()) << "cannot target_opts:" << target_opts;

    update_target_compiler_atttr(proto, session_option_target_opts);
  } else if (target_proto->has_target_opts()) {
    update_target_compiler_atttr(proto, target_proto->target_opts());
  }
}

static void update_xclbin(ConfigProto& proto, const TargetProto* target_proto) {
  std::string xclbin;
  auto session_option = proto.provider_options();
  if (session_option.find("xclbin") != session_option.end()) {
    xclbin = session_option.at("xclbin");
  } else if (target_proto->has_xclbin()) {
#ifdef _WIN32
    const std::filesystem::path dir("C:\\Windows\\System32\\AMD");
    auto full_path = dir / target_proto->xclbin();
    xclbin = full_path.string();
#else
    xclbin = target_proto->xclbin();
#endif
  } else if (!ENV_PARAM(XLNX_VART_FIRMWARE).empty()) {
    LOG_VERBOSE(1) << "XLNX_VART_FIRMWARE is deprecated";
    xclbin = ENV_PARAM(XLNX_VART_FIRMWARE);
  }
  if (xclbin.empty()) {
    return;
  }
  // now the session option has the real xclbin
  // so, the custom op/pass can get the correct one
  (*proto.mutable_provider_options())["xclbin"] = xclbin;
  // only for Backward Compatibility ， will be deleted
  // Note : writing back the xclbin from session option to PassDpuParam is
  // generally not recommended, but this is a special case.
  for (auto i = 0; i < proto.passes_size(); i++) {
    // todo support more pass param, such as PassDodParam
    if (proto.mutable_passes(i)->has_pass_dpu_param()) {
      proto.mutable_passes(i)->mutable_pass_dpu_param()->set_xclbin(xclbin);
    }

    if (proto.mutable_passes(i)->has_pass_vaiml_param()) {
      auto len =
          proto.mutable_passes(i)->mutable_pass_vaiml_param()->sub_pass_size();
      for (auto j = 0; j < len; j++) {

        auto sub_pass_proto = proto.mutable_passes(i)
                                  ->mutable_pass_vaiml_param()
                                  ->mutable_sub_pass(j);
        if (sub_pass_proto->has_pass_dpu_param()) {
          sub_pass_proto->mutable_pass_dpu_param()->set_xclbin(xclbin);
        }
      }
    }
  }

  std::filesystem::path xclbin_path(xclbin);
  if (std::filesystem::is_directory(xclbin_path)) {
    std::string err_msg =
        std::string{"xclbin is set to a directory: "} + xclbin;
    throw std::runtime_error(err_msg);
  }
}

static void update_hw_context_share(ConfigProto& proto,
                                    const TargetProto* target_proto) {

  auto session_option = proto.provider_options();

  // highest priority: user config via session option explicitly
  // second prioirty: the config in target proto, only work when user doesn't
  // provide 'share_context' config.
  constexpr char context_share_key[] = "share_context";

  if (session_option.contains(context_share_key))
    return;

  if (target_proto->has_share_hw_context()) {
    // share_context is of type bool.
    auto share_context = target_proto->share_hw_context();
    (*proto.mutable_provider_options())[context_share_key] =
        std::to_string(share_context);
  }
}

static void update_graph_engine_qos_priority(ConfigProto& proto,
                                             const TargetProto* target_proto) {

  auto session_option = proto.provider_options();

  // highest priority: user config via session option explicitly
  // second prioirty: the config in target proto, only work when user doesn't
  // provide 'share_context' config.
  constexpr char context_share_key[] = "priority";

  if (session_option.contains(context_share_key))
    return;

  if (target_proto->has_graph_engine_qos_priority()) {
    // share_context is of type int32.
    auto qos_priority = target_proto->graph_engine_qos_priority();
    (*proto.mutable_provider_options())[context_share_key] =
        std::to_string(qos_priority);
  }
}

void update_config_by_target(ConfigProto& proto, const MepConfigTable* mep) {
  auto target = std::string();
  auto xclbin = std::string();

  if (proto.provider_options().contains("xlnx_target")) {
    target = proto.provider_options().at("xlnx_target");
  } else if (mep != nullptr) {
    target = mep->target();
    if (mep->has_xclbin()) {
      xclbin = mep->xclbin();
    }
  }
  if (target.empty()) {
    LOG_VERBOSE(1)
        << "Target is empty, run all passes."; // old version, compatible
    return;
  }

  auto target_proto = get_target_proto(proto, target);
  CHECK(target_proto != nullptr)
      << "No valid target found: " << target
      << "Valid targets are: " << dump_all_targets(proto);
  // update xclbin (mepcofig -> target)
  if (!xclbin.empty()) {
    target_proto->set_xclbin(xclbin);
  }

  // update hw context sharing (mepconfig -> target)
  if (mep != nullptr && mep->has_share_hw_context()) {
    target_proto->set_share_hw_context(mep->share_hw_context());
  }

  if (mep != nullptr && mep->has_model_clone_threshold()) {
    (*proto.mutable_provider_options())
        ["XLNX_model_clone_external_data_threshold"] =
            std::to_string(mep->model_clone_threshold());
  }
  // For non-shell models, the default for use_old_qdq is true
  // For shell models, the default for use_old_qdq is false
  bool use_old_qdq = true;
  bool use_py3_round = false;
  if (mep) {
    use_old_qdq = false;
    if (target_proto->has_old_qdq()) {
      use_old_qdq = target_proto->old_qdq();
    }
  }
  if (target_proto->has_py3_round()) {
    bool use_py3_round = target_proto->py3_round();
    auto iter = proto.provider_options().find("xlnx_enable_py3_round");
    if (iter == proto.provider_options().end()) {
      (*proto.mutable_provider_options())["xlnx_enable_py3_round"] =
          std::to_string(use_py3_round);
    }
  }

  auto iter = proto.provider_options().find("xlnx_enable_old_qdq");
  if (iter == proto.provider_options().end()) {
    (*proto.mutable_provider_options())["xlnx_enable_old_qdq"] =
        std::to_string(use_old_qdq);
  }
  std::unordered_map<std::string, PassProto> pass_map;
  remove_pass(proto, pass_map);
  add_target_pass(proto, pass_map, target_proto);
  update_target_attr(proto, target_proto);
  update_xclbin(proto, target_proto);
  update_hw_context_share(proto, target_proto);
  update_graph_engine_qos_priority(proto, target_proto);
}

static const google::protobuf::FieldDescriptor*
FindFieldByNameOrCamelCase(const google::protobuf::Descriptor* descriptor,
                           const std::string& field_name) {
  const google::protobuf::FieldDescriptor* field =
      descriptor->FindFieldByName(field_name);
  if (!field) {
    const google::protobuf::FieldDescriptor* field_camel =
        descriptor->FindFieldByCamelcaseName(field_name);
    if (field_camel) {
      field = field_camel;
    }
  }
  return field;
}

void add_custom_field(ConfigProto& proto, const std::string& str) {
  const google::protobuf::Descriptor* descriptor = proto.GetDescriptor();
  const google::protobuf::Reflection* reflection = proto.GetReflection();
  std::set<std::string> to_be_delete;
  for (auto& el : proto.provider_options()) {
    const std::string& field_name = el.first;
    const std::string& value = el.second;

    // Find the field in the descriptor
    const google::protobuf::FieldDescriptor* field =
        FindFieldByNameOrCamelCase(descriptor, field_name);
    if (field) {
      if (field->is_repeated()) {
        continue;
      }
      // Field exists, set its value based on type
      switch (field->type()) {
      case google::protobuf::FieldDescriptor::TYPE_STRING:
        reflection->SetString(&proto, field, value);
        to_be_delete.insert(field_name);
        break;
      case google::protobuf::FieldDescriptor::TYPE_INT32: {
        long x;
        morphizen::parse_value(value, x);
        reflection->SetInt32(&proto, field, (int32_t)x);
        to_be_delete.insert(field_name);
        break;
      }
      case google::protobuf::FieldDescriptor::TYPE_BOOL: {
        bool x;
        // vitis::ai::parse_value(value, x);
        if (value == "yes" || value == "on" || value == "enable" ||
            value == "true" || value == "True" || value == "1") {
          x = true;
        } else {
          x = false;
        }
        reflection->SetBool(&proto, field, x);
        to_be_delete.insert(field_name);
        break;
      }
      default:
        // Handle other types as needed
        break;
      }
    }
  }
  auto po = proto.mutable_provider_options();
  for (auto key : to_be_delete) {
    LOG_VERBOSE(1) << "picked_out_config: " << key << " = " << po->at(key);
    po->erase(key);
  }
}

void Config::merge_config_proto(ConfigProto& config_proto,
                                const char* json_config) {
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
  add_custom_field(cache_dir_msg, json_str);
  config_proto.MergeFrom(cache_dir_msg);
}

ConfigProto Config::parse_from_string(const char* json_config) {
  auto config_proto = ConfigProto();
  if (json_config != nullptr && !std::string(json_config).empty()) {
    Config::merge_config_proto(config_proto, json_config);
  }
  return config_proto;
}
} // namespace vaip_core
