/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "./session-options.hpp"
#include "./env.hpp"
#include "test_environment.hpp"
#include <glog/logging.h>

namespace morphizen_e2e_test {

E2ETestSessionOptions::E2ETestSessionOptions(
    const E2ETestSessionOptionsProto& proto, Ort::Env& env,
    const std::vector<Ort::ConstEpDevice>& selected_devices)
    : proto_(proto), env_(env) {
  LOG(INFO) << "Creating E2ETestSessionOptions for proto: "
            << proto.DebugString();
  ort_session_options_ = std::make_unique<Ort::SessionOptions>();

  // add session config entries
  for (const auto& session_config : proto_.session_configs()) {
    LOG(INFO) << "Session config: " << session_config.first << " = "
              << session_config.second;
#ifdef BAZEL_CURRENT_REPOSITORY
    // In Bazel, the test CWD is read-only runfiles. Prefix relative
    // ep.context_file_path with TEST_CWD (TEST_TMPDIR) so ORT writes the
    // generated EP context model to the writable scratch directory.
    if (session_config.first == "ep.context_file_path") {
      auto ctx_path = std::filesystem::u8path(session_config.second);
      if (ctx_path.is_relative()) {
        auto abs_path = (TEST_CWD / ctx_path).u8string();
        LOG(INFO) << "Session config (abs): ep.context_file_path = "
                  << abs_path;
        ort_session_options_->AddConfigEntry(session_config.first.c_str(),
                                             abs_path.c_str());
        continue;
      }
    }
#endif
    ort_session_options_->AddConfigEntry(session_config.first.c_str(),
                                         session_config.second.c_str());
  }
  // Append execution provider
  if (proto_.has_v2_param()) {
    auto& provider_options_config = proto_.v2_param().provider_options();
    auto provider_options = std::unordered_map<std::string, std::string>(
        provider_options_config.begin(), provider_options_config.end());
    for (const auto& [key, value] : provider_options) {
      LOG(INFO) << "Provider option: " << key << " = " << value;
    }
    // Get selected devices
    CHECK(!selected_devices.empty())
        << "No devices found for the registered providers in the environment.";
    for (const auto& device : selected_devices) {
      LOG(INFO) << "Selected EP device: " << device.EpName()
                << " from vendor: " << device.EpVendor();
    }
    ort_session_options_->AppendExecutionProvider_V2(env_, selected_devices,
                                                     provider_options);
  } else {
    LOG(INFO) << "Defaulting to CPU execution provider.";
  }
}

std::vector<std::unique_ptr<E2ETestSession>>
E2ETestSessionOptions::create_e2e_test_sessions() {
  auto ret = std::vector<std::unique_ptr<E2ETestSession>>();

  for (const auto& session_proto : proto_.session()) {
    ret.emplace_back(std::make_unique<E2ETestSession>(
        env_, *ort_session_options_, session_proto));
  }

  return ret;
}

} // namespace morphizen_e2e_test
