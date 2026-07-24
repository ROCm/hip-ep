/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "./env.hpp"
#include "./wide-string.hpp"
#include <glog/logging.h>
#define ORT_API_MANUAL_INIT 1
#include <onnxruntime_cxx_api.h>
#include <unordered_set>

namespace morphizen_e2e_test {

static OrtLoggingLevel convert_log_level(const std::string &log_level) {
  if (log_level == "verbose") {
    return ORT_LOGGING_LEVEL_VERBOSE;
  } else if (log_level == "info") {
    return ORT_LOGGING_LEVEL_INFO;
  } else if (log_level == "warning") {
    return ORT_LOGGING_LEVEL_WARNING;
  } else if (log_level == "error") {
    return ORT_LOGGING_LEVEL_ERROR;
  } else if (log_level == "fatal") {
    return ORT_LOGGING_LEVEL_FATAL;
  }
  return ORT_LOGGING_LEVEL_WARNING; // Default to warning
}

E2ETestEnv::E2ETestEnv(const E2ETestEnvProto &env_proto)
    : env_proto_(env_proto) {
  LOG(INFO) << "E2ETestEnv created with proto: " << env_proto_.DebugString();
  ort_env_ =
      std::make_unique<Ort::Env>(convert_log_level(env_proto_.ort_log_level()),
                                 env_proto_.ort_log_id().c_str());

  const auto &registrations = env_proto_.registration();
  for (const auto &registration : registrations) {
    LOG(INFO) << "Registering: " << registration.name()
              << ", library: " << registration.library();
    auto status = Ort::GetApi().RegisterExecutionProviderLibrary(
        *ort_env_, registration.name().c_str(),
        ToOrtString<ORTCHAR_T>()(registration.library()).c_str());
    CHECK(status == nullptr)
        << "RegisterExecutionProviderLibrary failed: status = "
        << Ort::GetApi().GetErrorMessage(status);
  }

  // get selected devices
  auto ep_devices = ort_env_->GetEpDevices();
// on Linux, the ep_devices is empty
// error message :
// Check failed: !ep_devices.empty() No execution provider devices found. Please
// check your environment.
#ifdef _WIN32
  CHECK(!ep_devices.empty()) << "No execution provider devices found. Please "
                                "check your environment.";
#endif
  std::unordered_set<std::string> registration_names;
  registration_names.reserve(registrations.size());
  for (const auto &registration : registrations) {
    registration_names.emplace(registration.name());
  }
  std::copy_if(ep_devices.begin(), ep_devices.end(),
               std::back_inserter(selected_devices_),
               [&registration_names](const auto &device) {
                 LOG(INFO) << "Checking device: " << device.EpName()
                           << " from vendor: " << device.EpVendor();
                 return registration_names.find(device.EpName()) !=
                        registration_names.end();
               });
}

E2ETestEnv::~E2ETestEnv() {
  LOG(INFO) << "E2ETestEnv being destroyed.";
  // Now it's safe to unregister execution providers
  for (const auto &registration : env_proto_.registration()) {
    LOG(INFO) << "Unregistering: " << registration.name();
    auto status = Ort::GetApi().UnregisterExecutionProviderLibrary(
        *ort_env_, registration.name().c_str());
    CHECK(status == nullptr) << "UnregisterExecutionProvider failed: status = "
                             << Ort::GetApi().GetErrorMessage(status);
  }
  LOG(INFO) << "E2ETestEnv destroyed.";
}

std::vector<std::unique_ptr<E2ETestSessionOptions>>
E2ETestEnv::create_e2e_test_session_options() {
  auto ret = std::vector<std::unique_ptr<E2ETestSessionOptions>>();
  for (const auto &session_option_proto : env_proto_.session_options()) {
    ret.emplace_back(std::make_unique<E2ETestSessionOptions>(
        session_option_proto, *ort_env_, selected_devices_));
  }
  return std::move(ret);
}

} // namespace morphizen_e2e_test
