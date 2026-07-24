/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "./session.hpp"

#include "./session-run.hpp"
#include "./wide-string.hpp"
#include "test_environment.hpp"
#include <fstream>
#include <glog/logging.h>
#include <iostream>

namespace morphizen_e2e_test {

static std::vector<uint8_t> ReadBinaryFile(const std::string &file_path) {
  std::ifstream file(file_path, std::ios::binary | std::ios::ate);
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::vector<uint8_t> buffer(size);
  if (file.read(reinterpret_cast<char *>(buffer.data()), size)) {
    return buffer;
  } else {
    throw std::runtime_error("Failed to read model file");
  }
}
static std::unique_ptr<Ort::Session>
create_session(const std::filesystem::path &model_path, Ort::Env &env,
               Ort::SessionOptions &session_options, bool use_memory_model) {

  std::unique_ptr<Ort::Session> p_session;
  auto create_session_start_time = std::chrono::steady_clock::now();
  try {
    if (use_memory_model) {
      auto model_data = ReadBinaryFile(model_path.string());
      p_session = std::make_unique<Ort::Session>(
          env, model_data.data(), model_data.size(), session_options);
    } else {
      p_session = std::make_unique<Ort::Session>(
          env, PathToString<ORTCHAR_T>()(model_path).c_str(), session_options);
    }
  } catch (Ort::Exception &e) {
    std::cout << "Catched Ort Exception: " << e.what() << std::endl;
    exit(1);
  }
  auto create_session_end_time = std::chrono::steady_clock::now();
  std::cout << " ONNXRuntime session create :  "
            << std::chrono::duration_cast<std::chrono::microseconds>(
                   create_session_end_time - create_session_start_time)
                   .count()
            << " us" << std::endl;
  return std::move(p_session);
}

static std::optional<std::string>
get_session_config_option(const Ort::SessionOptions &session_options,
                          const std::string &key) {
  if (session_options.HasConfigEntry(key.c_str())) {
    return session_options.GetConfigEntry(key.c_str());
  }
  return std::nullopt;
}

static std::filesystem::path
get_ctx_model_path(const Ort::SessionOptions &session_options,
                   const std::filesystem::path &model_path) {
  auto ctx_path_name =
      model_path.stem().string() + "_ctx" + model_path.extension().string();
  auto ctx_path = model_path.parent_path() / ctx_path_name;
  auto cache_context_file_path =
      get_session_config_option(session_options, "ep.context_file_path");
  if (cache_context_file_path.has_value()) {
    ctx_path = std::filesystem::u8path(cache_context_file_path.value());
  }
  return ctx_path;
}

static void del_ctx_model(const std::filesystem::path &model_path) {
  try {
    std::filesystem::remove(model_path);
  } catch (std::exception &e) {
    std::cerr << "Exception: " << e.what() << std::endl;
  }
}

E2ETestSession::E2ETestSession(Ort::Env &env,
                               Ort::SessionOptions &session_options,
                               const E2ETestSessionProto &session_proto)
    : env_(env), session_options_(session_options),
      session_proto_(session_proto) {

  // Resolve model path.
  // In Bazel: source models live in runfiles → use Rlocation.
  // Generated EP context models are not in runfiles → fall back to TEST_CWD.
  // In CMake: model_path is already relative to TEST_CWD (CMAKE_BINARY_DIR).
#ifdef BAZEL_CURRENT_REPOSITORY
  std::filesystem::path model_path;
  {
    std::string err;
    auto rf = bazel::tools::cpp::runfiles::Runfiles::CreateForTest(
        BAZEL_CURRENT_REPOSITORY, &err);
    auto resolved =
        rf ? rf->Rlocation("_main/unit-test/data/" + session_proto.model_path())
           : std::string{};
    if (!resolved.empty() &&
        std::filesystem::exists(std::filesystem::u8path(resolved))) {
      model_path = std::filesystem::u8path(resolved);
    } else {
      // Generated model (e.g. EP context .onnx): falls back to TEST_CWD.
      model_path =
          TEST_CWD / std::filesystem::u8path(session_proto.model_path());
    }
  }
#else
  auto model_path = std::filesystem::u8path(session_proto.model_path());
#endif
  CHECK(std::filesystem::exists(model_path))
      << "Model path does not exist: " << model_path;
  LOG(INFO) << "Creating ORT session with model: " << model_path;

  for (auto i = 0; i < session_proto.session_count(); ++i) {

    auto is_context_enable =
        get_session_config_option(session_options_, "ep.context_enable")
            .value_or("0") == "1";
    if (is_context_enable) {
      auto ctx_model_path = get_ctx_model_path(session_options_, model_path);
      LOG(INFO) << "Context model path: " << ctx_model_path;
      del_ctx_model(ctx_model_path);
    }
    // Create a new session for each count
    ort_sessions_.push_back(create_session(model_path, env_, session_options_,
                                           session_proto.use_memory_model()));
  }
}

void E2ETestSession::run() {
  for (auto &session : ort_sessions_) {
    run_session(*session, session_proto_.run());
  }
}

} // namespace morphizen_e2e_test
