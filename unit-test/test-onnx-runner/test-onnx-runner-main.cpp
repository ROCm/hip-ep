/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "./config.hpp"
#include "./create-session.hpp"
#include "./run-session.hpp"
#include "./wide-string.hpp"
#include "morphizen/env_config.hpp"
#include "test_environment.hpp"
#include <algorithm> // std::generate
#include <chrono>
#include <fstream>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <onnxruntime_cxx_api.h>
#include <vector>
DEF_ENV_PARAM_2(TEST_ONNX_MODEL, "", std::string);

static std::unordered_map<std::string, std::string>
create_provider_options(const Config& config) {
  auto& provider_options = config.proto().session_options().provider_options();
  return std::unordered_map<std::string, std::string>(provider_options.begin(),
                                                      provider_options.end());
}

static std::unique_ptr<Ort::SessionOptions>
create_session_options(const Config& config) {
  auto& session_option_proto = config.proto().session_options();
  auto session_options = std::make_unique<Ort::SessionOptions>();

  // ORT_DISABLE_ALL = 0,
  // ORT_ENABLE_BASIC = 1,
  // ORT_ENABLE_EXTENDED = 2,
  // ORT_ENABLE_ALL = 99
  if (session_option_proto.has_graph_optimization_level()) {
    auto ort_opt_level = session_option_proto.graph_optimization_level();
    session_options->SetGraphOptimizationLevel(
        static_cast<GraphOptimizationLevel>(ort_opt_level));
  }

  for (auto& customop : session_option_proto.custom_op_library()) {
    session_options->RegisterCustomOpsLibrary(
        ToOrtString<ORTCHAR_T>()(customop).c_str());
  }

  if (session_option_proto.enable_profiling()) {
    auto profile_name = std::string("profile_test_onnx_runner");
    session_options->EnableProfiling(
        ToOrtString<ORTCHAR_T>()(profile_name).c_str());
  }

  for (const auto& session_config : session_option_proto.session_configs()) {
    session_options->AddConfigEntry(session_config.first.c_str(),
                                    session_config.second.c_str());
  }

  if (config.enable_vitisai_ep()) {
    auto provider_options = create_provider_options(config);
    // only append vitisai ep once each session_options
    session_options->AppendExecutionProvider_VitisAI(provider_options);
  }
  return std::move(session_options);
}

static void del_ctx_model(const std::filesystem::path& model_path) {
  try {
    std::filesystem::remove(model_path);
  } catch (std::exception& e) {
    std::cerr << "Exception: " << e.what() << std::endl;
  }
}

static void offline_compilation(const std::filesystem::path& model_path,
                                Ort::Env& env,
                                Ort::SessionOptions& session_options,
                                const Config& config) {
  Ort::ModelCompilationOptions compile_options(env, session_options);
  compile_options.SetInputModelPath(
      PathToString<ORTCHAR_T>()(model_path).c_str());
  auto output_model_path = config.get_ctx_model_path(model_path);
  del_ctx_model(output_model_path);
  compile_options.SetOutputModelPath(
      PathToString<ORTCHAR_T>()(output_model_path).c_str());
  auto embed_mode = config.embed_mode();
  compile_options.SetEpContextEmbedMode(embed_mode);
  auto status = Ort::CompileModel(env, compile_options);
  std::cout << "Offline compilation done" << std::endl;
}

static void run_many_sessions(const std::filesystem::path& model_path,
                              Ort::Env& env,
                              Ort::SessionOptions& session_options,
                              const Config& config) {
  // TODO: auto ep selection
  LOG(INFO) << "start to run model \"" << model_path << "\"" << std::endl;
  for (int i = 0; i < config.session_count(); ++i) {
    if (session_options.HasConfigEntry("ep.context_enable") &&
        session_options.GetConfigEntry("ep.context_enable") == "1") {
      del_ctx_model(config.get_ctx_model_path(model_path));
    }
    auto session = test_onnx_runner::create_session(model_path, env,
                                                    session_options, config);
    test_onnx_runner::run_session(*session, config);
  }
  LOG(INFO) << "run_session" << std::endl;
}

class TestOnnxRunner : public ::testing::Test {
protected:
  void SetUp() override {
    model_path = RESNET_50_PATH;
    if (!ENV_PARAM(TEST_ONNX_MODEL).empty()) {
      model_path = std::filesystem::u8path(ENV_PARAM(TEST_ONNX_MODEL));
    }
    env_config_map = Config::create(ENV_CONFIG_JSON_PATH);
    env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING,
                                     "test_onnx_runner");
  }
  void TearDown() override {
    // Cleanup code if needed
  }

  const Config& select_config(const std::string& name) {
    auto it = env_config_map.find(name);
    EXPECT_TRUE(it != env_config_map.end() && it->second != nullptr)
        << "Config with name \"" << name << "\" not found";
    const auto& ret = *(it->second);
    env = std::make_unique<Ort::Env>(ret.ort_log_level(),
                                     ret.ort_log_id().c_str());
    return ret;
  }

  std::filesystem::path model_path;
  std::unique_ptr<Ort::Env> env;

private:
  std::unordered_map<std::string, std::unique_ptr<Config>> env_config_map;
};

TEST_F(TestOnnxRunner, Run) {
  const auto& config = select_config("default_config");
  // shared session_options between generate ctx model and run ctx model
  auto session_options = create_session_options(config);
  // run model
  run_many_sessions(model_path, *env, *session_options, config);
  if (config.ep_context_enable()) {
    // run ctx model
    auto ctx_model_path = config.get_ctx_model_path(model_path);
    EXPECT_TRUE(std::filesystem::exists(ctx_model_path))
        << "ctx model not exist";
    session_options->AddConfigEntry("ep.context_enable", "0");
    run_many_sessions(ctx_model_path, *env, *session_options, config);
  }
}

TEST_F(TestOnnxRunner, SingleModelSingleSession) {
  const auto& config = select_config("single_session");
  auto session_options = create_session_options(config);
  run_many_sessions(model_path, *env, *session_options, config);
}

TEST_F(TestOnnxRunner, OfflineCompile) {
  const auto& config = select_config("offline_compilation_config");
  auto session_options = create_session_options(config);
  offline_compilation(model_path, *env, *session_options, config);
}
