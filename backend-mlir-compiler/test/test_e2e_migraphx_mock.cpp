/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

/**
 * @file test_e2e_migraphx_mock.cpp
 * @brief GPU-free E2E tests of whole-graph backend selection and the MIGraphX
 *        path (docs/design/migraphx-backend.md).
 *
 * The EP loads mock_migraphx_plugin.cpp instead of the real MIGraphX plugin
 * (HIPDNN_EP_MIGRAPHX_PLUGIN_DIR), so these tests cover routing, claiming,
 * Compile/Compute wiring and error handling, but not MIGraphX itself. Each
 * test asserts the EP's "hip-ep backend=" routing log line, so a session that
 * fails for an unrelated reason does not pass by accident.
 *
 * Runs in its own process: the EP loads the plugin once per process.
 * Needs MORPHIZEN_EP_ENABLE_CPU_DEVICE=1 without a GPU (set by main()).
 */

#include <gtest/gtest.h>
#include <onnxruntime_cxx_api.h>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef MORPHIZEN_EP_LIB_PATH
#error "MORPHIZEN_EP_LIB_PATH must be defined by CMake"
#endif
#ifndef MORPHIZEN_CONFIG_MLIR_PATH
#error "MORPHIZEN_CONFIG_MLIR_PATH must be defined by CMake"
#endif
#ifndef TEST_MODEL_DIR
#error "TEST_MODEL_DIR must be defined by CMake"
#endif
#ifndef MOCK_MGX_PLUGIN_DIR
#error "MOCK_MGX_PLUGIN_DIR must be defined by CMake"
#endif

namespace fs = std::filesystem;

namespace {

#ifdef _WIN32
void set_env(const char *k, const char *v) { _putenv_s(k, v); }
void unset_env(const char *k) { _putenv_s(k, ""); }
#else
void set_env(const char *k, const char *v) { setenv(k, v, /*overwrite=*/1); }
void unset_env(const char *k) { unsetenv(k); }
#endif

// Sets an environment variable for the lifetime of the object.
struct ScopedEnv {
  const char *key;
  ScopedEnv(const char *k, const char *v) : key(k) { set_env(k, v); }
  ~ScopedEnv() { unset_env(key); }
};

std::mutex g_log_mutex;
std::vector<std::string> g_routing_logs;

void ORT_API_CALL capture_log(void *, OrtLoggingLevel, const char *,
                              const char *, const char *, const char *message) {
  if (std::string(message).find("hip-ep backend=") == std::string::npos)
    return;
  std::lock_guard<std::mutex> lock(g_log_mutex);
  g_routing_logs.emplace_back(message);
}

std::string model(const char *name) {
  return (fs::path(TEST_MODEL_DIR) / name).string();
}

Ort::Env *g_env = nullptr;

} // namespace

class MigraphxMockE2ETest : public ::testing::Test {
protected:
  void SetUp() override {
    std::lock_guard<std::mutex> lock(g_log_mutex);
    g_routing_logs.clear();
  }

  // Single routing line logged by the EP for the last session.
  static std::string RoutingLog() {
    std::lock_guard<std::mutex> lock(g_log_mutex);
    return g_routing_logs.empty() ? "<no routing log>" : g_routing_logs.back();
  }

  // Session options with the EP appended and optional gpu_backend.
  static Ort::SessionOptions Options(const char *gpu_backend,
                                     bool disable_cpu_fallback = true) {
    const OrtEpDevice *device = nullptr;
    for (const auto &d : g_env->GetEpDevices()) {
      if (std::string(d.EpName()) == "MorphiZenExecutionProvider") {
        device = static_cast<const OrtEpDevice *>(d);
        break;
      }
    }
    if (device == nullptr)
      throw std::runtime_error("MorphiZen EP device not found");
    Ort::SessionOptions options;
    if (disable_cpu_fallback)
      options.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
    std::vector<const char *> keys = {"config_file"};
    std::vector<const char *> values = {MORPHIZEN_CONFIG_MLIR_PATH};
    if (gpu_backend != nullptr) {
      keys.push_back("gpu_backend");
      values.push_back(gpu_backend);
    }
    Ort::ThrowOnError(Ort::GetApi().SessionOptionsAppendExecutionProvider_V2(
        options, *g_env, &device, 1, keys.data(), values.data(), keys.size()));
    return options;
  }

  static Ort::Session Open(const std::string &path,
                           const Ort::SessionOptions &options) {
#ifdef _WIN32
    return Ort::Session(*g_env, fs::path(path).wstring().c_str(), options);
#else
    return Ort::Session(*g_env, path.c_str(), options);
#endif
  }

  // Runs elu_two_io with x0 = {1,2,3,4}, x1 = {10,20,30,40}.
  static std::vector<std::vector<float>> RunTwoIo(Ort::Session &session) {
    std::vector<float> x0 = {1, 2, 3, 4}, x1 = {10, 20, 30, 40};
    const int64_t shape[] = {1, 4};
    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<Ort::Value> inputs;
    inputs.push_back(
        Ort::Value::CreateTensor<float>(mem, x0.data(), 4, shape, 2));
    inputs.push_back(
        Ort::Value::CreateTensor<float>(mem, x1.data(), 4, shape, 2));
    const char *in_names[] = {"x0", "x1"};
    const char *out_names[] = {"y0", "y1"};
    auto outputs = session.Run(Ort::RunOptions{nullptr}, in_names,
                               inputs.data(), 2, out_names, 2);
    std::vector<std::vector<float>> result;
    for (auto &o : outputs) {
      const float *p = o.GetTensorData<float>();
      result.emplace_back(p, p + 4);
    }
    return result;
  }

  // Mock plugin result: y_j = x0 + 2 * x1 + 100 * (j + 1).
  static void ExpectMockTwoIoOutputs(Ort::Session &session) {
    const auto y = RunTwoIo(session);
    ASSERT_EQ(y.size(), 2u);
    EXPECT_EQ(y[0], (std::vector<float>{121, 142, 163, 184}));
    EXPECT_EQ(y[1], (std::vector<float>{221, 242, 263, 284}));
  }
};

/// gpu_backend=migraphx claims the whole graph, binds inputs by name and
/// outputs by graph index, and returns the plugin's results.
TEST_F(MigraphxMockE2ETest, MigraphxRunsWholeGraph) {
  ScopedEnv outputs("MOCK_MGX_NUM_OUTPUTS", "2");
  auto session = Open(model("elu_two_io.onnx"), Options("migraphx"));
  EXPECT_NE(RoutingLog().find("backend=migraphx policy_version=1 "
                              "requested=migraphx reason=gpu_backend=migraphx"),
            std::string::npos)
      << RoutingLog();
  ExpectMockTwoIoOutputs(session);
}

/// auto (the default) uses MIGraphX when HIP claims nothing (Elu has no HIP
/// converter).
TEST_F(MigraphxMockE2ETest, AutoUsesMigraphxWhenHipFails) {
  ScopedEnv outputs("MOCK_MGX_NUM_OUTPUTS", "2");
  auto session = Open(model("elu_two_io.onnx"), Options(nullptr));
  EXPECT_NE(RoutingLog().find("backend=migraphx policy_version=1 "
                              "requested=auto reason=HIP did not compile"),
            std::string::npos)
      << RoutingLog();
  ExpectMockTwoIoOutputs(session);
}

/// auto keeps HIP for a graph HIP compiles, even with the plugin available.
TEST_F(MigraphxMockE2ETest, AutoKeepsHipWhenHipCompiles) {
  auto session = Open(model("two_layer_conv.onnx"), Options("auto"));
  EXPECT_NE(RoutingLog().find("backend=hip policy_version=1 requested=auto"),
            std::string::npos)
      << RoutingLog();
}

/// gpu_backend=hip never tries MIGraphX.
TEST_F(MigraphxMockE2ETest, HipNeverUsesMigraphx) {
  ScopedEnv outputs("MOCK_MGX_NUM_OUTPUTS", "2");
  EXPECT_THROW(Open(model("elu_two_io.onnx"), Options("hip")), Ort::Exception);
  EXPECT_NE(RoutingLog().find("backend=none policy_version=1 requested=hip"),
            std::string::npos)
      << RoutingLog();
}

/// Each v1 eligibility rule rejects the graph through the real ORT graph.
TEST_F(MigraphxMockE2ETest, IneligibleGraphsAreNotClaimed) {
  struct Case {
    const char *model;
    const char *reason;
  } cases[] = {
      {"elu_dynamic.onnx", "a graph input has a dynamic or symbolic dimension"},
      {"elu_past.onnx", "graph input 'past_key' looks like a KV cache"},
      {"lstm.onnx", "graph contains stateful op LSTM"},
  };
  for (const auto &c : cases) {
    SetUp();
    EXPECT_THROW(Open(model(c.model), Options("migraphx")), Ort::Exception)
        << c.model;
    EXPECT_NE(RoutingLog().find(std::string("backend=none policy_version=1 "
                                            "requested=migraphx reason="
                                            "MIGraphX ineligible: ") +
                                c.reason),
              std::string::npos)
        << c.model << ": " << RoutingLog();
  }
}

/// A session created from bytes has no model path for MIGraphX to parse.
TEST_F(MigraphxMockE2ETest, InMemoryModelIsNotClaimed) {
  std::ifstream f(model("elu_two_io.onnx"), std::ios::binary);
  const std::vector<char> bytes((std::istreambuf_iterator<char>(f)),
                                std::istreambuf_iterator<char>());
  ASSERT_FALSE(bytes.empty());
  EXPECT_THROW(
      Ort::Session(*g_env, bytes.data(), bytes.size(), Options("migraphx")),
      Ort::Exception);
  EXPECT_NE(RoutingLog().find("needs the model file path"), std::string::npos)
      << RoutingLog();
}

/// With CPU fallback allowed, an ineligible graph under auto still runs on the
/// CPU EP exactly as before this feature.
TEST_F(MigraphxMockE2ETest, AutoIneligibleFallsBackToCpu) {
  auto session = Open(model("elu_dynamic.onnx"),
                      Options("auto", /*disable_cpu_fallback=*/false));
  EXPECT_NE(RoutingLog().find("backend=none policy_version=1 requested=auto "
                              "reason=MIGraphX ineligible"),
            std::string::npos)
      << RoutingLog();
  std::vector<float> x = {-1, 0, 1, 2, -2, 3, -3, 4};
  const int64_t shape[] = {2, 4};
  auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  auto input = Ort::Value::CreateTensor<float>(mem, x.data(), 8, shape, 2);
  const char *in_names[] = {"x"};
  const char *out_names[] = {"y"};
  auto out =
      session.Run(Ort::RunOptions{nullptr}, in_names, &input, 1, out_names, 1);
  const float *y = out[0].GetTensorData<float>();
  for (size_t i = 0; i < x.size(); ++i)
    EXPECT_FLOAT_EQ(y[i], x[i] >= 0 ? x[i] : std::expm1(x[i])) << i;
}

/// A MIGraphX compile failure is reported and nothing is claimed.
TEST_F(MigraphxMockE2ETest, CompileFailureIsNotClaimed) {
  ScopedEnv fail("MOCK_MGX_FAIL", "compile");
  EXPECT_THROW(Open(model("elu_two_io.onnx"), Options("migraphx")),
               Ort::Exception);
  EXPECT_NE(RoutingLog().find("reason=MIGraphX unavailable: mock compile "
                              "failure"),
            std::string::npos)
      << RoutingLog();
}

/// A program whose output count differs from the graph's is not used.
TEST_F(MigraphxMockE2ETest, OutputCountMismatchIsNotClaimed) {
  ScopedEnv outputs("MOCK_MGX_NUM_OUTPUTS", "1");
  EXPECT_THROW(Open(model("elu_two_io.onnx"), Options("migraphx")),
               Ort::Exception);
  EXPECT_NE(RoutingLog().find("MIGraphX program has 1 outputs, graph has 2"),
            std::string::npos)
      << RoutingLog();
}

/// A failure inside Compute reaches Session::Run as an error, and the
/// session keeps working afterwards.
TEST_F(MigraphxMockE2ETest, RunFailureIsReturnedAsError) {
  ScopedEnv outputs("MOCK_MGX_NUM_OUTPUTS", "2");
  auto session = Open(model("elu_two_io.onnx"), Options("migraphx"));
  {
    ScopedEnv fail("MOCK_MGX_FAIL", "run");
    try {
      RunTwoIo(session);
      FAIL() << "Run succeeded with MOCK_MGX_FAIL=run";
    } catch (const Ort::Exception &e) {
      EXPECT_NE(std::string(e.what()).find("mock run failure"),
                std::string::npos)
          << e.what();
    }
  }
  ExpectMockTwoIoOutputs(session);
}

/// EPContext export is not implemented for the MIGraphX path.
TEST_F(MigraphxMockE2ETest, EpContextIsRejected) {
  ScopedEnv outputs("MOCK_MGX_NUM_OUTPUTS", "2");
  auto options = Options("migraphx");
  options.AddConfigEntry("ep.context_enable", "1");
  options.AddConfigEntry(
      "ep.context_file_path",
      (fs::temp_directory_path() / "mgx_mock_ctx.onnx").string().c_str());
  try {
    Open(model("elu_two_io.onnx"), options);
    FAIL() << "session creation succeeded with ep.context_enable=1";
  } catch (const Ort::Exception &e) {
    EXPECT_NE(std::string(e.what()).find("ep.context_enable=1 is not "
                                         "supported"),
              std::string::npos)
        << e.what();
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  // Must be set before the EP library loads the plugin.
  set_env("HIPDNN_EP_MIGRAPHX_PLUGIN_DIR", MOCK_MGX_PLUGIN_DIR);
  if (std::getenv("MORPHIZEN_EP_ENABLE_CPU_DEVICE") == nullptr)
    set_env("MORPHIZEN_EP_ENABLE_CPU_DEVICE", "1");

  Ort::Env env(ORT_LOGGING_LEVEL_INFO, "MigraphxMockE2ETest", capture_log,
               nullptr);
  g_env = &env;
#ifdef _WIN32
  Ort::ThrowOnError(Ort::GetApi().RegisterExecutionProviderLibrary(
      env, "MorphiZenExecutionProvider",
      fs::path(MORPHIZEN_EP_LIB_PATH).wstring().c_str()));
#else
  Ort::ThrowOnError(Ort::GetApi().RegisterExecutionProviderLibrary(
      env, "MorphiZenExecutionProvider", MORPHIZEN_EP_LIB_PATH));
#endif
  return RUN_ALL_TESTS();
}
