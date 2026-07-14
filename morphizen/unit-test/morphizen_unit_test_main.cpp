/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#define _CRT_SECURE_NO_WARNINGS 1
#include <gtest/gtest.h>

#include <glog/logging.h>
#define ORT_API_MANUAL_INIT 1
#include <onnxruntime_cxx_api.h>
#if _WIN32
#ifdef _DEBUG
#include <crtdbg.h>
#endif
#endif
#include "morphizen-utils/morphizen_plugin.hpp"
#include "morphizen/morphizen-ort-api-ext.hpp"
template <typename... Args> void *morphizen_main_cmd(Args... args) {
  auto ep_dll = morphizen::Plugin::get("onnxruntime_morphizen_ep");
  if (ep_dll == nullptr) {
    LOG(ERROR) << "Failed to load MorphiZen EP";
    return nullptr;
  }
  const char *argv[] = {
      args...,
  };
  int argc = (int)(sizeof(argv) / sizeof(argv[0]));
  return ep_dll->get_method<void *, int, const char *[]>("morphizen_main")(
      argc, argv);
}

namespace gtest_example {
TEST(GTest, hello) { LOG(INFO) << "Hello GTest"; }
} // namespace gtest_example
static void show_test_case() {
  // Get the unit test singleton
  const ::testing::UnitTest &unit_test = *::testing::UnitTest::GetInstance();

  // Iterate over all test suites
  for (int i = 0; i < unit_test.total_test_suite_count(); ++i) {
    const ::testing::TestSuite *test_suite = unit_test.GetTestSuite(i);

    std::cout << test_suite->name() << "." << std::endl;

    // Iterate over all test infos in the suite
    for (int j = 0; j < test_suite->total_test_count(); ++j) {
      const ::testing::TestInfo *test_info = test_suite->GetTestInfo(j);

      std::cout << "  " << test_info->name() << " " << test_info->file() << ":"
                << test_info->line() << std::endl;
    }
  }
}
bool arg_get(int argc, const char *argv[], const char *name) {
  for (int i = 0; i < argc; ++i) {
    if (strcmp(argv[i], name) == 0) {
      return true;
    }
  }
  return false;
}

int main(int argc, const char *argv[]) {
#if _WIN32
#ifdef _DEBUG
  auto env_ci = getenv("CI");
  auto ci = std::string(env_ci ? env_ci : "");
  if (ci == "1") {
    // Disable assertion dialog in CI
    _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
    _CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ERROR, _CRTDBG_FILE_STDERR);
    _CrtSetReportMode(_CRT_WARN, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_WARN, _CRTDBG_FILE_STDERR);
    morphizen_main_cmd("disable_crt_diag");
  }
#endif
#endif
  auto ret = 0;
  {
    testing::InitGoogleTest(&argc, (char **)argv);
    if (arg_get(argc, argv, "--gtest_list_test_cases")) {
      std::cout << "List all test cases:" << std::endl;
      show_test_case();
      return 0;
    }
    Ort::InitApi();

    // Initialize the global MorphiZen ORT API for unit tests
    // This is required for tests that directly use MorphiZen APIs without
    // going through the EP registration path (which normally handles this)
#ifdef MORPHIZEN_ENABLE_MLIR_BACKEND
    // Enable text MLIR output mode for unit tests to allow verification of
    // MLIR content (e.g., checking for onnx.Return, onnx.Custom, etc.)
    // This must be set before any MLIR operations are performed.
#ifdef _WIN32
    _putenv("MORPHIZEN_SAVE_MLIR_AS_TEXT=1");
#else
    putenv(const_cast<char *>("MORPHIZEN_SAVE_MLIR_AS_TEXT=1"));
#endif

    morphizen::setup_global_morphizen_ort_api(morphizen::kMLIRBackend);
    LOG(INFO) << "Global MorphiZen ORT API initialized with MLIR backend";
    LOG(INFO) << "MORPHIZEN_SAVE_MLIR_AS_TEXT=1 (text mode enabled for tests)";
#else
    morphizen::setup_global_morphizen_ort_api(morphizen::kONNXIRBackend);
    LOG(INFO) << "Global MorphiZen ORT API initialized with ONNX-IR backend";
#endif
    ret = RUN_ALL_TESTS();
  }
  if (ret == 0) {
    std::cout << "All tests passed." << std::endl;
  } else {
    std::cout << "Some tests failed." << std::endl;
  }
  return ret;
}
