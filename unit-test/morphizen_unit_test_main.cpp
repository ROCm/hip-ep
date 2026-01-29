/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#define _CRT_SECURE_NO_WARNINGS 1
#include <gtest/gtest.h>

#ifdef __GNUC__
#  pragma GCC diagnostic ignored "-Wpedantic"
#  pragma GCC diagnostic ignored "-Wconversion"
#  pragma GCC diagnostic ignored "-Wsign-compare"
#  pragma GCC diagnostic ignored "-Wunused-variable"
#  pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#endif
#include <glog/logging.h>
#define ORT_API_MANUAL_INIT 1
#include <onnxruntime_cxx_api.h>
#if _WIN32
#  ifdef _DEBUG
#    include <crtdbg.h>
#  endif
#endif
#include "morphizen/morphizen.hpp"
template <typename... Args> void* morphizen_main_cmd(Args... args) {
  auto ep_dll = morphizen::Plugin::get("onnxruntime_morphizen_ep");
  if (ep_dll == nullptr) {
    LOG(ERROR) << "Failed to load MorphiZen EP";
    return nullptr;
  }
  const char* argv[] = {
      args...,
  };
  int argc = (int)(sizeof(argv) / sizeof(argv[0]));
  return ep_dll->get_method<void*, int, const char*[]>("morphizen_main")(argc,
                                                                         argv);
}

namespace gtest_example {
TEST(GTest, hello) { LOG(INFO) << "Hello GTest"; }
} // namespace gtest_example
static void show_test_case() {
  // Get the unit test singleton
  const ::testing::UnitTest& unit_test = *::testing::UnitTest::GetInstance();

  // Iterate over all test suites
  for (int i = 0; i < unit_test.total_test_suite_count(); ++i) {
    const ::testing::TestSuite* test_suite = unit_test.GetTestSuite(i);

    std::cout << test_suite->name() << "." << std::endl;

    // Iterate over all test infos in the suite
    for (int j = 0; j < test_suite->total_test_count(); ++j) {
      const ::testing::TestInfo* test_info = test_suite->GetTestInfo(j);

      std::cout << "  " << test_info->name() << " " << test_info->file() << ":"
                << test_info->line() << std::endl;
    }
  }
}
bool arg_get(int argc, const char* argv[], const char* name) {
  for (int i = 0; i < argc; ++i) {
    if (strcmp(argv[i], name) == 0) {
      return true;
    }
  }
  return false;
}

int main(int argc, const char* argv[]) {
  Ort::InitApi();
#if _WIN32
#  ifdef _DEBUG
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
#  endif
#endif
  auto ret = 0;
  {
    auto env =
        std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_ERROR, "vaip_unit_test");
    /* morphizen::StaticPluginRegister::sync_static_plugin_into_module(
        "onnxruntime_morphizen_ep");
        */
    Ort::SessionOptions().AppendExecutionProvider_VitisAI();
    morphizen::set_the_global_api(
        morphizen::Plugin::invoke<morphizen::OrtApiForMorphizen*>(
            "onnxruntime_morphizen_ep", "get_the_global_api"));
    testing::InitGoogleTest(&argc, (char**)argv);
    if (arg_get(argc, argv, "--gtest_list_test_cases")) {
      std::cout << "List all test cases:" << std::endl;
      show_test_case();
      return 0;
    }

    ret = RUN_ALL_TESTS();
  }
  if (ret == 0) {
    std::cout << "All tests passed." << std::endl;
  } else {
    std::cout << "Some tests failed." << std::endl;
  }
  return ret;
}
