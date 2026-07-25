/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#define _CRT_SECURE_NO_WARNINGS 1
#include <gtest/gtest.h>

#ifdef __GNUC__
#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#endif
#include <glog/logging.h>
#define ORT_API_MANUAL_INIT 1
#include <onnxruntime_cxx_api.h>
#if _WIN32
#ifdef _DEBUG
#include <crtdbg.h>
#endif
#endif
#include "./test-environment.hpp"
namespace gtest_example {
TEST(GTest, hello) {
  LOG(INFO) << "CMAKE_CURRENT_BINARY_PATH=" << CMAKE_CURRENT_BINARY_PATH;
  // print current test name
  LOG(INFO) << "Test name: "
            << ::testing::UnitTest::GetInstance()->current_test_info()->name();
  LOG(INFO) << "Hello GTest";
}
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
  Ort::InitApi();
  testing::InitGoogleTest(&argc, (char **)argv);
  if (arg_get(argc, argv, "--gtest_list_test_cases")) {
    std::cout << "List all test cases:" << std::endl;
    show_test_case();
    return 0;
  }

  auto ret = 0;
  {
    auto env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_ERROR,
                                          "morphizen_unit_test");

    ret = RUN_ALL_TESTS();
  }
  if (ret == 0) {
    std::cout << "All tests passed." << std::endl;
  } else {
    std::cout << "Some tests failed." << std::endl;
  }
  return ret;
}
