##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

##
# Memory manager unit tests
##

cmake_minimum_required(VERSION 3.20)

# Prefer a pre-installed GTest (e.g. the one built and cached by CI into
# CMAKE_PREFIX_PATH) before falling back to FetchContent download.  This
# avoids an unnecessary network round-trip in CI and ensures the version
# matches what was pre-built (1.15.0 in the current workflow).
find_package(GTest 1.15.0 CONFIG QUIET)
if(GTest_FOUND)
  # Pre-installed GTest found in CMAKE_PREFIX_PATH (e.g. CI cache).
  set(GTEST_MAIN_TARGET GTest::gtest_main)
else()
  include(FetchContent)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.0.zip
  )
  # For Windows: Prevent overriding the parent project's compiler/linker settings
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
  set(GTEST_MAIN_TARGET gtest_main)
endif()

add_executable(mm_tests
  mm_tests.cpp
)

target_link_libraries(mm_tests
  PRIVATE
    mm
    ${GTEST_MAIN_TARGET}
)

target_include_directories(mm_tests PRIVATE ${PROJECT_SOURCE_DIR}/include)

add_test(NAME mm_tests COMMAND mm_tests)
