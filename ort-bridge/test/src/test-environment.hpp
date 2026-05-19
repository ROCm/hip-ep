/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once
#include <filesystem>
#include <gtest/gtest.h>
#ifndef PYTHON_EXE_STR
#  define PYTHON_EXE_STR "python"
#endif
#ifndef TEST_CWD_STR
#  define TEST_CWD_STR "."
#endif
#ifndef MORPHIZEN_TAR_EXE_STR
#  define MORPHIZEN_TAR_EXE_STR "morphizen-tar"
#endif
#ifndef TEST_SRC_DIR_STR
#  define TEST_SRC_DIR_STR "."
#endif

static const std::filesystem::path PYTHON_EXE =
    std::filesystem::u8path(PYTHON_EXE_STR);
static const std::filesystem::path MORPHIZEN_TAR_EXE =
    std::filesystem::u8path(MORPHIZEN_TAR_EXE_STR);
static const std::filesystem::path TEST_SRC_DIR =
    std::filesystem::u8path(TEST_SRC_DIR_STR);
static const std::filesystem::path CMAKE_CURRENT_SOURCE_PATH =
    std::filesystem::u8path(TEST_SRC_DIR_STR);
static const std::filesystem::path ENV_CONFIG_JSON_PATH =
    std::filesystem::u8path(TEST_CWD_STR) / "env_config.json";
// In a Bazel build, ORT's RegisterExecutionProviderLibrary() resolves relative
// paths against GetRuntimePath() (the directory of onnxruntime.dll), not CWD.
// Use the C++ runfiles library to get an absolute path, mirroring CMake's
// $<TARGET_FILE:onnxruntime_morphizen_ep> generator expression.
// BAZEL_CURRENT_REPOSITORY is injected by Bazel only when
// @bazel_tools//tools/cpp/runfiles is in deps, so this guard is reliable.
#ifdef BAZEL_CURRENT_REPOSITORY
#  include "tools/cpp/runfiles/runfiles.h"
static const std::filesystem::path MORPHIZEN_MORPHIZEN_EP = []() {
  std::string err;
  auto rf = bazel::tools::cpp::runfiles::Runfiles::CreateForTest(
      BAZEL_CURRENT_REPOSITORY, &err);
  if (rf) {
    auto p = rf->Rlocation(std::string("_main/") + MORPHIZEN_MORPHIZEN_EP_STR);
    if (!p.empty())
      return std::filesystem::u8path(p);
  }
  return std::filesystem::u8path(MORPHIZEN_MORPHIZEN_EP_STR);
}();
static const std::filesystem::path RESNET_50_PATH = []() {
  std::string err;
  auto rf = bazel::tools::cpp::runfiles::Runfiles::CreateForTest(
      BAZEL_CURRENT_REPOSITORY, &err);
  if (rf) {
    auto p = rf->Rlocation("_main/" RESNET_50_ONNX_STR);
    if (!p.empty())
      return std::filesystem::u8path(p);
  }
  return std::filesystem::u8path(TEST_CWD_STR) / ".." / ".." / "unit-test" /
         "pt_resnet50.onnx";
}();
#else
static const std::filesystem::path MORPHIZEN_MORPHIZEN_EP =
    std::filesystem::u8path(MORPHIZEN_MORPHIZEN_EP_STR);
static const std::filesystem::path RESNET_50_PATH =
    std::filesystem::u8path(TEST_CWD_STR) / ".." / ".." / "unit-test" /
    "pt_resnet50.onnx";
#endif
static const std::filesystem::path TEST_CWD =
    std::filesystem::u8path(TEST_CWD_STR);
static const std::filesystem::path CMAKE_CURRENT_BINARY_PATH = TEST_CWD;
static const std::filesystem::path RESNET_50_MLIR_PATH =
    CMAKE_CURRENT_SOURCE_PATH / "src" / "pt_resnet50.onnx.mlir";
