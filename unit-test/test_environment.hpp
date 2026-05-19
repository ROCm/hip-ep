/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once
#include <filesystem>
#include <glog/logging.h>
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

#ifdef BAZEL_CURRENT_REPOSITORY
// ── Bazel build ──────────────────────────────────────────────────────────────
// BAZEL_CURRENT_REPOSITORY is injected only when
// @bazel_tools//tools/cpp/runfiles is in deps, so this guard is reliable.
#  include "tools/cpp/runfiles/runfiles.h"

// TEST_CWD is the per-invocation writable scratch directory provided by the
// Bazel test runner (TEST_TMPDIR).  All files generated during the test (ORT
// EP context models, tar archives, …) must go here — never into the runfiles
// tree, which is read-only on Linux sandboxed builds.
static const std::filesystem::path TEST_CWD = []() {
  const char* d = std::getenv("TEST_TMPDIR");
  CHECK(d != nullptr) << "TEST_TMPDIR not set — must run under bazel test";
  return std::filesystem::u8path(d);
}();

// CMAKE_CURRENT_BINARY_PATH is the historical alias used by non-E2E tests for
// the same writable output directory.
static const std::filesystem::path CMAKE_CURRENT_BINARY_PATH = TEST_CWD;

// Read-only source files are resolved via the Bazel runfiles library; they
// live in the runfiles tree, not in TEST_TMPDIR.
static const std::filesystem::path RESNET_50_PATH = []() {
  std::string err;
  auto rf = bazel::tools::cpp::runfiles::Runfiles::CreateForTest(
      BAZEL_CURRENT_REPOSITORY, &err);
  CHECK(rf != nullptr) << "Runfiles not available: " << err;
#  ifdef MORPHIZEN_ENABLE_MLIR_BACKEND
  // The MLIR backend's Model::load() parses MLIR text, not binary ONNX.
  // pt_resnet50.onnx.mlir lives in ort-bridge/test/src/ and is exported
  // via exports_files for use here.
  auto p = rf->Rlocation("_main/ort-bridge/test/src/pt_resnet50.onnx.mlir");
  CHECK(!p.empty()) << "pt_resnet50.onnx.mlir not found in runfiles";
#  else
  auto p = rf->Rlocation("_main/unit-test/data/pt_resnet50.onnx");
  CHECK(!p.empty()) << "pt_resnet50.onnx not found in runfiles";
#  endif
  return std::filesystem::u8path(p);
}();

// The E2E test config JSON is generated at build time and staged in runfiles.
static const std::filesystem::path E2E_TEST_CONFIG_JSON_PATH = []() {
  std::string err;
  auto rf = bazel::tools::cpp::runfiles::Runfiles::CreateForTest(
      BAZEL_CURRENT_REPOSITORY, &err);
  CHECK(rf != nullptr) << "Runfiles not available: " << err;
  auto p =
      rf->Rlocation("_main/unit-test/data/morphizen_e2e_tests_config.json");
  CHECK(!p.empty()) << "morphizen_e2e_tests_config.json not found in runfiles";
  return std::filesystem::u8path(p);
}();

// sample.src.tar is a read-only test fixture staged in runfiles.
static const std::filesystem::path SAMPLE_SRC_TAR_PATH = []() {
  std::string err;
  auto rf = bazel::tools::cpp::runfiles::Runfiles::CreateForTest(
      BAZEL_CURRENT_REPOSITORY, &err);
  CHECK(rf != nullptr) << "Runfiles not available: " << err;
  auto p = rf->Rlocation("_main/unit-test/data/sample.src.tar");
  CHECK(!p.empty()) << "sample.src.tar not found in runfiles";
  return std::filesystem::u8path(p);
}();

static const std::filesystem::path ENV_CONFIG_JSON_PATH =
    TEST_CWD / "env_config.json";

#else
// ── CMake / ctest build ──────────────────────────────────────────────────────
// TEST_CWD_STR is set to CMAKE_CURRENT_BINARY_DIR by CMake.  ctest runs the
// test with CWD equal to that directory, so data files copied there by CMake
// and files generated during the test all live in the same place.
static const std::filesystem::path TEST_CWD =
    std::filesystem::u8path(TEST_CWD_STR);
static const std::filesystem::path CMAKE_CURRENT_BINARY_PATH = TEST_CWD;
#  ifdef MORPHIZEN_ENABLE_MLIR_BACKEND
static const std::filesystem::path RESNET_50_PATH =
    TEST_CWD / "pt_resnet50.onnx.mlir";
#  else
static const std::filesystem::path RESNET_50_PATH =
    TEST_CWD / "pt_resnet50.onnx";
#  endif
static const std::filesystem::path SAMPLE_SRC_TAR_PATH =
    TEST_CWD / "sample.src.tar";
static const std::filesystem::path ENV_CONFIG_JSON_PATH =
    TEST_CWD / "env_config.json";
static const std::filesystem::path E2E_TEST_CONFIG_JSON_PATH =
    TEST_CWD / "morphizen_e2e_tests_config.json";
#endif
