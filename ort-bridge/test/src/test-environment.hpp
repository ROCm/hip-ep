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

static const std::filesystem::path TEST_CWD =
    std::filesystem::u8path(TEST_CWD_STR);
static const std::filesystem::path CMAKE_CURRENT_BINARY_PATH =
    std::filesystem::u8path(TEST_CWD_STR);
static const std::filesystem::path PYTHON_EXE =
    std::filesystem::u8path(PYTHON_EXE_STR);
static const std::filesystem::path MORPHIZEN_TAR_EXE =
    std::filesystem::u8path(MORPHIZEN_TAR_EXE_STR);
static const std::filesystem::path TEST_SRC_DIR =
    std::filesystem::u8path(TEST_SRC_DIR_STR);
static const std::filesystem::path CMAKE_CURRENT_SOURCE_PATH =
    std::filesystem::u8path(TEST_SRC_DIR_STR);
static const std::filesystem::path RESNET_50_PATH =
    TEST_CWD / ".." / ".." / "unit-test" / "pt_resnet50.onnx";
static const std::filesystem::path ENV_CONFIG_JSON_PATH =
    TEST_CWD / "env_config.json";
static const std::filesystem::path MORPHIZEN_VITISAI_EP =
    std::filesystem::u8path(MORPHIZEN_VITISAI_EP_STR);
static const std::filesystem::path RESNET_50_MLIR_PATH =
    CMAKE_CURRENT_SOURCE_PATH / "src" / "pt_resnet50.onnx.mlir";
