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
static const std::filesystem::path RESNET_50_PATH =
    TEST_CWD / "pt_resnet50.onnx";
