#pragma once
#include <gtest/gtest.h>
#ifndef PYTHON_EXE_STR
#  define PYTHON_EXE_STR "python"
#endif
#ifndef TEST_CWD_STR
#  define TEST_CWD_STR "."
#endif

static constexpr std::filesystem::path TEST_CWD =
    std::filesystem::u8path(TEST_CWD_STR);
static constexpr std::filesystem::path PYTHON_EXE =
    std::filesystem::u8path(PYTHON_EXE_STR);
;