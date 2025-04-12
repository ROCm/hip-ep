/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */
#define _CRT_SECURE_NO_WARNINGS
#include "../vaip-core/src/file_stream.hpp"
#include "morphizen/vaip.hpp"
#include <cstdio>
#include <glog/logging.h>
#include <gtest/gtest.h>
TEST(FileStreamTest, HelloWorld) {
  { // Test the FileBuf class
    FILE* file = std::fopen("test.txt", "w+");
    ASSERT_NE(file, nullptr);
    vaip_core::FileBuf fileBuf(file, 1024);
    std::ostream os(&fileBuf);
    os << "Hello, World!";
  }
  { // Test the FileStream class
    FILE* file = std::fopen("test.txt", "r");
    ASSERT_NE(file, nullptr);
    vaip_core::FileStream fileStream(file);
    std::string line;
    std::getline(fileStream, line);
    ASSERT_EQ(line, "Hello, World!");
  }
  { // Test the FileStream class with a different buffer size
    FILE* file = std::fopen("test.txt", "r");
    ASSERT_NE(file, nullptr);
    vaip_core::FileStream fileStream(file, 2048);
    std::string line;
    std::getline(fileStream, line);
    ASSERT_EQ(line, "Hello, World!");
  }
}
