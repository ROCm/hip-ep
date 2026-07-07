/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#define _CRT_SECURE_NO_WARNINGS
#include "morphizen/morphizen.hpp"
#include <cstdio>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <morphizen/file_stream.hpp>
TEST(FileStreamTest, HelloWorld) {
  { // Test the FileBuf class
    FILE *file = std::fopen("test.txt", "w+");
    ASSERT_NE(file, nullptr);
    morphizen::FileBuf fileBuf(file, 1024);
    std::ostream os(&fileBuf);
    os << "Hello, World!";
  }
  { // Test the FileStream class
    FILE *file = std::fopen("test.txt", "r");
    ASSERT_NE(file, nullptr);
    morphizen::FileStream fileStream(file);
    std::string line;
    std::getline(fileStream, line);
    ASSERT_EQ(line, "Hello, World!");
  }
  { // Test the FileStream class with a different buffer size
    FILE *file = std::fopen("test.txt", "r");
    ASSERT_NE(file, nullptr);
    morphizen::FileStream fileStream(file, 2048);
    std::string line;
    std::getline(fileStream, line);
    ASSERT_EQ(line, "Hello, World!");
  }
}
