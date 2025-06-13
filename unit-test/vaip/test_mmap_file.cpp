/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../../vaip-core/src/mmap_file.hpp"
#include "../../vaip-core/src/tar_file.hpp"
#include "./test_environment.hpp"
#include <gtest/gtest.h>
TEST(MMapfileTest, create) {
#ifdef _WIN32
  auto mmap_file = vaip_core::MemFile::create(RESNET_50_PATH);
  ASSERT_TRUE(mmap_file) << "Failed to create MMapFile object";
  auto mmap_file_obj = std::move(mmap_file);
  ASSERT_TRUE(mmap_file_obj->base() != nullptr)
      << "MMapFile base should be nullptr";
  ASSERT_EQ(mmap_file_obj->size(), 102196389) << "MMapFile size should be 0";
#endif
}
template <typename T> static void show_entry(const T& entry) {
  LOG(INFO) << "entry: " << entry->path()
            << (entry->real_path()
                    ? std::string("->") + entry->real_path().value()
                    : "")
            << " size=" << entry->size();
}
TEST(MMapfileTest, CreateTar) {
  auto tarFileName = CMAKE_CURRENT_BINARY_PATH / "sample.src.tar";

  auto tar_file_obj = vaip_core::TarFile::create_from_path(tarFileName);
  ASSERT_TRUE(tar_file_obj) << "Failed to create TarFile object";
  for (auto& entry : tar_file_obj->entries()) {
    show_entry(entry);
#ifdef _WIN32
    auto mmap = entry->mmap();
    ASSERT_TRUE(mmap) << "with mmap";
#endif
  }
}

TEST(MMapfileTest, CreateTarNoMMap) {
  auto tarFileName = CMAKE_CURRENT_BINARY_PATH / "sample.src.tar";

  auto tar_file_obj =
      vaip_core::TarFile::create_from_path(tarFileName, false /*enable mmap*/);
  ASSERT_TRUE(tar_file_obj) << "Failed to create TarFile object";
  for (auto& entry : tar_file_obj->entries()) {
    show_entry(entry);
#ifdef _WIN32
    auto mmap = entry->mmap();
    ASSERT_FALSE(mmap) << "no mmap";
#endif
  }
}
