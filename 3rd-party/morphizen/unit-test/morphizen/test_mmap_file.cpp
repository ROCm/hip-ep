/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../../morphizen-core/src/mmap_file.hpp"
#include "../../morphizen-core/src/tar_file.hpp"
#include "./test_environment.hpp"
#include <gtest/gtest.h>
TEST(MMapfileTest, create) {
#ifdef _WIN32
  auto RESNET_50_PATH_MMAP =
      RESNET_50_PATH.parent_path() / "MMapfileTest.create";
  // Fix: Remove target file first to ensure clean copy in CI cache environment
  std::filesystem::remove(RESNET_50_PATH_MMAP);

  // Get expected size from source file before copying
  auto expected_size = std::filesystem::file_size(RESNET_50_PATH);
  std::filesystem::copy_file(RESNET_50_PATH, RESNET_50_PATH_MMAP);

  auto mmap_file = morphizen::MemFile::create(RESNET_50_PATH_MMAP);
  ASSERT_TRUE(mmap_file) << "Failed to create MMapFile object";
  auto mmap_file_obj = std::move(mmap_file);
  ASSERT_TRUE(mmap_file_obj->base() != nullptr)
      << "MMapFile base should not be nullptr";
  ASSERT_EQ(mmap_file_obj->size(), expected_size)
      << "MMapFile size should match source file size";
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
  auto tarFileName1 = SAMPLE_SRC_TAR_PATH;
  auto tarFileName2 =
      CMAKE_CURRENT_BINARY_PATH / "sample.src.tar.MMapfileTest.CreateTar";
  // Fix: Remove update_existing to ensure file is always overwritten
  std::filesystem::copy_file(tarFileName1, tarFileName2,
                             std::filesystem::copy_options::overwrite_existing);
  auto tar_file_obj = morphizen::TarFile::create_from_path(tarFileName2);
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
  auto tarFileName1 = SAMPLE_SRC_TAR_PATH;
  auto tarFileName2 =
      CMAKE_CURRENT_BINARY_PATH / "sample.src.tar.MMapfileTest.CreateTarNoMMap";
  // Fix: Remove update_existing to ensure file is always overwritten
  std::filesystem::copy_file(tarFileName1, tarFileName2,
                             std::filesystem::copy_options::overwrite_existing);
  auto tar_file_obj =
      morphizen::TarFile::create_from_path(tarFileName2, false /*enable mmap*/);
  ASSERT_TRUE(tar_file_obj) << "Failed to create TarFile object";
  for (auto& entry : tar_file_obj->entries()) {
    show_entry(entry);
#ifdef _WIN32
    auto mmap = entry->mmap();
    ASSERT_FALSE(mmap) << "no mmap";
#endif
  }
}
