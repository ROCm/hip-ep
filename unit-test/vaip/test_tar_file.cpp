/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#define _CRT_SECURE_NO_WARNINGS
#include "../../vaip-core/src/tar_file.hpp"
#include "debug_logger.hpp"
#include <boost/process.hpp>
#include <cerrno>
#include <cstring>
#include <gtest/gtest.h>
TEST(TarFileTest, ReadFrom) {
  auto tarFileName = CMAKE_CURRENT_BINARY_PATH / "sample.src.tar";
  auto tarStream = std::make_unique<std::fstream>(
      tarFileName, std::ios::binary | std::ios::in | std::ios::out);
  auto tar_file_obj = vaip_core::TarFile::create(std::move(tarStream));
  ASSERT_TRUE(tar_file_obj) << "Failed to create TarFile object";
  for (auto& entry : tar_file_obj->entries()) {
    LOG(INFO) << "entry: " << entry->path()
              << (entry->real_path()
                      ? std::string("->") + entry->real_path().value()
                      : "")
              << " size=" << entry->size();
  }
}

TEST(TarFileTest, DoubleRead) {
  auto tarFileName = CMAKE_CURRENT_BINARY_PATH / "sample.src.tar";

  auto tarStream = std::make_unique<std::fstream>(
      tarFileName, std::ios::binary | std::ios::in | std::ios::out);
  ASSERT_TRUE(tarStream->is_open())
      << "Failed to open tar file: " << tarFileName
      << " Error opening file: " << std::strerror(errno);
  auto tar_file_obj = vaip_core::TarFile::create(std::move(tarStream));
  ASSERT_TRUE(tar_file_obj) << "Failed to create TarFile object";
  auto& entries = tar_file_obj->entries();
  ASSERT_TRUE(!entries.empty()) << "Failed to read tar entries";
  ASSERT_EQ(entries.size(), 2) << "Expected two tar entries";
  // read first entry twice
  auto& entry = entries[0];
  auto name = entry->path();
  // 1
  auto stream = tar_file_obj->open_for_read(name);
  ASSERT_TRUE(stream) << "Failed to open entry for read: " << entry->path();
  std::string buffer1;
  buffer1.resize(entry->size());
  stream->read(&buffer1[0], entry->size());
  ASSERT_TRUE(stream) << "Failed to read entry: " << entry->path();
  ASSERT_EQ(stream->gcount(), entry->size())
      << "Entry size mismatch for entry: " << entry->path();
  // 2
  auto stream2 = tar_file_obj->open_for_read(name);
  ASSERT_TRUE(stream) << "Failed to open entry for read: " << entry->path();
  std::string buffer2;
  buffer2.resize(entry->size());
  stream2->read(&buffer2[0], entry->size());
  ASSERT_TRUE(stream) << "Failed to read entry: " << entry->path();
  ASSERT_EQ(stream->gcount(), entry->size())
      << "Entry size mismatch for entry: " << entry->path();
  ASSERT_EQ(buffer1, buffer2) << "double read got different result";
}
static void test_write_override(bool write_same_data) {
  {
    auto const_test_content = std::string("1234567890");
    auto srcFileName = CMAKE_CURRENT_BINARY_PATH / "sample.src.tar";
    auto destFileName =
        CMAKE_CURRENT_BINARY_PATH / "sample.src.tar.copy_for_test";

    {
      // copy the sample.src.tar to a new file
      auto copy_is_ok = std::filesystem::copy_file(
          srcFileName, destFileName,
          std::filesystem::copy_options::overwrite_existing);
      // check if the copy is ok
      ASSERT_TRUE(copy_is_ok)
          << "Failed to copy file: " << srcFileName << " to " << destFileName;
      auto tarStream = std::make_unique<std::fstream>(
          destFileName, std::ios::binary | std::ios::in | std::ios::out);
      ASSERT_TRUE(tarStream->is_open())
          << "Failed to open tar file: " << srcFileName
          << " Error opening file: " << std::strerror(errno);
      auto tar_file_obj = vaip_core::TarFile::create(std::move(tarStream));
      ASSERT_TRUE(tar_file_obj) << "Failed to create TarFile object";
      auto& entries = tar_file_obj->entries();
      ASSERT_TRUE(!entries.empty()) << "Failed to read tar entries";
      ASSERT_EQ(entries.size(), 2) << "Expected two tar entries";
      // read first entry twice
      auto& entry = entries[0];
      auto name = entry->path();
      if (write_same_data) {
        // read the data into const_test_content
        auto stream = tar_file_obj->open_for_read(name);
        ASSERT_TRUE(stream)
            << "Failed to open entry for read: " << entry->path();
        std::string buffer;
        buffer.resize(entry->size());
        stream->read(&buffer[0], entry->size());
        ASSERT_TRUE(stream) << "Failed to read entry: " << entry->path();
        ASSERT_EQ(stream->gcount(), entry->size())
            << "Entry size mismatch for entry: " << entry->path();
        const_test_content = buffer;
      }
      // write
      std::string buffer = const_test_content;
      auto stream = tar_file_obj->open_for_write(name);
      stream->write(buffer.data(), buffer.size());
      stream->flush();
      stream.reset();
      // show
      for (auto& entry_1 : entries) {
        auto name_1 = entry_1->path();
        auto size = entry_1->size();
        // use LOG(INFO) print all entry name and size
        LOG(INFO) << "Entry name: " << name_1 << ", size: " << size;
      }
      ASSERT_TRUE(true);
      LOG(INFO) << " ======== overwrite " << name << " ========";
    }
    // after the overwriting, the original file is corrupted.
    auto first_entry_name = "sample_src_tar/test_config.cpp";
    {
      LOG(INFO) << " ======== read " << first_entry_name
                << " and check ========";

      auto tarStream = std::make_unique<std::fstream>(
          destFileName, std::ios::binary | std::ios::in | std::ios::out);
      ASSERT_TRUE(tarStream->is_open())
          << "Failed to open tar file: " << destFileName
          << " Error opening file: " << std::strerror(errno);
      auto tar_file_obj = vaip_core::TarFile::create(std::move(tarStream));
      ASSERT_TRUE(tar_file_obj) << "Failed to create TarFile object";
      auto& entries = tar_file_obj->entries();
      ASSERT_TRUE(!entries.empty()) << "Failed to read tar entries";
      ASSERT_EQ(entries.size(), 3) << "Expected two tar entries";
      // clang-format off
    // the content after overwriting
    /*
md5                                      size   blk-begin     blk-end  data-begin    data-end path
618eac9918e0638aaa8b4d931a6ba357         2512           0         512         512        3024 sample_src_tar/test_config.cpp
8558330fe7b37b216f838296deb75257         6077        3072        3584        3584        9661 sample_src_tar/test_tarball.cpp
e807f1fcf82d132f9bb018ca6738a19f           10        9728       10240       10240       10250 _data/e807f1fcf82d132f9bb018ca6738a19f
e807f1fcf82d132f9bb018ca6738a19f           10       10752       11264       10240       10250 sample_src_tar/test_config.cpp -> _data/e807f1fcf82d132f9bb018ca6738a19f
    */
      // clang-format on
      auto stream = tar_file_obj->open_for_read(first_entry_name);
      ASSERT_TRUE(stream) << "Failed to open entry for read: "
                          << first_entry_name;
      std::string buffer;
      buffer.resize(const_test_content.size());
      stream->read(&buffer[0], const_test_content.size());
      ASSERT_TRUE(stream) << "Failed to read entry: " << first_entry_name;
      ASSERT_EQ(stream->gcount(), const_test_content.size())
          << "Entry size mismatch for entry: " << first_entry_name;
      ASSERT_STREQ(buffer.c_str(), const_test_content.c_str())
          << "Contents mismatch for entry: " << first_entry_name;
    }
  }
}
TEST(TarFileTest, WriteOverride) {
  //
  LOG(INFO) << " ======== test write override, write same data ========";
  test_write_override(true);
  //
  LOG(INFO) << " ======== test write override, write different data ========";
  test_write_override(false);
}
static void write_to_stream(const std::string& file, vaip_core::TarFile& tar,
                            const std::string& data) {
  LOG(INFO) << " start to write to stream. file=" << file;
  auto stream = tar.open_for_write(file);
  ASSERT_TRUE(stream) << " cannot create stream";
  LOG(INFO) << "before writing stream pos " << stream->tellp();
  stream->write(data.data(), data.size());
  stream->flush();
  LOG(INFO) << "after writing stream pos " << stream->tellp();
  ASSERT_TRUE(*stream) << "Failed to write to entry: a.txt";
  LOG(INFO) << "after writing " << file << ", stream pos " << stream->tellp();
}
static void read_and_check(const std::string& file, vaip_core::TarFile& tar,
                           const std::string& data) {
  LOG(INFO) << " start to read from stream. file=" << file;
  auto stream = tar.open_for_read(file);
  ASSERT_TRUE(stream) << " cannot create stream";
  auto buf = std::string(data.size(), ' ');
  stream->read(buf.data(), buf.size());
  ASSERT_TRUE(!stream->eof()) << "read failed.";
  ASSERT_EQ(stream->gcount(), buf.size())
      << "Entry size mismatch for entry: a.txt";
  EXPECT_EQ(buf, data) << "Contents mismatch for entry: a.txt";
  stream->read(buf.data(), buf.size());
  ASSERT_TRUE(stream->eof()) << "EOF test failed.";
  LOG(INFO) << " read from stream. file=" << file << " done";
}
template <typename T>
bool ExpectEqualAndReturn(const T& val1, const T& val2,
                          const std::string& comment) {
  bool result = (val1 == val2);
  EXPECT_EQ(val1, val2) << comment;
  return result;
}

static void expect_tar_entry(const vaip_core::TarEntryInputStream& entry,
                             const std::string& path,
                             const std::optional<std::string>& real_path,
                             size_t size, int block_begin, int block_end,
                             int data_begin, int data_end) {
  auto ok = true;
  ok = ExpectEqualAndReturn(entry.path(), path, "Path mismatch") && ok;
  ok = ExpectEqualAndReturn(entry.real_path(), real_path,
                            "Real path mismatch") &&
       ok;
  ok = ExpectEqualAndReturn(entry.size(), size, "Size mismatch") && ok;
  ok =
      ExpectEqualAndReturn(entry.block_begin_pos(), std::streampos(block_begin),
                           "Block begin mismatch") &&
      ok;
  ok = ExpectEqualAndReturn(entry.block_end_pos(), std::streampos(block_end),
                            "Block end mismatch") &&
       ok;
  ok = ExpectEqualAndReturn(entry.data_begin_pos(), std::streampos(data_begin),
                            "Data begin mismatch") &&
       ok;
  ok = ExpectEqualAndReturn(entry.data_end_pos(), std::streampos(data_end),
                            "Data end mismatch") &&
       ok;
  if (ok) {
    LOG(INFO) << "Entry: path=" << entry.path()
              << " real_path=" << entry.real_path().value_or("N/A")
              << " size=" << entry.size()
              << " block_begin_pos=" << entry.block_begin_pos()
              << " block_end_pos=" << entry.block_end_pos()
              << " data_begin_pos=" << entry.data_begin_pos()
              << " data_end_pos=" << entry.data_end_pos() //
              << " OK";
  }
}
TEST(TarFileTest, WriteTo) {
  // Assuming you have a tar file named "sample.src.tar" in the current
  // directory
  {
    auto tarFileName =
        CMAKE_CURRENT_BINARY_PATH / "written_by_tar_file_test_write_to.tar";
    auto tarStream = std::make_unique<std::fstream>(
        tarFileName,
        std::ios::binary | std::ios::in | std::ios::out | std::ios::trunc);
    /*std::string zeros(1024, '\0');
    tarStream->write(zeros.data(), zeros.size());
    ASSERT_TRUE(tarStream->good()) << "Failed to write zeros to tar file";*/
    ASSERT_TRUE(tarStream->is_open())
        << "Failed to open tar file: " << tarFileName
        << " Error opening file: " << std::strerror(errno);
    auto tar_file_obj = vaip_core::TarFile::create(std::move(tarStream));
    ASSERT_TRUE(tar_file_obj) << "Failed to create TarFile object";
    LOG(INFO) << " ====== begin to write to tar file. ==== ";
    { write_to_stream("a.txt", *tar_file_obj, "hello"); }
    { write_to_stream("b.txt", *tar_file_obj, "hello"); }
    { write_to_stream("c.txt", *tar_file_obj, "hello"); }
    { write_to_stream("a.txt", *tar_file_obj, "hello"); }
    // write to the same file with different data
    { write_to_stream("a.txt", *tar_file_obj, "world!"); }
    // write to the same file with same data
    { write_to_stream("b.txt", *tar_file_obj, "hello"); }
    auto& entries = tar_file_obj->entries();

    ASSERT_EQ(entries.size(), 5)
        << "Expected five tar entries, but got " << entries.size();

    expect_tar_entry(*entries[0], "_data/5d41402abc4b2a76b9719d911017c592",
                     std::nullopt, //
                     5, 0, 512, 512, 517);
    expect_tar_entry(*entries[1], "b.txt",
                     std::string("_data/5d41402abc4b2a76b9719d911017c592"), //
                     5, 1536, 2048, 512, 517);
    expect_tar_entry(*entries[2], "c.txt",
                     std::string("_data/5d41402abc4b2a76b9719d911017c592"), //
                     5, 2048, 2560, 512, 517);
    expect_tar_entry(*entries[3], "_data/08cf82251c975a5e9734699fadf5e9c0",
                     std::nullopt, 6, 2560, 3072, 3072, 3078);
    expect_tar_entry(*entries[4], "a.txt",
                     std::string("_data/08cf82251c975a5e9734699fadf5e9c0"), 6,
                     3584, 4096, 3072, 3078);
    LOG(INFO) << " ====== end write to tar file. ==== ";
    LOG(INFO) << " ====== begin to read and check ==== ";
    { read_and_check("a.txt", *tar_file_obj, "world!"); }
    { read_and_check("b.txt", *tar_file_obj, "hello"); }
    { read_and_check("c.txt", *tar_file_obj, "hello"); }
    LOG(INFO) << " ====== end to read and check ==== ";
  }
  //  run tar -tvf to check the result
  {
    auto tarFileName =
        CMAKE_CURRENT_BINARY_PATH / "written_by_tar_file_test_write_to.tar";
    auto tar_exe_path = boost::process::search_path("tar");
    if (tar_exe_path.empty()) {
      LOG(INFO) << "cannot find tar exe. cancel testing";
    } else {
      LOG(INFO) << "run " << tar_exe_path << " -tvf " << tarFileName.u8string();
      auto exit_code =
          boost::process::system(tar_exe_path, "-tvf", tarFileName.u8string());
      ASSERT_EQ(exit_code, 0)
          << "Failed to run tar command. Exit code: " << exit_code;
    }
    auto morphizen_tar_exe_path = boost::process::search_path("tar");
  }
  // frehs read.
  {
    LOG(INFO) << " =======================================";
    LOG(INFO) << " ======  start a fresh read ============";
    LOG(INFO) << " =======================================";
    auto tarFileName =
        CMAKE_CURRENT_BINARY_PATH / "written_by_tar_file_test_write_to.tar";
    auto tarStream = std::make_unique<std::fstream>(
        tarFileName, std::ios::binary | std::ios::in | std::ios::out);
    auto tar_file_obj = vaip_core::TarFile::create(std::move(tarStream));
    ASSERT_TRUE(tar_file_obj) << "Failed to create TarFile object";
    { read_and_check("a.txt", *tar_file_obj, "world!"); }
    { read_and_check("b.txt", *tar_file_obj, "hello"); }
    { read_and_check("c.txt", *tar_file_obj, "hello"); }
    LOG(INFO) << " =======================================";
    LOG(INFO) << " ======  fresh read is done ============";
    LOG(INFO) << " =======================================";
  }
}
