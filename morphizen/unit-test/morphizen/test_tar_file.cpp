/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#define _CRT_SECURE_NO_WARNINGS
#include "../../morphizen-core/src/tar_file.hpp"
#include "morphizen/dll_safe.h"
#include "test_environment.hpp"
#ifdef MORPHIZEN_ENABLE_BOOST
#include <boost/process.hpp>
#endif
#include <cerrno>
#include <cstring>
#include <fstream>
#include <glog/logging.h>
#include <gtest/gtest.h>
TEST(TarFileTest, ReadFrom) {
  auto tarFileName = SAMPLE_SRC_TAR_PATH;
  auto tarStream = std::make_unique<std::fstream>(
      tarFileName, std::ios::binary | std::ios::in);
  auto tar_file_obj = morphizen::TarFile::create(std::move(tarStream));
  ASSERT_TRUE(tar_file_obj) << "Failed to create TarFile object";
  for (auto &entry : tar_file_obj->entries()) {
    LOG(INFO) << "entry: " << entry->path()
              << (entry->real_path()
                      ? std::string("->") + entry->real_path().value()
                      : "")
              << " size=" << entry->size();
  }
}

TEST(TarFileTest, DoubleRead) {
  auto tarFileName = SAMPLE_SRC_TAR_PATH;

  auto tarStream = std::make_unique<std::fstream>(
      tarFileName, std::ios::binary | std::ios::in);
  ASSERT_TRUE(tarStream->is_open())
      << "Failed to open tar file: " << tarFileName
      << " Error opening file: " << std::strerror(errno);
  auto tar_file_obj = morphizen::TarFile::create(std::move(tarStream));
  ASSERT_TRUE(tar_file_obj) << "Failed to create TarFile object";
  auto &entries = tar_file_obj->entries();
  ASSERT_TRUE(!entries.empty()) << "Failed to read tar entries";
  ASSERT_EQ(entries.size(), 2) << "Expected two tar entries";
  // read first entry twice
  auto &entry = entries[0];
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
    auto srcFileName = SAMPLE_SRC_TAR_PATH;
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
      auto tar_file_obj = morphizen::TarFile::create(std::move(tarStream));
      ASSERT_TRUE(tar_file_obj) << "Failed to create TarFile object";
      auto &entries = tar_file_obj->entries();
      ASSERT_TRUE(!entries.empty()) << "Failed to read tar entries";
      ASSERT_EQ(entries.size(), 2) << "Expected two tar entries";
      // read first entry twice
      auto &entry = entries[0];
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
      for (auto &entry_1 : entries) {
        auto name_1 = entry_1->path();
        auto size = entry_1->size();
        // use LOG(INFO) print all entry name and size
        LOG(INFO) << "Entry name: " << name_1 << ", size: " << size;
      }
      ASSERT_TRUE(true);
      LOG(INFO) << " ======== overwrite " << name << " ========";
    }
    // after the overwriting, the original file is corrupted.
    auto first_entry_name = "sample_src_tar/hello.txt";
    {
      LOG(INFO) << " ======== read " << first_entry_name
                << " and check ========";

      auto tarStream = std::make_unique<std::fstream>(
          destFileName, std::ios::binary | std::ios::in | std::ios::out);
      ASSERT_TRUE(tarStream->is_open())
          << "Failed to open tar file: " << destFileName
          << " Error opening file: " << std::strerror(errno);
      auto tar_file_obj = morphizen::TarFile::create(std::move(tarStream));
      ASSERT_TRUE(tar_file_obj) << "Failed to create TarFile object";
      auto &entries = tar_file_obj->entries();
      ASSERT_TRUE(!entries.empty()) << "Failed to read tar entries";

      ASSERT_EQ(entries.size(), 3);

      // clang-format off
    // the content after overwriting fot same data  write_same_data == true
    /*
        size   blk-begin     blk-end  data-begin    data-end path
           6           0         512         512         518 sample_src_tar/hello.txt
           9        1024        1536        1536        1545 sample_src_tar/tar_file.txt
           6        2048        2560        2560        2566 _data/b1946ac92492d2347c6235b4d2611184
           0        3072        3584        3584        3584 sample_src_tar/hello.txt

    // the content after overwriting fot same data  write_same_data == false
        size   blk-begin     blk-end  data-begin    data-end path
           6           0         512         512         518 sample_src_tar/hello.txt
           9        1024        1536        1536        1545 sample_src_tar/tar_file.txt
           6        2048        2560        2560        2566 _data/b1946ac92492d2347c6235b4d2611184
           0        3072        3584        3584        3584 sample_src_tar/hello.txt
 $ for i in 1 2 3 4 5; do echo $i $((i*4096 - 512)); done
1 3584
2 7680
3 11776
4 15872
5 19968
*/ // clang-format on
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
static void write_to_stream(const std::string &file, morphizen::TarFile &tar,
                            const std::string &data) {
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
static void read_and_check(const std::string &file, morphizen::TarFile &tar,
                           const std::string &data) {
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
bool ExpectEqualAndReturn(const T &val1, const T &val2,
                          const std::string &comment) {
  bool result = (val1 == val2);
  EXPECT_EQ(val1, val2) << comment;
  return result;
}

static void expect_tar_entry(const morphizen::TarEntryInputStream &entry,
                             const std::string &path,
                             const std::optional<std::string> &real_path,
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

static void
check_entries(const std::vector<std::unique_ptr<morphizen::TarEntryInputStream>>
                  &entries) {
  ASSERT_EQ(entries.size(), 5)
      << "Expected five tar entries, but got " << entries.size();
  // clang-format off
    /*
md5                                      size   blk-begin     blk-end  data-begin    data-end path
5d41402abc4b2a76b9719d911017c592            5        3584        4096        4096        4101 _data/5d41402abc4b2a76b9719d911017c592
5d41402abc4b2a76b9719d911017c592            5        7680        8192        4096        4101 b.txt -> _data/5d41402abc4b2a76b9719d911017c592
5d41402abc4b2a76b9719d911017c592            5       11776       12288        4096        4101 c.txt -> _data/5d41402abc4b2a76b9719d911017c592
08cf82251c975a5e9734699fadf5e9c0            6       15872       16384       16384       16390 _data/08cf82251c975a5e9734699fadf5e9c0
08cf82251c975a5e9734699fadf5e9c0            6       16896       17408       16384       16390 a.txt -> _data/08cf82251c975a5e9734699fadf5e9c0
	*/
  // clang-format on
  expect_tar_entry(*entries[0], "_data/5d41402abc4b2a76b9719d911017c592",
                   std::nullopt, //
                   5, 3584, 4096, 4096, 4101);
  expect_tar_entry(*entries[1], "b.txt",
                   std::string("_data/5d41402abc4b2a76b9719d911017c592"), //
                   5, 7680, 8192, 4096, 4101);
  expect_tar_entry(*entries[2], "c.txt",
                   std::string("_data/5d41402abc4b2a76b9719d911017c592"), //
                   5, 11776, 12288, 4096, 4101);
  expect_tar_entry(*entries[3], "_data/08cf82251c975a5e9734699fadf5e9c0",
                   std::nullopt, //
                   6, 15872, 16384, 16384, 16390);
  expect_tar_entry(*entries[4], "a.txt",
                   std::string("_data/08cf82251c975a5e9734699fadf5e9c0"), //
                   6, 16896, 17408, 16384, 16390);
}
static void check_abc(morphizen::TarFile &tar_file_obj) {
  check_entries(tar_file_obj.entries());
  { read_and_check("a.txt", tar_file_obj, "world!"); }
  { read_and_check("b.txt", tar_file_obj, "hello"); }
  { read_and_check("c.txt", tar_file_obj, "hello"); }
}
static void test_abc(morphizen::TarFile &tar_file_obj) {
  LOG(INFO) << " ====== begin to write to tar file. ==== ";
  { write_to_stream("a.txt", tar_file_obj, "hello"); }
  { write_to_stream("b.txt", tar_file_obj, "hello"); }
  { write_to_stream("c.txt", tar_file_obj, "hello"); }
  { write_to_stream("a.txt", tar_file_obj, "hello"); }
  // write to the same file with different data
  { write_to_stream("a.txt", tar_file_obj, "world!"); }
  // write to the same file with same data
  { write_to_stream("b.txt", tar_file_obj, "hello"); }
  LOG(INFO) << " ====== end write to tar file. ==== ";
  LOG(INFO) << " ====== begin to read and check ==== ";
  check_abc(tar_file_obj);
  LOG(INFO) << " ====== end to read and check ==== ";
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
    auto tar_file_obj = morphizen::TarFile::create(std::move(tarStream));
    ASSERT_TRUE(tar_file_obj) << "Failed to create TarFile object";
    test_abc(*tar_file_obj);
  }
  //  run tar -tvf to check the result
#ifdef MORPHIZEN_ENABLE_BOOST
  {
    auto tarFileName =
        CMAKE_CURRENT_BINARY_PATH / "written_by_tar_file_test_write_to.tar";
    auto tar_exe_path = boost::process::search_path("tar");
    if (tar_exe_path.empty()) {
      LOG(INFO) << "cannot find tar exe. cancel testing";
    } else {
      LOG(INFO) << "run " << tar_exe_path << " -tvf " << tarFileName.u8string();
      // on Windows, "tar -tvf C:/..." results in an error.
      // /usr/bin/tar: Cannot connect to C: resolve failed
      // because tar is part of git. it cannnot recognize windows drive C:
      auto exit_code = boost::process::system(
          tar_exe_path, "-tvf", tarFileName.filename().u8string(),
          boost::process::start_dir(CMAKE_CURRENT_BINARY_PATH.u8string()));
      ASSERT_EQ(exit_code, 0)
          << "Failed to run tar command. Exit code: " << exit_code;
    }
  }
#else
  GTEST_SKIP()
      << "Boost::Process not available (morphizen_ENABLE_BOOST is OFF)";
#endif
  // frehs read.
  {
    LOG(INFO) << " =======================================";
    LOG(INFO) << " ======  start a fresh read ============";
    LOG(INFO) << " =======================================";
    auto tarFileName =
        CMAKE_CURRENT_BINARY_PATH / "written_by_tar_file_test_write_to.tar";
    auto tarStream = std::make_unique<std::fstream>(
        tarFileName, std::ios::binary | std::ios::in | std::ios::out);
    auto tar_file_obj = morphizen::TarFile::create(std::move(tarStream));
    ASSERT_TRUE(tar_file_obj) << "Failed to create TarFile object";
    check_abc(*tar_file_obj);
    LOG(INFO) << " =======================================";
    LOG(INFO) << " ======  fresh read is done ============";
    LOG(INFO) << " =======================================";
  }
}
TEST(TarFileTest, MemoryTar) {
  // Helper to create buffer with tar data
  auto create_buf = []() {
    auto tar_file_obj = morphizen::TarFile::create_from_tmpfile();
    CHECK(tar_file_obj) << "Failed to create TarFile object";
    test_abc(*tar_file_obj);
    auto size = tar_file_obj->current_size();
    std::vector<char> buf(size);
    CHECK_GE(size, 0);
    tar_file_obj->dump_to(buf.data(), buf.size());
    return buf;
  };

  {
    LOG(INFO) << " ==========================================================";
    LOG(INFO) << " ======  start a fresh write and check to tmpfile =========";
    LOG(INFO) << " ==========================================================";
    auto tar_file_obj = morphizen::TarFile::create_from_tmpfile();
    ASSERT_TRUE(tar_file_obj) << "Failed to create TarFile object";
    test_abc(*tar_file_obj);
    check_abc(*tar_file_obj);
    LOG(INFO) << " ==========================================================";
    LOG(INFO) << " ======  a fresh write and check to tmpfile is done =======";
    LOG(INFO) << " ==========================================================";
  }
  {
    LOG(INFO) << " ==========================================================";
    LOG(INFO) << " ======  start a fresh read on vector buffer ==============";
    LOG(INFO) << " ==========================================================";
    auto buf = create_buf();
    auto tar_file_obj = morphizen::TarFile::create_from_buffer(std::move(buf));
    ASSERT_TRUE(tar_file_obj) << "Failed to create TarFile object";
    check_abc(*tar_file_obj);
    LOG(INFO) << " =======================================";
    LOG(INFO) << " ======  fresh read is done ============";
    LOG(INFO) << " =======================================";
  }
  {
    LOG(INFO) << " ==========================================================";
    LOG(INFO)
        << " ======  start a fresh read on dllsafe buffer ===============";
    LOG(INFO) << " ==========================================================";
    auto buf = create_buf();
    std::string new_buf(buf.begin(), buf.end());
    auto tar_file_obj =
        morphizen::TarFile::create_from_buffer(std::move(new_buf));
    ASSERT_TRUE(tar_file_obj) << "Failed to create TarFile object";
    check_abc(*tar_file_obj);
    LOG(INFO) << " =======================================";
    LOG(INFO) << " ======  fresh read is done ============";
    LOG(INFO) << " =======================================";
  }
  {
    LOG(INFO) << " ==========================================================";
    LOG(INFO) << " ======  start a fresh read on owned buffer ===============";
    LOG(INFO) << " ==========================================================";
    auto buf = create_buf();
    auto tar_file_obj = morphizen::TarFile::create_from_buffer(std::move(buf));
    ASSERT_TRUE(tar_file_obj) << "Failed to create TarFile object";
    check_abc(*tar_file_obj);
    LOG(INFO) << " =======================================";
    LOG(INFO) << " ======  fresh read is done ============";
    LOG(INFO) << " =======================================";
  }
}
