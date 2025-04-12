#define _CRT_SECURE_NO_WARNINGS
#include "../../vaip-core/src/tar_file.hpp"
#include "debug_logger.hpp"
#include <cerrno>
#include <cstring>
#include <gtest/gtest.h>
TEST(TarFileTest, ReadFrom) {
  // Assuming you have a tar file named "sample.src.tar" in the current
  // directory
  {
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
    for (const auto& entry : entries) {
      auto name = entry->path();
      auto size = entry->size();
      auto stream =
          std::ifstream(CMAKE_CURRENT_BINARY_PATH / name, std::ios::binary);
      ASSERT_TRUE(stream.is_open()) << "Failed to open entry file: " << name;
      stream.seekg(0, std::ios::end);
      auto fileSize = stream.tellg();
      ASSERT_EQ(fileSize, size) << "File size mismatch for entry: " << name;

      ASSERT_TRUE(stream.seekg(0, std::ios::beg).good())
          << "Failed to seek to the beginning of the stream";
      std::string buffer;
      buffer.resize(size);
      stream.read(&buffer[0], size);
      ASSERT_TRUE(stream) << "Failed to read entry file: " << name;
      ASSERT_EQ(stream.gcount(), size)
          << "stream size mismatch for entry: " << name;
      // Compare the contents of the entry with the file
      std::string entryBuffer;
      entryBuffer.resize(size);
      entry->read(&entryBuffer[0], size);

      ASSERT_TRUE(entry) << "Failed to read entry: " << name;
      ASSERT_EQ(entry->gcount(), size)
          << "Entry size mismatch for entry: " << name;
      ASSERT_EQ(buffer, entryBuffer) << "Contents mismatch for entry: " << name;
      stream.close();
    }
    for (auto& entry : entries) {
      auto name = entry->path();
      auto size = entry->size();
      auto stream = tar_file_obj->open_for_read(name);
      ASSERT_TRUE(stream) << "Failed to open entry for read: " << entry->path();
      std::string buffer;
      buffer.resize(entry->size());
      stream->read(&buffer[0], entry->size());
      ASSERT_TRUE(stream) << "Failed to read entry: " << entry->path();
      ASSERT_EQ(stream->gcount(), entry->size())
          << "Entry size mismatch for entry: " << entry->path();
      auto raw_stream =
          std::ifstream(CMAKE_CURRENT_BINARY_PATH / name, std::ios::binary);
      ASSERT_TRUE(raw_stream.is_open()) << "Failed to open entry file: "
                                        << (CMAKE_CURRENT_BINARY_PATH / name);
      raw_stream.seekg(0, std::ios::beg);
      std::string raw_buffer;
      raw_buffer.resize(size);
      raw_stream.read(&raw_buffer[0], size);
      ASSERT_EQ(raw_stream.gcount(), size)
          << "raw_stream size mismatch for entry: " << name;
      // Compare the contents of the entry with the file
      ASSERT_STREQ(buffer.c_str(), raw_buffer.c_str())
          << "Contents mismatch for entry: " << name;
    }
  }
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
    { write_to_stream("a.txt", *tar_file_obj, "world"); }
    LOG(INFO) << " ====== end write to tar file. ==== ";
    LOG(INFO) << " ====== begin to read and check ==== ";
    { read_and_check("a.txt", *tar_file_obj, "world"); }
    { read_and_check("b.txt", *tar_file_obj, "hello"); }
    { read_and_check("c.txt", *tar_file_obj, "hello"); }
    LOG(INFO) << " ====== end to read and check ==== ";
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
    { read_and_check("a.txt", *tar_file_obj, "world"); }
    { read_and_check("b.txt", *tar_file_obj, "hello"); }
    { read_and_check("c.txt", *tar_file_obj, "hello"); }
    LOG(INFO) << " =======================================";
    LOG(INFO) << " ======  fresh read is done ============";
    LOG(INFO) << " =======================================";
  }
}
