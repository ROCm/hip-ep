/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once
#include "morphizen/vaip_io.hpp"
#include <string>

#ifndef VAIP_DLL_SPEC
#  if defined(_WIN32) || defined(_WIN64)
#    define VAIP_DLL_SPEC __declspec(dllexport)
#  else
#    define VAIP_DLL_SPEC __attribute__((visibility("default")))
#  endif
#endif
namespace vaip_core {
class TarWriter {
public:
  TarWriter(IStreamWriter& tall_ball_writer) : tarball_(tall_ball_writer) {}
  VAIP_DLL_SPEC int write(const IStreamReader& src, const std::string& name);
  VAIP_DLL_SPEC ~TarWriter();

private:
  int write_internal(const IStreamReader& src, const std::string& name,
                     size_t size);

private:
  IStreamWriter& tarball_;
};
class TarReader {
public:
  TarReader(const IStreamReader& tall_ball_reader)
      : tarball_(tall_ball_reader) {}
  VAIP_DLL_SPEC int read(IStreamWriterBuilder& dst_builder);

private:
  const IStreamReader& tarball_;
};
} // namespace vaip_core

#include <filesystem>
#include <iostream>
#include <memory>
#include <vector>
namespace vaip_tar {
class TarEntry : public std::enable_shared_from_this<TarEntry> {
public:
  static std::shared_ptr<TarEntry> create_from_mem(std::vector<char>&&);
  std::string get_name();
  // not support long filename
  int rename(const std::string& name);
  // Return to the starting point of the entire entry(header included)
  const char* entry_data();
  // Return to the size of the entire entry(header included)
  size_t entry_size();
  // starting point of real data
  const char* data();
  // size of real data
  size_t size();

private:
  TarEntry(std::vector<char>&&);
  std::vector<char> datas;
};

class TarFile {
public:
  TarFile(const std::filesystem::path& file);
  int save(bool overwrite = true);
  int append(std::shared_ptr<TarEntry>);

  auto begin() { return entries.begin(); }
  auto end() { return entries.end(); }
  auto begin() const { return entries.cbegin(); }
  auto end() const { return entries.cend(); }

private:
  TarFile(const TarFile&) = delete;
  TarFile& operator=(const TarFile&) = delete;
  std::filesystem::path file_path;
  std::vector<std::shared_ptr<TarEntry>> entries;
};
} // namespace vaip_tar