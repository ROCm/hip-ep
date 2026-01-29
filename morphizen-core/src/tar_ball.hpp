/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once
#include <functional>
#include <iosfwd>
#include <string>

#ifndef MORPHIZEN_DLL_SPEC
#  if defined(_WIN32) || defined(_WIN64)
#    define MORPHIZEN_DLL_SPEC __declspec(dllexport)
#  else
#    define MORPHIZEN_DLL_SPEC __attribute__((visibility("default")))
#  endif
#endif
namespace morphizen {
class TarWriter {
public:
  TarWriter(std::ostream& tall_ball_writer) : tarball_(tall_ball_writer) {}
  MORPHIZEN_DLL_SPEC int write(std::istream& src, const std::string& name);
  MORPHIZEN_DLL_SPEC ~TarWriter();

private:
  int write_internal(std::istream& src, const std::string& name, size_t size);

private:
  std::ostream& tarball_;
};
class TarReader {
public:
  TarReader(std::istream& tall_ball_reader) : tarball_(tall_ball_reader) {}
  // Builder pattern: function returns ostream& for given filename
  MORPHIZEN_DLL_SPEC int
  read(std::function<std::ostream&(const std::string&)> dst_builder);

private:
  std::istream& tarball_;
};
} // namespace morphizen

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
