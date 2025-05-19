/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

extern "C" {
#include "getopt.h"
}
#include "../vaip-core/src/tar_file.hpp"
#include "../vaip-core/src/tar_header.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
using std::cerr;
using std::cout;
using std::endl;
template <typename T> static std::string get_readable_path(const T& entry) {
  return (entry.path() + (!entry.is_symlink() ? std::string("")
                                              : std::string(" -> ") +
                                                    entry.real_path().value()));
}
static int list_tar(const char* file) {
  auto tar_path = std::filesystem::u8path(file);
  auto tar_file_obj = vaip_core::TarFile::create(tar_path);
  if (tar_file_obj == nullptr) {
    std::cerr << "Failed to open tar file: " << file << std::endl;
    return 1;
  }
  auto width = 12;
  std::cout << std::left                        // Align fields to the left
            << std::setw(33) << "md5"           // Set width for "MD5 checksum"
            << std::right                       // Align fields to the right
            << std::setw(width) << "size"       // Set width for "Size"
            << std::setw(width) << "blk-begin"  // Set width for "Block Begin"
            << std::setw(width) << "blk-end"    // Set width for " Block End "
            << std::setw(width) << "data-begin" // Set width for "Data Begin"
            << std::setw(width) << "data-end"   // Set width for " Data End "
            << std::left                        //
            << std::setw(30) << " path"         // Set width for "Path"
            << std::endl;
  auto& entries = tar_file_obj->entries();
  for (const auto& entry : entries) {
    auto md5 = entry->md5();
    auto name = get_readable_path(*entry);
    std::cout << std::left                         // Align fields to the left
              << std::setw(33) << md5              // MD5 checksum
              << std::right                        // Align fields to the right
              << std::setw(width) << entry->size() // File size
              << std::setw(width)
              << entry->block_begin_pos() // Block begin position
              << std::setw(width)
              << entry->block_end_pos() // Block end position
              << std::setw(width)
              << entry->data_begin_pos() // Data begin position
              << std::setw(width) << entry->data_end_pos() // Data end position
              << std::left    // Align fields to the left
              << std::setw(0) // File path
              << (" " + name) // path info
              << std::endl;
  }
  return 0;
}
static int list_header_tar(const char* file) {
  auto tar_path = std::filesystem::u8path(file);
  auto stream =
      std::make_unique<std::fstream>(tar_path, std::ios::binary | std::ios::in);
  auto entry = vaip_core::TarHeader::read_header(*stream);
  auto width = 12;
  std::cout << std::right                       // Align fields to the right
            << std::setw(width) << "size"       // Set width for "Size"
            << std::setw(width) << "blk-begin"  // Set width for "Block Begin"
            << std::setw(width) << "blk-end"    // Set width for " Block End "
            << std::setw(width) << "data-begin" // Set width for "Data Begin"
            << std::setw(width) << "data-end"   // Set width for " Data End "
            << std::left                        //
            << std::setw(30) << " path"         // Set width for "Path"
            << std::endl;
  while (entry) {
    auto name = get_readable_path(*entry);
    std::cout << std::left                         // Align fields to the left
              << std::right                        // Align fields to the right
              << std::setw(width) << entry->size() // File size
              << std::setw(width)
              << entry->block_begin_pos() // Block begin position
              << std::setw(width)
              << entry->block_end_pos() // Block end position
              << std::setw(width)
              << entry->data_begin_pos() // Data begin position
              << std::setw(width) << entry->data_end_pos() // Data end position
              << std::left    // Align fields to the left
              << std::setw(0) // File path
              << (" " + name) // path info
              << std::endl;
    entry = vaip_core::TarHeader::read_header(*stream);
  }
  stream->clear();
  cout << "read end at " << stream->tellg() << endl;
  stream->seekg(0, std::ios::end);
  auto file_size = stream->tellg();
  cout << " file size: " << file_size << endl;
  return 0;
}
int main(int argc, char* argv[]) {
  int opt = 0;
  int list_flag = 0;
  int header_flag = 0;
  const char* file = nullptr;
  while ((opt = getopt(argc, argv, "lh")) != -1) {
    switch (opt) {
    case 'l': {
      list_flag = 1;
      break;
    }
    case 'h': {
      header_flag = 1;
      break;
    }
    default: {
      std::cerr << "Usage: " << argv[0] << " [-l]" << std::endl;
      return 1;
    }
    }
  }
  if (optind < argc) {
    file = argv[optind];
  } else {
    std::cerr << "Usage: " << argv[0] << " [-l] <file>" << std::endl;
    return 1;
  }
  if (list_flag) {
    list_tar(file);
  } else if (header_flag) {
    list_header_tar(file);
  } else {
    std::cerr << "Usage: " << argv[0] << " [-l] <file>" << std::endl;
    return 1;
  }
  return 0;
}
#include "getopt.c"
