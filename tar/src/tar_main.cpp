/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

extern "C" {
#include "getopt.h"
}
#include "../vaip-core/src/tar_file.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
static std::string
get_readable_path(const vaip_core::TarEntryInputStream& entry);

int main(int argc, char* argv[]) {
  int opt = 0;
  int list_flag = 0;
  const char* file = nullptr;
  while ((opt = getopt(argc, argv, "l")) != -1) {
    switch (opt) {
    case 'l': {
      list_flag = 1;
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
    std::filesystem::path tar_path(file);
    auto tar_stream = std::make_unique<std::fstream>(
        tar_path, std::ios::binary | std::ios::in | std::ios::out);
    if (!tar_stream->is_open()) {
      std::cerr << "Failed to open tar file: " << file << std::endl;
      return 1;
    }
    auto tar_file_obj = vaip_core::TarFile(std::move(tar_stream));
    auto width = 12;
    std::cout << std::left                       // Align fields to the left
              << std::setw(33) << "md5"          // Set width for "MD5 checksum"
              << std::right                      // Align fields to the right
              << std::setw(width) << "size"      // Set width for "Size"
              << std::setw(width) << "blk-begin" // Set width for "Block Begin"
              << std::setw(width) << "blk-end"   // Set width for " Block End "
              << std::setw(width) << "data-begin" // Set width for "Data Begin"
              << std::setw(width) << "data-end"   // Set width for " Data End "
              << std::left                        //
              << std::setw(30) << " path"         // Set width for "Path"
              << std::endl;
    auto& entries = tar_file_obj.entries();
    for (const auto& entry : entries) {
      auto md5 = entry->md5();
      auto name = get_readable_path(*entry);
      std::cout << std::left            // Align fields to the left
                << std::setw(33) << md5 // MD5 checksum
                << std::right           // Align fields to the right
                << std::setw(width) << entry->size() // File size
                << std::setw(width)
                << entry->block_begin_pos()          // Block begin position
                << std::setw(width)
                << entry->block_end_pos()            // Block end position
                << std::setw(width)
                << entry->data_begin_pos()           // Data begin position
                << std::setw(width)
                << entry->data_end_pos()             // Data end position
                << std::left                         // Align fields to the left
                << std::setw(0)                      // File path
                << (" " + name)                      // path info
                << std::endl;
    }
  } else {
    std::cerr << "Usage: " << argv[0] << " [-l] <file>" << std::endl;
    return 1;
  }
  return 0;
}
static std::string
get_readable_path(const vaip_core::TarEntryInputStream& entry) {
  return (entry.path() + (!entry.is_symlink() ? std::string("")
                                              : std::string(" -> ") +
                                                    entry.real_path().value()));
}
#include "getopt.c"
