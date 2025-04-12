
extern "C" {
#include "getopt.h"
}
#include "../vaip-core/src/tar_file.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
int main(int argc, char* argv[]) {
  int opt = 0;
  int option_index = 0;
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
    auto& entries = tar_file_obj.entries();
        std::cout << std::left // Align fields to the left
              << std::setw(15) << "Size"           // Set width for "Size"
              << std::setw(15) << "Block Begin"    // Set width for "Block Begin"
              << std::setw(15) << "Block End"      // Set width for "Block End"
              << std::setw(15) << "Data Begin"     // Set width for "Data Begin"
              << std::setw(15) << "Data End"       // Set width for "Data End"
              << std::setw(30) << "Path"           // Set width for "Path"
              << std::endl;

    for (const auto& entry : entries) {
        std::cout << std::left // Align fields to the left
                  << std::setw(15) << entry->size()              // File size
                  << std::setw(15) << entry->block_begin_pos()   // Block begin position
                  << std::setw(15) << entry->block_end_pos()     // Block end position
                  << std::setw(15) << entry->data_begin_pos()    // Data begin position
                  << std::setw(15) << entry->data_end_pos()      // Data end position
                  << std::setw(30) << entry->path()              // File path
                  << (!entry->is_symlink()
                          ? std::string("")
                          : std::string("->") + entry->real_path().value()) // Symlink info
                  << std::endl;
    }
  } else {
    std::cerr << "Usage: " << argv[0] << " [-l] <file>" << std::endl;
    return 1;
  }
  return 0;
}
#include "getopt.c"
