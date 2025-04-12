
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
    for (const auto& entry : entries) {
      std::cout << entry->path()
                << (!entry->is_symlink()
                        ? std::string("")
                        : std::string("->") + entry->real_path().value()) //
                << "\t" << entry->size()                                  //
                << std::endl;
    }
  } else {
    std::cerr << "Usage: " << argv[0] << " [-l] <file>" << std::endl;
    return 1;
  }
  return 0;
}
#include "getopt.c"
