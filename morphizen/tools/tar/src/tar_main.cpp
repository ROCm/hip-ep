/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "../morphizen-core/src/tar_file.hpp"
#include "../morphizen-core/src/tar_header.hpp"
#include <boost/program_options.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace po = boost::program_options;
using std::cerr;
using std::cout;
using std::endl;

// Validate file path to mitigate path traversal risks
static bool validate_path(const std::string &path,
                          const std::string &param_name) {
  if (path.empty()) {
    return false;
  }

  // Check for null bytes (potential for path truncation attacks)
  if (path.find('\0') != std::string::npos) {
    std::cerr << "Error: " << param_name << " contains null byte: " << path
              << std::endl;
    return false;
  }

  // Warn about suspicious patterns but don't block them (users may legitimately
  // need them)
  if (path.find("..") != std::string::npos) {
    std::cerr << "Warning: " << param_name << " contains '..': " << path
              << std::endl;
  }

  return true;
}

template <typename T> static std::string get_readable_path(const T &entry) {
  return (entry.path() + (!entry.is_symlink() ? std::string("")
                                              : std::string(" -> ") +
                                                    entry.real_path().value()));
}
static int list_tar(const char *file) {
  auto tar_path = std::filesystem::u8path(file);
  auto tar_file_obj = morphizen::TarFile::create_from_path(tar_path);
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
  auto &entries = tar_file_obj->entries();
  for (const auto &entry : entries) {
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
static int list_header_tar(const char *file) {
  auto tar_path = std::filesystem::u8path(file);
  auto stream =
      std::make_unique<std::fstream>(tar_path, std::ios::binary | std::ios::in);
  auto entry = morphizen::TarHeader::read_header(*stream);
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
    entry = morphizen::TarHeader::read_header(*stream);
  }
  stream->clear();
  cout << "read end at " << stream->tellg() << endl;
  stream->seekg(0, std::ios::end);
  auto file_size = stream->tellg();
  cout << " file size: " << file_size << endl;
  return 0;
}
int main(int argc, char *argv[]) {
  // Define command line options
  po::options_description desc("Allowed options");
  desc.add_options()("help", "produce help message")(
      "list,l", "list tar file contents")("header,h", "list tar headers")(
      "file", po::value<std::string>(), "tar file to process");

  po::positional_options_description p;
  p.add("file", 1);

  po::variables_map vm;
  po::store(
      po::command_line_parser(argc, argv).options(desc).positional(p).run(),
      vm);
  po::notify(vm);

  // Handle help option
  if (vm.count("help") || !vm.count("file")) {
    std::cout << "Usage: " << argv[0] << " [options] <file>\n"
              << desc << std::endl;
    return vm.count("help") ? 0 : 1;
  }

  // Extract option values
  auto file_str = vm["file"].as<std::string>();
  bool list_flag = vm.count("list") > 0;
  bool header_flag = vm.count("header") > 0;

  // Validate path at entry point
  if (!validate_path(file_str, "file")) {
    return 1;
  }

  if (list_flag) {
    list_tar(file_str.c_str());
  } else if (header_flag) {
    list_header_tar(file_str.c_str());
  } else {
    std::cerr << "Usage: " << argv[0] << " [--list | --header] <file>"
              << std::endl;
    return 1;
  }
  return 0;
}
