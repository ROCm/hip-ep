/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "morphizen/xclbin_file.hpp"
#include <fstream>
#include <glog/logging.h>
#include <vector>
#define _KERNEL_MODE
#include "xclbin.h"

namespace vaip_core {
uint64_t get_fingerprint(std::istream& file,
                         const axlf_section_header& section_hdr) {
  file.seekg(section_hdr.m_sectionOffset);
  aie_partition section;
  file.read((char*)&section, sizeof(section));
  return section.inference_fingerprint;
}

axlf_section_header get_section_hdr(std::istream& file, const axlf& file_hdr,
                                    axlf_section_kind kind) {
  auto section_count = file_hdr.m_header.m_numSections;
  for (unsigned int section_idx = 0; section_idx < section_count;
       ++section_idx) {
    axlf_section_header hdr;
    long long offset =
        sizeof(file_hdr) + (section_idx * sizeof(hdr)) - sizeof(hdr);
    file.seekg(offset);

    file.read((char*)&hdr, sizeof(axlf_section_header));
    if (file.gcount() != sizeof(axlf_section_header)) {
      std::string error = "xclbin section header too small";
      throw std::runtime_error(error);
    }
    if (hdr.m_sectionKind == static_cast<decltype(hdr.m_sectionKind)>(kind)) {
      return hdr;
    }
  }
  std::string error =
      std::string("section not found for kind ") + std::to_string(kind);
  throw std::runtime_error(error);
}

axlf get_hdr(std::istream& file, const std::string& filename) {
  file.seekg(0);
  axlf hdr;
  file.read((char*)&hdr, sizeof(hdr));

  if (file.gcount() != sizeof(hdr)) {
    std::string error = "xclbin header too small " + filename;
    throw std::runtime_error(error);
  }

  if (hdr.m_magic != std::string("xclbin2")) {
    // axlf::m_magic is a char array. Ref.:
    // https://github.com/Xilinx/XRT/blob/347acbd5e2b2d658ecd21d024547703fccc5572c/src/runtime_src/core/include/xclbin.h#L250
    std::string error = "xclbin magic (" + std::string(hdr.m_magic) +
                        ") mismatched " + filename;
    throw std::runtime_error(error);
  }
  return hdr;
}

std::optional<uint64_t>
get_xclbin_fingerprint(const vaip_core::PassContext& pass_context,
                       const std::filesystem::path& filename) {
  auto basename = filename.filename();
  auto stream = std::unique_ptr<std::istream>();
  auto buffer = std::string();
  auto content = pass_context.read_file_c8(basename.u8string());
  if (!content.has_value()) {
    LOG(ERROR) << "Failed to open xclbin " << filename;
    return std::nullopt;
  }
  buffer.assign(content->begin(), content->end());
  auto p = new std::istringstream(buffer);
  stream = std::unique_ptr<std::istream>(p);
  auto hdr = get_hdr(*stream, filename.u8string());
  auto section_hdr = get_section_hdr(*stream, hdr, AIE_PARTITION);
  auto fingerprint = get_fingerprint(*stream, section_hdr);
  return fingerprint;
}

} // namespace vaip_core
