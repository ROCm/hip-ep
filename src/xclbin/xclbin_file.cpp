/*
 *     The Xilinx Vitis AI Vaip in this distribution are provided under the
 * following free and permissive binary-only license, but are not provided in
 * source code form.  While the following free and permissive license is similar
 * to the BSD open source license, it is NOT the BSD open source license nor
 * other OSI-approved open source license.
 *
 *      Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights
 * reserved.
 *
 *      Redistribution and use in binary form only, without modification, is
 * permitted provided that the following conditions are met:
 *
 *      1. Redistributions must reproduce the above copyright notice, this list
 * of conditions and the following disclaimer in the documentation and/or other
 * materials provided with the distribution.
 *
 *      2. The name of Xilinx, Inc. may not be used to endorse or promote
 * products redistributed with this software without specific prior written
 * permission.
 *
 *      THIS SOFTWARE IS PROVIDED BY XILINX, INC. "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL XILINX, INC. BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 *      PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE
 */
#include "xclbin_file.hpp"
#include "vitis/ai/target_factory.hpp"
#include <fstream>
#include <glog/logging.h>
#include <vector>
#ifdef ENABLE_XRT
#  include <xclbin.h>
#endif

namespace vaip_core {
#ifdef ENABLE_XRT
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
    if (hdr.m_sectionKind == kind) {
      return hdr;
    }
  }
  std::string error =
      std::string("section not found for kind ") + std::to_string(kind);
  throw std::runtime_error(error);
  return {};
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
#else
std::optional<uint64_t>
get_xclbin_fingerprint(const std::filesystem::path& filename) {
  LOG(ERROR) << "Package built without XRT";
  return std::nullopt;
}
#endif
} // namespace vaip_core
