/*
 *     The Xilinx Vitis AI Vaip in this distribution are provided under the
 * following free and permissive binary-only license, but are not provided in
 * source code form.  While the following free and permissive license is similar
 * to the BSD open source license, it is NOT the BSD open source license nor
 * other OSI-approved open source license.
 *
 *      Copyright (C) 2022 Xilinx, Inc.
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

#include <fstream>
#include <glog/logging.h>
#include <iostream>
#include <unordered_map>
#include <vaip/mem_xclbin.hpp>
#include <zlib.h>
namespace vaip_core {

struct CompressionInfo {
  const uint8_t* data;
  size_t compressed_size;
  size_t origin_size;

  CompressionInfo(const uint8_t* d, size_t c, size_t o)
      : data(d), compressed_size(c), origin_size(o) {}
};
std::vector<char> uncompress(const uint8_t* byte, size_t compressed_size,
                             size_t origin_size) {
  std::vector<char> ret;
  ret.resize(static_cast<size_t>(origin_size));
  z_stream infstream;
  infstream.zalloc = Z_NULL;
  infstream.zfree = Z_NULL;
  infstream.opaque = Z_NULL;
  infstream.avail_in = static_cast<unsigned int>(compressed_size);
  infstream.next_in = reinterpret_cast<Bytef*>(
      const_cast<char*>(reinterpret_cast<const char*>(byte)));
  infstream.avail_out = static_cast<unsigned int>(origin_size);
  infstream.next_out = reinterpret_cast<Bytef*>(ret.data());
  inflateInit(&infstream);
  inflate(&infstream, Z_NO_FLUSH);
  inflateEnd(&infstream);
  return ret;
}
#include "mem_xclbin_file.hpp.inc"
std::vector<char> get_mem_xclbin(const std::string& filename) {
  auto iter = xclbin_map.find(filename);
  auto info = iter->second;
  return uncompress(info.data, info.compressed_size, info.origin_size);
}

bool has_mem_xclbin(const std::string& filename) {
  auto iter = xclbin_map.find(filename);
  return (iter != xclbin_map.end());
}

} // namespace vaip_core