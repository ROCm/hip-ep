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