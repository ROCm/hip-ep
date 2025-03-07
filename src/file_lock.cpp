/*
 *     The Xilinx Vitis AI Vaip in this distribution are provided under the
 * following free and permissive binary-only license, but are not provided in
 * source code form.  While the following free and permissive license is similar
 * to the BSD open source license, it is NOT the BSD open source license nor
 * other OSI-approved open source license.
 *
 *      Copyright (C) 2022 Xilinx, Inc. All rights reserved.
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
#include "file_lock.hpp"
#include <exception>
#include <filesystem>
#include <fstream>
#include <glog/logging.h>
#include <iostream>
#include <vitis/ai/env_config.hpp>
DEF_ENV_PARAM(DEBUG_FILE_LOCK, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_FILE_LOCK) >= n)

namespace vaip_core {
#ifdef ENABLE_BOOST
WithFileLock::WithFileLock(std::filesystem::path filename) {
  try {
    MY_LOG(1) << "get boost file lock, filename : " << filename.u8string();
    if (!std::filesystem::exists(filename)) {
      MY_LOG(1) << "=== create lock file : " << filename.u8string();
      std::ofstream ofs(filename);
      ofs.close();
    }

    lock_ = boost::interprocess::file_lock(filename.c_str());
    lock_.lock();
    MY_LOG(1) << "get boost file lock success, filename : "
              << filename.u8string() << " lock : " << &lock_;
  } catch (const std::exception& e) {
    std::cerr << "exception occurs : " << e.what() << "\n";
  }
}
WithFileLock::~WithFileLock() {
  try {
    MY_LOG(1) << "unlock boost file lock ... " << &lock_;
    lock_.unlock();
  } catch (const std::exception& e) {
    std::cerr << "exception occurs : " << e.what() << "\n";
  }
}
#else
static std::mutex& get_mutex_lock() {
  MY_LOG(1) << "get std::mutex lock ... ";
  static std::mutex mutex;
  MY_LOG(1) << "get std::mutex lock success ... ";
  return mutex;
}
WithFileLock::WithFileLock(std::filesystem::path filename)
    : lock_(get_mutex_lock()) {}
WithFileLock::~WithFileLock() {
  MY_LOG(1) << "unlock std::mutex lock ... " << std::endl;
}
#endif
} // namespace vaip_core
