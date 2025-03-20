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

#include "./cache_dir.hpp"

#include "morphizen/env_config.hpp"
#include <filesystem>
#include <glog/logging.h>
DEF_ENV_PARAM_2(USERNAME, "", std::string)
DEF_ENV_PARAM_2(USER, "", std::string)
DEF_ENV_PARAM_2(XLNX_CACHE_DIR, "", std::string)

namespace fs = std::filesystem;
namespace vaip_core {
static std::string get_user_name() {
  auto ret = std::string();
  if (!ENV_PARAM(USERNAME).empty()) {
    ret = ENV_PARAM(USERNAME);
  } else if (!ENV_PARAM(USER).empty()) {
    ret = ENV_PARAM(USER);
  }
  return ret;
}

static fs::path default_cache_directory() {
  auto tmp_dir =
#ifdef _WIN32
      fs::path("C:\\temp");
#else
      fs::path("/tmp");
#endif
  return tmp_dir / get_user_name() / "vaip" / ".cache";
}

bool file_exists(const fs::path& filename) { return fs::exists(filename); }

fs::path get_cache_file_name(const PassContext& context,
                             const std::string& filename) {
  auto cache_dir = context.get_log_dir();
  return cache_dir / filename;
}

void update_cache_dir(PassContextImp& context) {
  auto cache_dir = fs::path(ENV_PARAM(XLNX_CACHE_DIR));
  // use json config first.
  auto config_cache_dir = context.context_proto.config().cache_dir();
  if (!config_cache_dir.empty()) {
    cache_dir = fs::u8path(config_cache_dir);
  }
  if (ENV_PARAM(XLNX_CACHE_DIR).empty() && config_cache_dir.empty()) {
    cache_dir = default_cache_directory();
  }

  context.log_dir =
      cache_dir / fs::u8path(context.context_proto.config().cache_key());
  *context.context_proto.mutable_config()->mutable_cache_dir() =
      cache_dir.u8string();
  if (context.cache_in_mem()) {
    LOG(WARNING) << "skip update cache dir: in-mem mode";
    return;
  }
  if (!fs::exists(context.log_dir) &&
      !fs::create_directories(context.log_dir)) {
    LOG(WARNING) << "cannot create cache directory: dir=" << context.log_dir;
  }
}

} // namespace vaip_core
