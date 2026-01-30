/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "./cache_dir.hpp"

#include "morphizen/env_config.hpp"
#include <filesystem>
#include <glog/logging.h>
DEF_ENV_PARAM_2(USERNAME, "", std::string)
DEF_ENV_PARAM_2(USER, "", std::string)
DEF_ENV_PARAM_2(XLNX_CACHE_DIR, "", std::string)

namespace fs = std::filesystem;
namespace morphizen {
static std::string get_user_name() {
  auto ret = std::string();
  if (!ENV_PARAM(USERNAME).empty()) {
    ret = ENV_PARAM(USERNAME);
  } else if (!ENV_PARAM(USER).empty()) {
    ret = ENV_PARAM(USER);
  }
  std::ostringstream str;
  for (auto x : ret) {
    if (std::isalnum(x)) {
      str << x;
    } else {
      str << "_" << std::hex << ((unsigned int)(x & 0xF))
          << (unsigned int)(x >> 4 & 0xF);
    }
  }
  return str.str();
}

static fs::path default_cache_directory() {
  auto tmp_dir =
#ifdef _WIN32
      fs::path("C:\\temp");
#else
      fs::path("/tmp");
#endif
  return tmp_dir / get_user_name() / "morphizen" / ".cache";
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

  context.pass_context_log_dir_ =
      cache_dir / fs::u8path(context.context_proto.config().cache_key());
  *context.context_proto.mutable_config()->mutable_cache_dir() =
      cache_dir.u8string();
  // Cache is always in memory, skip creating cache directory
}

} // namespace morphizen
