/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "version_info.hpp"
#include "vaip/vaip_ort_api.h"
#include <glog/logging.h>
#include <sstream>

#ifndef PROJECT_GIT_COMMIT_ID
#  define PROJECT_GIT_COMMIT_ID "N/A"
#endif
namespace vaip_core {
const std::string get_lib_name() {
  const auto ret = std::string{"morphizen"} + "." +
                   std::to_string(get_vaip_version_major()) + "." +
                   std::to_string(get_vaip_version_minor()) + "." +
                   std::to_string(get_vaip_version_patch());
  return ret;
}

const std::string get_lib_id() {
  const auto ret = std::string{PROJECT_GIT_COMMIT_ID};
  return ret;
}

unsigned int get_vaip_version_major() {
#ifdef VAIP_ORT_API_MAJOR
  return VAIP_ORT_API_MAJOR;
#else
  return 1;
#endif
}

unsigned int get_vaip_version_minor() {
#ifdef VAIP_ORT_API_MINOR
  return VAIP_ORT_API_MINOR;
#else
  return 0;
#endif
}

unsigned int get_vaip_version_patch() {
#ifdef VAIP_ORT_API_PATCH
  return VAIP_ORT_API_PATCH;
#else
  return 0;
#endif
}
extern "C" uint32_t vaip_get_version() {
  return (get_vaip_version_major() << 24) | (get_vaip_version_minor() << 16) |
         (get_vaip_version_patch() << 8);
}
} // namespace vaip_core
extern "C" const char* morphizen_get_build_info() {
  static char ret[2048] = {'\0'};
  if (ret[0] == '\0') {
    std::ostringstream str;
    using version_vec_tuple =
        std::vector<std::tuple<std::string, std::string, std::string>>;
    str << "\t"
        << "VAIP_ORT_API: " << vaip_core::get_vaip_version_major() << "."
        << vaip_core::get_vaip_version_minor() << "."
        << vaip_core::get_vaip_version_patch() << "\n";
    str << "\tBUILD: " << PROJECT_GIT_COMMIT_ID << "\n";
    for (auto& info : version_vec_tuple{
#include "vaip_version_info.hpp.inc"
         }) {
      str << "\t" << std::get<0>(info) << ";" << std::get<1>(info) << ";"
          << std::get<2>(info) << "\n";
    }
    auto c = str.str();
    CHECK_LE(c.size(), sizeof(ret)) << " buffer overflow";
    std::memcpy(ret, c.data(), c.size());
  }
  return &ret[0];
}
// Local Variables:
// mode:c++
// coding: utf-8-unix
// End:
