/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "morphizen/util.hpp"
#include <dlfcn.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace morphizen {
MORPHIZEN_DLL_SPEC std::filesystem::path get_morphizen_path() {
  Dl_info info;
  if (dladdr(reinterpret_cast<const void*>(&get_morphizen_path), &info)) {
    return std::filesystem::path(info.dli_fname);
  }
  return {};
}
unsigned int get_tid() {
  return static_cast<unsigned int>(syscall(SYS_gettid));
}

unsigned int get_pid() {
  return static_cast<unsigned int>(syscall(SYS_getpid));
}

} // namespace morphizen
