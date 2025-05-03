/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#define WIN32_LEAN_AND_MEAN

#include <windows.h>
//
#include "glog/logging.h"
#include "morphizen/util.hpp"

namespace vaip_core {
VAIP_DLL_SPEC std::filesystem::path get_vaip_path() {
  wchar_t path[MAX_PATH];
  HMODULE hm = NULL;

  if (GetModuleHandleEx(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                            GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                        (LPCSTR)&get_vaip_path, &hm) == 0) {
    int ret = GetLastError();
    LOG(ERROR) << "GetModuleHandle failed, error = " << ret;
    return {};
  }
  if (GetModuleFileNameW(hm, path, sizeof(path) / sizeof(wchar_t)) == 0) {
    int ret = GetLastError();
    LOG(ERROR) << "GetModuleFileName failed, error = " << ret;
    return {};
  }
  return std::filesystem::path(path);
}

unsigned int get_tid() {
  return static_cast<unsigned int>(GetCurrentThreadId());
}

unsigned int get_pid() {
  return static_cast<unsigned int>(GetCurrentProcessId());
}
} // namespace vaip_core
