/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#define _CRT_SECURE_NO_WARNINGS

#include "morphizen-foundation/env_config.hpp"
#include <cstdlib>
#include <stdio.h>

// Custom getenv implementation
extern "C" const char* vitis_ai_getenv_s(const char* name) {
#if _WIN32
  size_t len = 0;
  char* ret = NULL;
  errno_t err = _dupenv_s(&ret, &len, name);

  if (err != 0) {
    fprintf(stderr, "cannot read env %s", name);
    abort();
  }
  return ret;
#else
  return getenv(name);
#endif
}

namespace morphizen::foundation {

std::string get_env_string(const char* name, const std::string& default_value) {
  std::string ret;

  // Use custom vitis_ai_getenv_s if available, otherwise fallback to standard
  // getenv
  auto p = vitis_ai_getenv_s(name);
  if (p == nullptr) {
    ret = default_value;
  } else {
    ret = p;
#if _WIN32
    free((void*)p); // Free memory on Windows as per original implementation
#endif
  }
  return ret;
}

// Alternative function name for compatibility
std::string my_getenv_s(const char* name, const std::string& default_value) {
  return get_env_string(name, default_value);
}

} // namespace morphizen::foundation
