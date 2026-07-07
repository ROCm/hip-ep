/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "morphizen/onnxruntime_morphizen_ep.hpp"
#if MORPHIZEN_HAS_PATTERN_MATCHING
#include "morphizen/pattern.hpp"
#endif
#include "morphizen/morphizen.hpp"
#if _WIN32
#ifdef _DEBUG
#include <crtdbg.h>
#endif
#endif

typedef void *voidp;
static struct {
  const char *name;
  void *symbol;
} table[] = {{"deinitialize_onnxruntime_morphizen_ep",
              (void *)deinitialize_onnxruntime_morphizen_ep},
             {"morphizen_get_version", (void *)morphizen_get_version}
#if MORPHIZEN_HAS_PATTERN_MATCHING
             ,
             {"morphizen::Pattern::enable_trace",
              (void *)morphizen::Pattern::enable_trace}
#endif
};

static void *lookup_symbol(const char *name) {
  for (size_t i = 0; i < sizeof(table) / sizeof(table[0]); ++i) {
    if (strcmp(name, table[i].name) == 0) {
      return table[i].symbol;
    }
  }
  return nullptr;
}
static void *disable_crt_diag() {
#if _WIN32
#ifdef _DEBUG
  // Disable assertion dialog in CI
  _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE);
  _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
  _CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_FILE);
  _CrtSetReportFile(_CRT_ERROR, _CRTDBG_FILE_STDERR);
  _CrtSetReportMode(_CRT_WARN, _CRTDBG_MODE_FILE);
  _CrtSetReportFile(_CRT_WARN, _CRTDBG_FILE_STDERR);
#endif
#endif
  return nullptr;
}

extern "C" MORPHIZEN_DLL_SPEC void *morphizen_main(int argc, char *argv[]) {
  if (argc == 0) {
    return nullptr;
  }
  auto cmd = std::string(argv[0]);
  if (cmd == "help") {
    std::cout << "Available commands:" << std::endl;
    for (size_t i = 0; i < sizeof(table) / sizeof(table[0]); ++i) {
      std::cout << table[i].name << std::endl;
    }
    return nullptr;
  } else if (cmd == "disable_crt_diag") {
    return disable_crt_diag();
  } else if (cmd == "get_global_morphizen_ort_api") {
    return (void *)morphizen::api();
  } else if (cmd == "symbol") {
    return lookup_symbol(argv[1]);
  }
  return nullptr;
}
