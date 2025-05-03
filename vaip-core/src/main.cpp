/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "morphizen/onnxruntime_vitisai_ep.hpp"
#include "morphizen/pattern.hpp"
#include "morphizen/vaip.hpp"
#if _WIN32
#  ifdef _DEBUG
#    include <crtdbg.h>
#  endif
#endif
extern "C" {
VAIP_DLL_SPEC
void initialize_onnxruntime_vitisai_ep(
    vaip_core::OrtApiForVaip* api, std::vector<OrtCustomOpDomain*>& ret_domain);
VAIP_DLL_SPEC
void deinitialize_onnxruntime_vitisai_ep();
}

typedef void* voidp;
static struct {
  const char* name;
  void* symbol;
} table[] = {{"deinitialize_onnxruntime_vitisai_ep",
              (void*)deinitialize_onnxruntime_vitisai_ep},
             {"vaip_core::Pattern::enable_trace",
              (void*)vaip_core::Pattern::enable_trace}};

static void* lookup_symbol(const char* name) {
  for (int i = 0; i < sizeof(table) / sizeof(table[0]); ++i) {
    if (strcmp(name, table[i].name) == 0) {
      return table[i].symbol;
    }
  }
  return nullptr;
}
static void* disable_crt_diag() {
#if _WIN32
#  ifdef _DEBUG
  // Disable assertion dialog in CI
  _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE);
  _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
  _CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_FILE);
  _CrtSetReportFile(_CRT_ERROR, _CRTDBG_FILE_STDERR);
  _CrtSetReportMode(_CRT_WARN, _CRTDBG_MODE_FILE);
  _CrtSetReportFile(_CRT_WARN, _CRTDBG_FILE_STDERR);
#  endif
#endif
  return nullptr;
}

extern "C" VAIP_DLL_SPEC void* morphizen_main(int argc, char* argv[]) {
  if (argc == 0) {
    return nullptr;
  }
  auto cmd = std::string(argv[0]);
  if (cmd == "help") {
    std::cout << "Available commands:" << std::endl;
    for (int i = 0; i < sizeof(table) / sizeof(table[0]); ++i) {
      std::cout << table[i].name << std::endl;
    }
    return nullptr;
  } else if (cmd == "disable_crt_diag") {
    return disable_crt_diag();
  } else if (cmd == "get_global_vaip_ort_api") {
    return (void*)vaip_core::api();
  } else if (cmd == "symbol") {
    return lookup_symbol(argv[1]);
  }
  return nullptr;
}
