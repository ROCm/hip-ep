/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "CrashHandler.h"

#include <cpptrace/cpptrace.hpp>
#include <csignal>
#include <iostream>
#include <mutex>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

namespace hip {

static const char *g_component_name = "hip";

void install_crash_handlers(const char *component_name) {
  static std::once_flag flag;
  std::call_once(flag, [component_name] {
    g_component_name = component_name;

    cpptrace::register_terminate_handler();

    std::signal(SIGABRT, [](int) {
      std::cerr << "\n=== " << g_component_name << ": abort() ===" << std::endl;
      cpptrace::generate_trace().print(std::cerr);
    });

#ifdef _WIN32
    AddVectoredExceptionHandler(TRUE, [](EXCEPTION_POINTERS *info) -> LONG {
      static thread_local bool in_handler = false;
      const auto code = info->ExceptionRecord->ExceptionCode;
      if (!in_handler && (code & 0xF0000000) == 0xC0000000 &&
          code != 0xC0000008 && code != 0xC00000FE) {
        in_handler = true;
        std::cerr << "\n=== " << g_component_name << ": exception 0x"
                  << std::hex << code << " at "
                  << info->ExceptionRecord->ExceptionAddress << std::dec
                  << " ===" << std::endl;
        cpptrace::generate_raw_trace().resolve().print(std::cerr);
        in_handler = false;
      }
      return EXCEPTION_CONTINUE_SEARCH;
    });
#endif
  });
}

} // namespace hip
