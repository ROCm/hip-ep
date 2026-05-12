/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

/**
 * DLL entry point for hip-compiler.dll
 *
 * This file provides the minimal DLL initialization code required by Windows.
 * The actual exports are defined in hip-compiler.def and implemented
 * in the MorphizenCInterface library.
 */

#ifdef _WIN32
#include <windows.h>

#include "CrashHandler.h"

BOOL APIENTRY DllMain(HMODULE hModule, DWORD ul_reason_for_call,
                      LPVOID lpReserved) {
  switch (ul_reason_for_call) {
  case DLL_PROCESS_ATTACH:
    hip::install_crash_handlers("hip-compiler");
    break;
  case DLL_THREAD_ATTACH:
  case DLL_THREAD_DETACH:
  case DLL_PROCESS_DETACH:
    break;
  }
  return TRUE;
}
#endif
