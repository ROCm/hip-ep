/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

/**
 * DLL entry point for morphizen-mlir-compiler.dll
 *
 * This file provides the minimal DLL initialization code required by Windows.
 * The actual exports are defined in morphizen-mlir-compiler.def and implemented
 * in the MorphizenCInterface library.
 */

#ifdef _WIN32
#  include <windows.h>

BOOL APIENTRY DllMain(HMODULE hModule, DWORD ul_reason_for_call,
                      LPVOID lpReserved) {
  switch (ul_reason_for_call) {
  case DLL_PROCESS_ATTACH:
  case DLL_THREAD_ATTACH:
  case DLL_THREAD_DETACH:
  case DLL_PROCESS_DETACH:
    break;
  }
  return TRUE;
}
#endif
