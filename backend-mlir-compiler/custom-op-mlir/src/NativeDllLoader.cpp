/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "NativeDllLoader.h"

// CRITICAL: morphizen.hpp must be included before other morphizen headers
#include "morphizen/env_config.hpp"
#include "morphizen/morphizen.hpp"
#include <cstdlib>
#include <fstream>
#include <glog/logging.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

// Environment parameters (global scope, before namespace)
DEF_ENV_PARAM(MORPHIZEN_DEBUG_MLIR_BACKEND, "0")

#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_MLIR_BACKEND) >= n)

namespace mlir_compilation {
namespace customop {

DllHandle::DllHandle(void *handle, const DllFunctions &functions)
    : handle_(handle), functions_(functions) {}

std::optional<DllHandle>
DllHandle::load(const std::vector<uint8_t> &dll_bytes) {
  MY_LOG(1) << "Loading native DLL from memory...";

  // Strategy: Write to temp file, load, delete file
  // TODO: Use MemoryModule library for true in-memory loading

  char temp_path[L_tmpnam];
  if (!std::tmpnam(temp_path)) {
    LOG(WARNING) << "Failed to generate temporary DLL path";
    return std::nullopt;
  }

  std::string dll_path = std::string(temp_path) + ".dll";
  MY_LOG(2) << "Temporary DLL path: " << dll_path;

  // Write DLL to temp file
  {
    std::ofstream dll_out(dll_path, std::ios::binary);
    if (!dll_out) {
      LOG(WARNING) << "Failed to create temporary DLL file: " << dll_path;
      return std::nullopt;
    }
    dll_out.write(reinterpret_cast<const char *>(dll_bytes.data()),
                  dll_bytes.size());
    dll_out.close();
  }

  // Load DLL
  void *handle = nullptr;
  DllFunctions functions{nullptr, nullptr, nullptr};

#ifdef _WIN32
  handle = LoadLibraryA(dll_path.c_str());
  if (!handle) {
    DWORD error = GetLastError();
    LOG(WARNING) << "Failed to load DLL: " << dll_path
                 << " (error code: " << error << ")";
    std::remove(dll_path.c_str());
    return std::nullopt;
  }

  // Get function pointers
  functions.init = reinterpret_cast<init_fn>(
      GetProcAddress(static_cast<HMODULE>(handle), "inference_init"));
  functions.compute = reinterpret_cast<compute_fn>(
      GetProcAddress(static_cast<HMODULE>(handle), "inference_compute"));
  functions.cleanup = reinterpret_cast<cleanup_fn>(
      GetProcAddress(static_cast<HMODULE>(handle), "inference_cleanup"));

#else
  handle = dlopen(dll_path.c_str(), RTLD_LAZY);
  if (!handle) {
    LOG(WARNING) << "Failed to load shared library: " << dll_path << " ("
                 << dlerror() << ")";
    std::remove(dll_path.c_str());
    return std::nullopt;
  }

  // Get function pointers
  functions.init = reinterpret_cast<init_fn>(dlsym(handle, "inference_init"));
  functions.compute =
      reinterpret_cast<compute_fn>(dlsym(handle, "inference_compute"));
  functions.cleanup =
      reinterpret_cast<cleanup_fn>(dlsym(handle, "inference_cleanup"));
#endif

  // Delete temp file
  std::remove(dll_path.c_str());

  // Validate all function pointers
  if (!functions.init || !functions.compute || !functions.cleanup) {
    LOG(WARNING) << "Failed to load inference functions from DLL";
#ifdef _WIN32
    FreeLibrary(static_cast<HMODULE>(handle));
#else
    dlclose(handle);
#endif
    return std::nullopt;
  }

  MY_LOG(1) << "Native DLL loaded successfully";

  return DllHandle(handle, functions);
}

DllHandle::~DllHandle() {
  if (handle_) {
#ifdef _WIN32
    FreeLibrary(static_cast<HMODULE>(handle_));
#else
    dlclose(handle_);
#endif
    handle_ = nullptr;
  }
}

DllHandle::DllHandle(DllHandle &&other) noexcept
    : handle_(other.handle_), functions_(other.functions_) {
  other.handle_ = nullptr;
}

DllHandle &DllHandle::operator=(DllHandle &&other) noexcept {
  if (this != &other) {
    if (handle_) {
#ifdef _WIN32
      FreeLibrary(static_cast<HMODULE>(handle_));
#else
      dlclose(handle_);
#endif
    }
    handle_ = other.handle_;
    functions_ = other.functions_;
    other.handle_ = nullptr;
  }
  return *this;
}

const DllFunctions &DllHandle::functions() const { return functions_; }

} // namespace customop
} // namespace mlir_compilation
