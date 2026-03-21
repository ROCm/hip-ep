/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef HIP_TOOLS_COMMON_DLLLOADER_H
#define HIP_TOOLS_COMMON_DLLLOADER_H

#include "llvm/Support/DynamicLibrary.h"
#include <iostream>
#include <string>

/// Cross-platform DLL loader using LLVM's DynamicLibrary.
/// Shared by hip-test-dll and hip-inspect-dll.
class DllLoader {
public:
  explicit DllLoader(const std::string &path) {
    std::string errMsg;
    if (llvm::sys::DynamicLibrary::LoadLibraryPermanently(path.c_str(),
                                                          &errMsg)) {
      std::cerr << "Failed to load DLL: " << path << " - " << errMsg << "\n";
    } else {
      lib_ =
          llvm::sys::DynamicLibrary::getPermanentLibrary(path.c_str(), &errMsg);
      valid_ = lib_.isValid();
    }
  }

  void *getSymbol(const char *name) {
    if (!valid_)
      return nullptr;
    return lib_.getAddressOfSymbol(name);
  }

  bool isValid() const { return valid_; }

private:
  llvm::sys::DynamicLibrary lib_;
  bool valid_ = false;
};

#endif // HIP_TOOLS_COMMON_DLLLOADER_H
