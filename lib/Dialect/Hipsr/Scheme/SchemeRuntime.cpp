/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "llvm/Support/raw_ostream.h"
#include <cstddef>

extern "C" {
typedef void* ptr;
typedef long iptr;

void Sscheme_init(void *(*)(size_t), void (*)(void*));
void Sregister_boot_file_bytes(const char*, const unsigned char*, size_t);
const char* Skernel_version();

extern const unsigned char chez_boot_data[];
extern const size_t chez_boot_size;
}

namespace {
static bool scheme_initialized = false;
}

namespace mlir {
namespace hipsr {

bool initializeSchemeRuntime() {
  if (scheme_initialized)
    return true;

  llvm::errs() << "Initializing Chez Scheme runtime...\n";
  llvm::errs() << "Boot size: " << chez_boot_size << " bytes\n";
  llvm::errs() << "Chez Scheme version: " << Skernel_version() << "\n";

  Sscheme_init(nullptr, nullptr);
  Sregister_boot_file_bytes("hip-patterns.boot", chez_boot_data, chez_boot_size);

  llvm::errs() << "Chez Scheme C API successfully linked.\n";
  llvm::errs() << "Note: Full runtime initialization (Sbuild_heap) deferred to future work.\n";

  scheme_initialized = true;
  return true;
}

} // namespace hipsr
} // namespace mlir
