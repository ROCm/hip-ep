//===- SchemeRuntime.cpp - Chez Scheme Runtime Initialization ---*- C++ -*-===//
//
// Initializes Chez Scheme with embedded boot file containing rime
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/raw_ostream.h"
#include <cstddef>

// Chez Scheme C API (minimal declarations)
extern "C" {
void Sscheme_init(void *(*)(size_t), void (*)(void*));
void Sregister_boot_file_bytes(const char*, const unsigned char*, size_t);
void Sbuild_heap(const char*, void (*)(const char*));
void* Scons(void*, void*);
void* Sstring_to_symbol(const char*);
void* Stop_level_value(void*);

// Embedded boot file
extern const unsigned char chez_boot_data[];
extern const size_t chez_boot_size;
}

namespace {
static bool scheme_initialized = false;
}

namespace mlir {
namespace hipsr {

/// Initialize Chez Scheme runtime with embedded boot file
bool initializeSchemeRuntime() {
  if (scheme_initialized)
    return true;

  llvm::errs() << "Initializing Chez Scheme with embedded boot (rime included)...\n";
  llvm::errs() << "Boot size: " << chez_boot_size << " bytes\n";

  // Initialize Chez with default allocators
  Sscheme_init(nullptr, nullptr);

  // Register embedded boot file
  Sregister_boot_file_bytes("hip-patterns.boot", chez_boot_data, chez_boot_size);

  // Build heap
  Sbuild_heap(nullptr, nullptr);

  llvm::errs() << "Chez Scheme initialized successfully. rime/loop is available.\n";

  scheme_initialized = true;
  return true;
}

/// Evaluate Scheme expression (simple wrapper)
void* evalSchemeString(const char* expr) {
  // TODO: Implement using Seval_string or similar
  return nullptr;
}

} // namespace hipsr
} // namespace mlir
