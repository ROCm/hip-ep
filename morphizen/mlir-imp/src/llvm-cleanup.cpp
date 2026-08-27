/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "morphizen-utils/cleanup.hpp"

#include "llvm/Support/ManagedStatic.h"

namespace {

static const bool __register_llvm_shutdown = []() {
  ::morphizen::add_cleanup_function("llvm shutdown", &llvm::llvm_shutdown);
  return true;
}();

} // namespace
