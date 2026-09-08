/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef LIB_DIALECT_HIPSR_SCHEME_SCHEMERUNTIME_H
#define LIB_DIALECT_HIPSR_SCHEME_SCHEMERUNTIME_H

#include <string>
#include <vector>

extern "C" {
typedef void* ptr;
}

namespace mlir {
namespace hipsr {

bool initializeSchemeRuntime();

std::string callSchemeFunction(const char* functionName,
                                const std::vector<ptr>& args);

ptr makeSchemeString(const char* str);
ptr makeSchemeInteger(long value);

} // namespace hipsr
} // namespace mlir

#endif
