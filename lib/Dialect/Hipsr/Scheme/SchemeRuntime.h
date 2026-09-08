/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef LIB_DIALECT_HIPSR_SCHEME_SCHEMERUNTIME_H
#define LIB_DIALECT_HIPSR_SCHEME_SCHEMERUNTIME_H

#include <string>
#include <vector>

namespace mlir {
namespace hipsr {

using SchemeValue = void*;

bool initializeSchemeRuntime();

std::string callSchemeFunction(const char* functionName,
                                const std::vector<SchemeValue>& args);

SchemeValue makeSchemeString(const char* str);
SchemeValue makeSchemeInteger(long value);

} // namespace hipsr
} // namespace mlir

#endif
