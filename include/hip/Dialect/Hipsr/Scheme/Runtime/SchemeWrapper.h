/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef LIB_DIALECT_HIPSR_SCHEME_SCHEME_WRAPPER_H
#define LIB_DIALECT_HIPSR_SCHEME_SCHEME_WRAPPER_H

// Wrapper for Chez Scheme's scheme.h
//
// Chez Scheme's generated scheme.h does NOT have include guards,
// which causes redefinition errors if included multiple times.
// This wrapper provides the protection.

extern "C" {
#include "boot/ta6le/scheme.h"
}

#endif // LIB_DIALECT_HIPSR_SCHEME_SCHEME_WRAPPER_H
