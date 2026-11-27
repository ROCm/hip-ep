/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Ops - which operations the library compiles: those the compiler lowers, of
// dense types (but the packed weight a MUL_MAT multiplies by), that write no
// tensor outside the split but a cache's rows (SET_ROWS: the cache is a state
// of the split, updated in place).

#ifndef HIPEP_BACKEND_OPS_H
#define HIPEP_BACKEND_OPS_H

#include "hipep_backend.h"

namespace hipep_backend {

/// Whether the description's last tensor is an operation the library
/// compiles. HIPEP_BACKEND_OPS (ggml op names, comma separated) narrows the
/// set, for bisecting.
bool supports(const hipep_backend_split_desc &node);

} // namespace hipep_backend

#endif // HIPEP_BACKEND_OPS_H
