/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
/* Empty Gemm LUT payload, for builds with no measured table (and for the sweep
 * driver, which must not get LUT hits for the shapes it is measuring).
 *
 * CMake links this when lut/<arch>.fb is absent (no measured table for that
 * arch); when the .fb exists CMake instead embeds its bytes directly via
 * file(READ ... HEX). Either way the "no table" path reports size 0, a miss,
 * and every shape falls through to the heuristic / runtime sweep exactly as
 * before the LUT was wired in.
 */
#include <cstddef>

extern "C" const unsigned char kGemmLutData[1] = {0};
extern "C" const size_t kGemmLutData_size = 0;
