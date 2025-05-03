/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#pragma once
#include "./_sanity_check.hpp"
namespace xir {
struct bfloat16_t;
}
#include <cstdlib>
#include <stdint.h>
#include <vaip/export.h>
#include <vector>
namespace vaip_core {
VAIP_DLL_SPEC
void transpose_f(const float* src, float* dst,
                 const std::vector<int64_t>& shape,
                 const std::vector<int64_t>& perm);
VAIP_DLL_SPEC
void transpose_i8(const int8_t* src, int8_t* dst,
                  const std::vector<int64_t>& shape,
                  const std::vector<int64_t>& perm);

VAIP_DLL_SPEC
void transpose_ui8(const uint8_t* src, uint8_t* dst,
                   const std::vector<int64_t>& shape,
                   const std::vector<int64_t>& perm);
VAIP_DLL_SPEC
void transpose_i16(const int16_t* src, int16_t* dst,
                   const std::vector<int64_t>& shape,
                   const std::vector<int64_t>& perm);
VAIP_DLL_SPEC
void transpose_u16(const uint16_t* src, uint16_t* dst,
                   const std::vector<int64_t>& shape,
                   const std::vector<int64_t>& perm);
VAIP_DLL_SPEC
void transpose_bf16(const xir::bfloat16_t* src, xir::bfloat16_t* dst,
                    const std::vector<int64_t>& shape,
                    const std::vector<int64_t>& perm);
} // namespace vaip_core
