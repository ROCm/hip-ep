/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef HIP_COMPILER_MATMULNBITSROWPADS_H
#define HIP_COMPILER_MATMULNBITSROWPADS_H

#include <string>
#include <vector>

namespace hip::compiler {

/// Per-weight row pads that the custom kernels library measured for the
/// running device, as `N:K:block_size:zero_points=pad_bytes` entries for
/// hip-pad-matmul-nbits-weights (`matmul-nbits-row-pads`). Read from
/// hip_matmul_nbits_u4_row_pad_table() in custom_kernels_<arch>, located like
/// the native link does (HIP_CUSTOM_KERNELS_DIR / HIP_CUSTOM_KERNELS_ARCH, else
/// the directory of this module and the build's default arch). Empty when the
/// library, the export or a table for the device is unavailable; the pass then
/// applies its default rule.
std::vector<std::string> queryMatmulNBitsRowPads();

} // namespace hip::compiler

#endif // HIP_COMPILER_MATMULNBITSROWPADS_H
