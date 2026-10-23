/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef HIP_COMPILER_MATMULNBITSROWPADS_H
#define HIP_COMPILER_MATMULNBITSROWPADS_H

#include <cstdint>
#include <string>
#include <vector>

namespace hip::compiler {

/// How hip-pad-matmul-nbits-weights lays out int4 weights on the running
/// device, from its autotune table in custom_kernels_<arch> (located like the
/// native link does: HIP_CUSTOM_KERNELS_DIR / HIP_CUSTOM_KERNELS_ARCH, else the
/// directory of this module and the build's default arch).
struct MatmulNBitsRowPads {
  /// Pad for an aliasing weight `entries` does not list
  /// (hip_matmul_nbits_u4_default_row_pad); 0 pads none.
  int64_t defaultPad = 0;
  /// `N:K:block_size:zero_points=pad_bytes` entries
  /// (hip_matmul_nbits_u4_row_pad_table).
  std::vector<std::string> entries;
};

/// Both empty when the library, either export or a table for the device is
/// unavailable, so a device that was not measured keeps the dense layout and
/// the runtime's per-call repack.
MatmulNBitsRowPads queryMatmulNBitsRowPads();

} // namespace hip::compiler

#endif // HIP_COMPILER_MATMULNBITSROWPADS_H
