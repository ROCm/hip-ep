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

/// How hip-pad-matmul-nbits-weights lays out int4 weights for the compile
/// target (ROCK_ARCH when set, else the running device), from the target's
/// autotune table in a custom kernels library: custom_kernels_<arch> or
/// custom_kernels_<generic ISA> as the runtime loads it,
/// HIP_CUSTOM_KERNELS_ARCH first and the build's arch last, from
/// HIP_CUSTOM_KERNELS_DIR, else the directory of this module, else the build
/// tree.
struct MatmulNBitsRowPads {
  /// Pad for an aliasing weight `entries` does not list; 0 pads none.
  int64_t defaultPad = 0;
  /// The table's per-weight pads (hip_matmul_nbits_u4_row_pads) in the pass's
  /// `row-pads` format, `N:K:block_size:zero_points=pad_bytes`.
  std::vector<std::string> entries;
};

/// Both empty when no library with a table for the target is found, so a
/// target that was not measured keeps the dense layout and the runtime's
/// per-call repack. Answered once per (library, target) in a process.
MatmulNBitsRowPads queryMatmulNBitsRowPads();

} // namespace hip::compiler

#endif // HIP_COMPILER_MATMULNBITSROWPADS_H
