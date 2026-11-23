/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- WinogradKernel.h - Build hip.winograd_conv kernels -------*- C++ -*-===//
//
// Each hip.winograd_conv gets a Winograd F(2x2, 3x3) kernel built for its shape
// and epilogue from the vendored MIGraphX kernel headers
// (third_party/migraphx), compiled with hiprtc while the model compiles. The
// kernel reads every extent as a compile-time constant, so one build serves
// one shape; identical (shape, epilogue) pairs share one build.
//
//===----------------------------------------------------------------------===//

#ifndef HIP_COMPILER_WINOGRADKERNEL_H
#define HIP_COMPILER_WINOGRADKERNEL_H

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Support/LogicalResult.h>

#include <llvm/Support/raw_ostream.h>

#include <cstdint>
#include <string>

namespace mlir::hip {

/// Whether the Winograd kernel has a code path for `arch` (gfx1151 and the
/// gfx12 family).
bool winogradKernelSupportsArch(llvm::StringRef arch);

/// Tuning parameters of the MIGraphX kernel template
/// (winograd_conv_f23_wmma<NW, CB, KW, SK, FE, FT, ...>).
struct WinogradConfig {
  int64_t waves = 4;          // NW: waves per workgroup
  int64_t channelBlock = 16;  // CB: input channels per WMMA K step
  int64_t outBlocks = 1;      // KW: 16-wide output-channel blocks per group
  int64_t channelSplit = 1;   // SK: waves that split the channel sum
  bool foldAtEnd = true;      // FE: output transform once, after the c loop
  bool fullTransform = false; // FT: read the raw filter g, not T = G g
};

/// Output channels per 256-thread block of the generated `<name>_fixup`
/// kernel. wrap_winograd_conv (lib/Runtime/real/winograd_conv.cpp) launches
/// ceil(K / this) blocks and must use the same value.
inline constexpr int64_t kFixupChannels = 16;

/// The configuration used for an [N, C, H, W] -> [N, K, H, W] conv.
WinogradConfig chooseWinogradConfig(int64_t n, int64_t c, int64_t h, int64_t w,
                                    int64_t k);

/// The kernel source for one shape, configuration and epilogue, plus its
/// launch geometry. `activation` is "none", "relu" or "leaky_relu".
struct WinogradKernelSource {
  std::string source;
  std::string name;
  int64_t gridSize = 0;
  int64_t blockSize = 0;
};
WinogradKernelSource
buildWinogradKernelSource(int64_t n, int64_t c, int64_t h, int64_t w, int64_t k,
                          const WinogradConfig &config, bool hasBias,
                          llvm::StringRef activation, double alpha);

/// Build a kernel for every hip.winograd_conv in `module` and store its code
/// object, symbol, launch geometry and filter store kind on the op. Fails if
/// hiprtc is unavailable or a build fails.
LogicalResult compileWinogradKernels(ModuleOp module, llvm::StringRef arch,
                                     llvm::raw_ostream *log = nullptr);

} // namespace mlir::hip

#endif // HIP_COMPILER_WINOGRADKERNEL_H
