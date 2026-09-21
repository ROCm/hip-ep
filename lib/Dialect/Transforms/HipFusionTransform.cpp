/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- HipFusionTransform.cpp - HIP-to-HIP pattern rewriting -------------===//
//
// Applies the rewrite patterns in `fusion_pattern/` to the HIP dialect. Both
// sides of every pattern are `hip.*`, and the pass places no restriction on
// the arity of a rewrite: one op may become many, many may collapse into one,
// or a subgraph may be replaced by a different subgraph.
//
// How those patterns are written, built and ordered is `fusion_pattern/`'s
// concern; `fusion_pattern/hip_fusion_transform.hpp` is the driver this pass
// calls.
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/Passes.h"

// The pass lists the PDL dialects in `dependentDialects`, so the generated
// getDependentDialects() needs these declarations even in a build where PDLL
// pattern compilation is disabled.
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"

#include "fusion_pattern/hip_fusion_transform.hpp"

#include "llvm/Support/Debug.h"
#include "llvm/Support/MemoryBufferRef.h"

#include <cstddef>

#define DEBUG_TYPE "hip-fusion-transform"

// The compiled PDLL patterns, embedded as a byte array by CMake:
// mlir-pdll -> mlir-opt --strip-debuginfo --emit-bytecode -> xxd.py ->
// HipFusionTransformPdll_data.cpp. Nothing is loaded from disk.
extern "C" const unsigned char *hip_pdl_fusion_transform_data(void);
extern "C" size_t hip_pdl_fusion_transform_size(void);

namespace mlir {
namespace hip {

#define GEN_PASS_DEF_HIPFUSIONTRANSFORMPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

namespace {

// Buffer identifier for the embedded patterns. Parser diagnostics quote it, so
// it names the build artifact the bytes came from.
constexpr llvm::StringLiteral kPatternsName = "HipFusionTransformPatterns.pdl";

class HipFusionTransformPass
    : public impl::HipFusionTransformPassBase<HipFusionTransformPass> {
public:
  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Unlike convert-onnx-to-hip, whose patterns are the lowering itself,
    // every pattern here is an optimization: a build without mlir-pdll
    // degrades to leaving the matched IR in place, which still compiles and
    // runs. The native patterns are unaffected and run either way, so this is
    // a diagnostic rather than an early exit.
    LLVM_DEBUG({
      if (hip_pdl_fusion_transform_size() == 0)
        llvm::dbgs() << "no embedded PDLL patterns; only native rewrites run\n";
    });

    const llvm::MemoryBufferRef patterns(
        llvm::StringRef(
            reinterpret_cast<const char *>(hip_pdl_fusion_transform_data()),
            hip_pdl_fusion_transform_size()),
        kPatternsName);
    if (failed(::hip::fusion_transform::run(module, patterns)))
      return signalPassFailure();
  }
};

} // namespace

} // namespace hip
} // namespace mlir
