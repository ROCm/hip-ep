//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// TableGen definitions for hipsr dialect transform passes, mirroring the hip
// dialect's Dialect/Transforms/Passes.td. New hipsr transforms add their `def`
// here; the aggregate header Dialect/Hipsr/Transforms/Passes.h pulls the
// declarations and registration via GEN_PASS_DECL / GEN_PASS_REGISTRATION.
//

#ifndef HIPSR_TRANSFORMS_PASSES
#define HIPSR_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def HipsrExternalizeConstantsPass
    : Pass<"hipsr-externalize-constants", "mlir::ModuleOp"> {
  let summary = "Externalize hipsr.constant: stamp offset/size/index, write constants file";
  let description = [{
    Externalizes each `hipsr.constant`: stamps `offset`/`size`/`index`.
    With a FileSystem injected, also writes the constants file.
  }];
  let options = [
    Option<"constantsFile", "constants-file", "std::string",
           /*default=*/"\"constants.bin\"",
           "Constants-file name written through the injected FileSystem">
  ];
  let dependentDialects = [
    "::mlir::hipsr::HipsrDialect"
  ];
}

def HipsrPoolAllocPass : Pass<"hipsr-pool-alloc", "mlir::ModuleOp"> {
  let summary = "Pool intermediate buffers per pool_domain by lifetime";
  let description = [{
    Pools each `hipsr.pool_domain`'s `memref.alloc`s by lifetime:
    non-overlapping ones share one `hipsr.get_pool` buffer as `memref.view`s.
  }];
  let options = [
    Option<"emitPoolReport", "emit-pool-report", "bool", /*default=*/"false",
           "Debug-only: emit remarks describing the pooling decisions. "
           "No effect on generated code.">
  ];
  let dependentDialects = [
    "::mlir::hipsr::HipsrDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::arith::ArithDialect"
  ];
}

#endif // HIPSR_TRANSFORMS_PASSES
