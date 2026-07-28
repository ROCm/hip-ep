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

def PopulateShapeRegionPass
    : Pass<"hipsr-populate-shape-region", "mlir::func::FuncOp"> {
  let summary = "Fill empty hipsr shape regions via ShapeRegionInterface";
  let description = [{
    Walks every op implementing `ShapeRegionInterface` and, for each one whose
    shape region is still empty (zero blocks, i.e. unpopulated), calls
    `populateShapeRegion()` to emit its output-shape computation. Ops whose
    region is already populated are left untouched, so the pass is idempotent.

    This is the single place that turns the "unpopulated" ops produced by the
    conversion stage (e.g. `convert-onnx-to-hipsr`, which leaves shape regions
    empty) into fully-shaped ops. It is also the driver that exercises each
    op's `populateShapeRegion()` end-to-end in LIT tests, so the shape
    computation is verified against generated (not hand-written) IR.
  }];
  // Union of the dialects that ops' populateShapeRegion() emit (can't be
  // auto-derived: getDependentDialects runs before any IR exists). Extend it
  // when a new op's populateShapeRegion() uses a new dialect; a missing entry
  // asserts at op creation, caught by that op's populate LIT test.
  let dependentDialects = [
    "::mlir::hipsr::HipsrDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::shape::ShapeDialect",
    "::mlir::tensor::TensorDialect"
  ];
}

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

#endif // HIPSR_TRANSFORMS_PASSES
