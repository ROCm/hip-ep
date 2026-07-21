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
    "::mlir::shape::ShapeDialect"
  ];
}

def HipsrExternalizeConstantsPass
    : Pass<"hipsr-externalize-constants", "mlir::ModuleOp"> {
  let summary = "Write hipsr.constant data to a sidecar and mark offset/size";
  let description = [{
    Phase 2 of the hipsr constant subsystem. Runs in two stages:

    Phase 1 (always): walk every `hipsr.constant` in the module (across all
    functions) that opts in (`shouldExternalize()` is true) and is not already
    externalized; assign a 64-byte aligned cumulative offset and stamp the op
    with `offset`/`size`.

    Module-scoped by design: offsets accumulate into a single sidecar shared by
    the whole module, so the pass must see every constant at once. A per-function
    pass would restart offsets at 0 per function (overlapping placements) and
    write the shared file once per function (races / last-writer-wins).
    Append-only -- `value`/`source` are kept, so `getDataValues()` still works
    afterward. Builds one `hip::ConstantEntry` per constant (file_source keeps
    its path/offset with no byte read; inline/mem_source keep a byte view).

    Phase 2 (only when a FileSystem is injected on the HipsrDialect via
    `setFileSystem`): write the entries through
    `writeConstantsBinToFileSystem`. When no FileSystem is injected (e.g.
    hip-mlir-opt) the pass is a pure IR transform (no file written).

    Constants with `externalize = false` and ones already externalized are
    skipped. This pass emits no module attributes; runtime metadata generation
    is a separate follow-up.
  }];
  let options = [
    Option<"constantsFile", "constants-file", "std::string",
           /*default=*/"\"constants.bin\"",
           "Sidecar file name written through the injected FileSystem">
  ];
  let dependentDialects = [
    "::mlir::hipsr::HipsrDialect"
  ];
}

#endif // HIPSR_TRANSFORMS_PASSES
