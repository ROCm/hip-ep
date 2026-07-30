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

def PartitionPoolDomainsPass
    : Pass<"hipsr-partition-pool-domains", "mlir::func::FuncOp"> {
  let summary = "Split a function into hipsr pool domains";
  let description = [{
    Groups as many ops as possible into each `hipsr.pool_domain` in a
    single-block function. Each domain is `IsolatedFromAbove`. An active
    `StartBarrierInterface` op starts a domain. An active
    `EndBarrierInterface` op stays in its domain; the first op that uses any
    result starts the next domain.

    See `hipsr.placeholder` for placeholder requirements.

    Values defined outside a domain become domain operands and block arguments.
    Values used outside the domain become domain results. The pass rejects
    multi-block functions and existing pool domains before changing the IR.

    `hipsr.example_end_barrier` below stands in for a future DPS end-barrier
    op.

    Before:
    ```mlir
    func.func @graph(%ctx: !hipsr.context, %input: tensor<?x4xf32>,
        %shape: tensor<2xi64>) -> tensor<?x4xf16> {
      %cast_init = hipsr.placeholder(
          %ctx, %input : !hipsr.context, tensor<?x4xf32>)
          {type = #hipsr.placeholder_type<normal>} : tensor<?x4xf16>
      %cast = hipsr.cast(%ctx) ins(%input : tensor<?x4xf32>)
          outs(%cast_init : tensor<?x4xf16>) : tensor<?x4xf16>
      %ready_init = hipsr.placeholder(
          %ctx, %cast : !hipsr.context, tensor<?x4xf16>)
          {type = #hipsr.placeholder_type<barrier>} : tensor<?x4xf16>
      %ready = hipsr.example_end_barrier(%ctx)
          ins(%cast : tensor<?x4xf16>)
          outs(%ready_init : tensor<?x4xf16>) : tensor<?x4xf16>
      %add_init = hipsr.placeholder(
          %ctx, %ready, %ready
          : !hipsr.context, tensor<?x4xf16>, tensor<?x4xf16>)
          {type = #hipsr.placeholder_type<normal>} : tensor<?x4xf16>
      %sum = hipsr.add(%ctx)
          ins(%ready, %ready : tensor<?x4xf16>, tensor<?x4xf16>)
          outs(%add_init : tensor<?x4xf16>) : tensor<?x4xf16>
      %expand_init = hipsr.placeholder(
          %ctx, %sum, %shape
          : !hipsr.context, tensor<?x4xf16>, tensor<2xi64>)
          {type = #hipsr.placeholder_type<barrier>} : tensor<?x4xf16>
      %expanded = hipsr.expand(%ctx)
          ins(%sum, %shape : tensor<?x4xf16>, tensor<2xi64>)
          outs(%expand_init : tensor<?x4xf16>) : tensor<?x4xf16>
      %result_init = hipsr.placeholder(
          %ctx, %expanded, %expanded
          : !hipsr.context, tensor<?x4xf16>, tensor<?x4xf16>)
          {type = #hipsr.placeholder_type<normal>} : tensor<?x4xf16>
      %result = hipsr.add(%ctx)
          ins(%expanded, %expanded : tensor<?x4xf16>, tensor<?x4xf16>)
          outs(%result_init : tensor<?x4xf16>) : tensor<?x4xf16>
      return %result : tensor<?x4xf16>
    }
    ```

    After:
    ```mlir
    func.func @graph(%ctx: !hipsr.context, %input: tensor<?x4xf32>,
        %shape: tensor<2xi64>) -> tensor<?x4xf16> {
      %0 = hipsr.pool_domain(%ctx, %input
          : !hipsr.context, tensor<?x4xf32>) {
      ^bb0(%domain_ctx: !hipsr.context,
           %domain_input: tensor<?x4xf32>):
        %cast_init = hipsr.placeholder(
            %domain_ctx, %domain_input : !hipsr.context, tensor<?x4xf32>)
            {type = #hipsr.placeholder_type<normal>} : tensor<?x4xf16>
        %cast = hipsr.cast(%domain_ctx)
            ins(%domain_input : tensor<?x4xf32>)
            outs(%cast_init : tensor<?x4xf16>) : tensor<?x4xf16>
        %ready_init = hipsr.placeholder(
            %domain_ctx, %cast : !hipsr.context, tensor<?x4xf16>)
            {type = #hipsr.placeholder_type<barrier>}
            : tensor<?x4xf16>
        %ready = hipsr.example_end_barrier(%domain_ctx)
            ins(%cast : tensor<?x4xf16>)
            outs(%ready_init : tensor<?x4xf16>) : tensor<?x4xf16>
        hipsr.pool_domain_yield %ready : tensor<?x4xf16>
      } -> tensor<?x4xf16>
      %1 = hipsr.pool_domain(%ctx, %0
          : !hipsr.context, tensor<?x4xf16>) {
      ^bb0(%domain_ctx: !hipsr.context,
           %domain_input: tensor<?x4xf16>):
        %add_init = hipsr.placeholder(
            %domain_ctx, %domain_input, %domain_input
            : !hipsr.context, tensor<?x4xf16>, tensor<?x4xf16>)
            {type = #hipsr.placeholder_type<normal>} : tensor<?x4xf16>
        %sum = hipsr.add(%domain_ctx)
            ins(%domain_input, %domain_input
                : tensor<?x4xf16>, tensor<?x4xf16>)
            outs(%add_init : tensor<?x4xf16>) : tensor<?x4xf16>
        hipsr.pool_domain_yield %sum : tensor<?x4xf16>
      } -> tensor<?x4xf16>
      %2 = hipsr.pool_domain(%ctx, %1, %shape
          : !hipsr.context, tensor<?x4xf16>, tensor<2xi64>) {
      ^bb0(%domain_ctx: !hipsr.context,
           %domain_input: tensor<?x4xf16>,
           %domain_shape: tensor<2xi64>):
        %expand_init = hipsr.placeholder(
            %domain_ctx, %domain_input, %domain_shape
            : !hipsr.context, tensor<?x4xf16>, tensor<2xi64>)
            {type = #hipsr.placeholder_type<barrier>}
            : tensor<?x4xf16>
        %expanded = hipsr.expand(%domain_ctx)
            ins(%domain_input, %domain_shape
                : tensor<?x4xf16>, tensor<2xi64>)
            outs(%expand_init : tensor<?x4xf16>) : tensor<?x4xf16>
        %result_init = hipsr.placeholder(
            %domain_ctx, %expanded, %expanded
            : !hipsr.context, tensor<?x4xf16>, tensor<?x4xf16>)
            {type = #hipsr.placeholder_type<normal>} : tensor<?x4xf16>
        %result = hipsr.add(%domain_ctx)
            ins(%expanded, %expanded : tensor<?x4xf16>, tensor<?x4xf16>)
            outs(%result_init : tensor<?x4xf16>) : tensor<?x4xf16>
        hipsr.pool_domain_yield %result : tensor<?x4xf16>
      } -> tensor<?x4xf16>
      return %2 : tensor<?x4xf16>
    }
    ```

    The end barrier stays in the first domain. Its first user starts the second
    domain. Expand is a start barrier, so it starts the third domain.
  }];
  let dependentDialects = [
    "::mlir::hipsr::HipsrDialect"
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
