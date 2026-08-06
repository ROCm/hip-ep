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
  let summary = "Fill empty hipsr shape regions";
  let description = [{
    Populates each empty `hipsr.placeholder` shape region using the
    operation-specific recipe selected for its DPS consumer. The placeholder
    category determines the block argument layout.

    Already-populated placeholders are left unchanged. The pass fails if an
    empty placeholder has an unsupported consumer or an incompatible layout.
  }];
  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::hipsr::HipsrDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::shape::ShapeDialect",
    "::mlir::tensor::TensorDialect"
  ];
}

def PartitionPoolDomainsPass
    : Pass<"hipsr-partition-pool-domains", "mlir::func::FuncOp"> {
  let summary = "Partition a function into dependency-depth pool domains";
  let description = [{
    Assigns each top-level operation in a single-block function to the deepest
    domain used by its operands. A barrier `hipsr.placeholder` with an operation
    dependency advances to the next domain. Block-argument roots remain in
    domain zero.

    The pass computes the complete assignment before creating
    `hipsr.pool_domain` operations in domain order. Each domain is
    `IsolatedFromAbove`; captured values become operands and block arguments,
    and escaping non-placeholder values become results. The function must have
    a `!hipsr.context` as argument zero; it becomes operand and block argument
    zero in every domain.

    A placeholder and its tied DPS consumer stay in one domain. When a
    placeholder shape input crosses a boundary, the pass replaces that input
    with the tied DPS data result exported from the source domain. Placeholder
    results are not yielded across domain boundaries.

    Before:
    ```mlir
    func.func @graph(%ctx: !hipsr.context, %input: tensor<4xf32>)
        -> tensor<4xf32> {
      %init0 = hipsr.placeholder(%ctx)
          ins(%input : tensor<4xf32>)
          {placeholder_type = #hipsr.placeholder_type<normal>}
          : tensor<4xf16>
      %data0 = hipsr.cast(%ctx) ins(%input : tensor<4xf32>)
          outs(%init0 : tensor<4xf16>) : tensor<4xf16>
      %init1 = hipsr.placeholder(%ctx)
          ins(%init0 : tensor<4xf16>)
          {placeholder_type = #hipsr.placeholder_type<barrier>}
          : tensor<4xf32>
      %data1 = hipsr.cast(%ctx) ins(%data0 : tensor<4xf16>)
          outs(%init1 : tensor<4xf32>) : tensor<4xf32>
      return %data1 : tensor<4xf32>
    }
    ```

    After:
    ```mlir
    func.func @graph(%ctx: !hipsr.context, %input: tensor<4xf32>)
        -> tensor<4xf32> {
      %0 = hipsr.pool_domain(
          %ctx, %input : !hipsr.context, tensor<4xf32>) {
      ^bb0(%domain_ctx: !hipsr.context, %domain_input: tensor<4xf32>):
        %init0 = hipsr.placeholder(%domain_ctx)
            ins(%domain_input : tensor<4xf32>)
            {placeholder_type = #hipsr.placeholder_type<normal>}
            : tensor<4xf16>
        %data0 = hipsr.cast(%domain_ctx)
            ins(%domain_input : tensor<4xf32>)
            outs(%init0 : tensor<4xf16>) : tensor<4xf16>
        hipsr.pool_domain_yield %data0 : tensor<4xf16>
      } -> tensor<4xf16> {domain_id = 0 : i64}
      %1 = hipsr.pool_domain(%ctx, %0
          : !hipsr.context, tensor<4xf16>) {
      ^bb0(%domain_ctx: !hipsr.context, %domain_input: tensor<4xf16>):
        %init1 = hipsr.placeholder(%domain_ctx)
            ins(%domain_input : tensor<4xf16>)
            {placeholder_type = #hipsr.placeholder_type<barrier>}
            : tensor<4xf32>
        %data1 = hipsr.cast(%domain_ctx)
            ins(%domain_input : tensor<4xf16>)
            outs(%init1 : tensor<4xf32>) : tensor<4xf32>
        hipsr.pool_domain_yield %data1 : tensor<4xf32>
      } -> tensor<4xf32> {domain_id = 1 : i64}
      return %1 : tensor<4xf32>
    }
    ```
  }];
  let options = [
    Option<"emitAnalysisReport", "emit-analysis-report", "bool",
           /*default=*/"false",
           "Debug-only: report the read-only domain assignment before "
           "materialization. Does not change generated IR.">
  ];
  let dependentDialects = [
    "::mlir::hipsr::HipsrDialect"
  ];
}

def MaterializeInitTensorsPass
    : Pass<"hipsr-materialize-init-tensors", "mlir::ModuleOp"> {
  let summary = "Materialize hipsr.placeholder inits into tensor.empty";
  let description = [{
    Materializes every `hipsr.placeholder` init tensor inside each
    `hipsr.pool_domain`, leaving the domain body in three ordered groups:

    1. shape computation: one `scf.execute_region` per placeholder, yielding
       `!shape.shape` and holding the body moved out of the placeholder's
       `shape_region`;
    2. tensor allocation: `shape.get_extent` plus `tensor.empty` per
       placeholder result, taking rank and element type from that result type;
    3. data: the hipsr DPS ops, now initialized by the `tensor.empty` values.

    Every `scf.execute_region` precedes every `tensor.empty`, which precedes
    every data op, so `hipsr-pool-alloc` sees all buffer sizes at domain entry.

    Runs after `hipsr-partition-pool-domains` and requires shape regions
    already filled by `hipsr-populate-shape-region`. `hipsr.placeholder` has no
    bufferization interface, so this pass must run before bufferization.
  }];
  let dependentDialects = [
    "::mlir::hipsr::HipsrDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::scf::SCFDialect",
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
