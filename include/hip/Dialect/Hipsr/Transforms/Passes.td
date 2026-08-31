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

def AddContextArgPass : Pass<"hipsr-add-context-arg", "mlir::ModuleOp"> {
  let summary = "Insert !hipsr.context as arg 0 into every func.func";
  let description = [{
    Inserts a !hipsr.context argument at position 0 into every func.func in
    the module. The OnnxToHipsr conversion patterns assume that the context
    value is always available as the first function argument.

    Skips functions that already have a !hipsr.context as argument 0.
  }];
  let dependentDialects = [
    "::mlir::hipsr::HipsrDialect",
    "::mlir::func::FuncDialect"
  ];
}

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
  let summary = "Split a function into pool domains based on its inputs";
  let description = [{
    Splits the top-level ops in a single-block function into pool domains. Each
    op goes into the highest domain used by its inputs. A barrier
    `hipsr.placeholder` with an input from another op starts the next domain.
    Ops whose inputs are only block arguments stay in domain zero.

    The pass chooses all domains before changing the IR, then builds
    `hipsr.pool_domain` ops in order. Each domain has `IsolatedFromAbove`.
    Values from outside the domain become operands and block arguments.
    Results used outside the domain become its results, including placeholder
    results that keep the shape graph connected across domains.
    Function argument zero must be `!hipsr.context`; it stays operand and block
    argument zero in every domain.

    A placeholder stays in the same domain as the DPS op that uses it as an
    init. Its result and the matching DPS data result are returned separately
    when both are used by a later domain.

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
      %0:2 = hipsr.pool_domain(
          %ctx, %input : !hipsr.context, tensor<4xf32>) {
      ^bb0(%domain_ctx: !hipsr.context, %domain_input: tensor<4xf32>):
        %init0 = hipsr.placeholder(%domain_ctx)
            ins(%domain_input : tensor<4xf32>)
            {placeholder_type = #hipsr.placeholder_type<normal>}
            : tensor<4xf16>
        %data0 = hipsr.cast(%domain_ctx)
            ins(%domain_input : tensor<4xf32>)
            outs(%init0 : tensor<4xf16>) : tensor<4xf16>
        hipsr.pool_domain_yield %init0, %data0
            : tensor<4xf16>, tensor<4xf16>
      } -> tensor<4xf16>, tensor<4xf16> {domain_id = 0 : i64}
      %1 = hipsr.pool_domain(%ctx, %0#0, %0#1
          : !hipsr.context, tensor<4xf16>, tensor<4xf16>) {
      ^bb0(%domain_ctx: !hipsr.context,
           %domain_shape_input: tensor<4xf16>,
           %domain_data_input: tensor<4xf16>):
        %init1 = hipsr.placeholder(%domain_ctx)
            ins(%domain_shape_input : tensor<4xf16>)
            {placeholder_type = #hipsr.placeholder_type<barrier>}
            : tensor<4xf32>
        %data1 = hipsr.cast(%domain_ctx)
            ins(%domain_data_input : tensor<4xf16>)
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
           "Debug-only: report the chosen domains before building them. "
           "Does not change the generated IR.">
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
       one extent tensor per result and holding the body moved out of the
       placeholder's `shape_region`;
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

def HipsrUseOutputAllocatorPass
    : Pass<"hipsr-use-output-allocator", "mlir::func::FuncOp"> {
  let summary =
      "Rewrite returned memref.alloc outputs to hipsr.alloc_output";
  let description = [{
    Replaces each graph-output `memref.alloc` with `hipsr.alloc_output`, reusing
    the alloc's dynamic-size operands and setting `out_idx` to its position in
    `func.return`. Output buffers then come from the EP output allocator and are
    owned by the EP/runtime instead of being deallocated or pooled.

    An allocation counts as a graph output when `func.return` returns it either
    directly or through view-like aliases. The view operations are left in
    place.

    Only public graph-entry functions whose argument 0 is `!hipsr.context` are
    rewritten. Private helper functions and functions without a HIPSR context
    are skipped. The function signature and `func.return` are left unchanged.
  }];
  let dependentDialects = [
    "::mlir::hipsr::HipsrDialect",
    "::mlir::memref::MemRefDialect"
  ];
}
#endif // HIPSR_TRANSFORMS_PASSES
