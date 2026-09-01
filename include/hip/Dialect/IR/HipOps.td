//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

#ifndef HIP_OPS
#define HIP_OPS

include "hip/Dialect/IR/HipDialect.td"
include "hip/Dialect/IR/HipDpsOpInterface.td"
include "hip/Dialect/IR/HipOpStateInterface.td"
include "hip/Dialect/IR/HipTypes.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"

class Hip_Op<string mnemonic, list<Trait> traits = []> :
    Op<Hip_Dialect, mnemonic, traits>;

// Base class for all HIP destination-passing-style (DPS) compute ops.
//
// Provides:
//   - MemoryEffectsOpInterface
//   - DestinationStyleOpInterface (`getDpsInitsMutable` is op-defined)
//   - HipDpsOpInterface (carries the shared default `reifyResultShapes`)
//   - ReifyRankedShapedTypeOpInterface (decl; body auto-emitted below)
//   - Variadic<AnyRankedTensor>:$result_tensors (empty in memref mode)
//
// Shape behavior is selected by named family subclasses below rather than by
// boolean and string-body parameters on this structural root. Simple families
// emit fixed forwarding methods; semantic long-tail ops implement reification
// in `HipReifyResultShapesImpl.cpp`.
class Hip_DpsOp<string mnemonic, list<Trait> traits = []> :
    Hip_Op<mnemonic, !listconcat([
        DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
        DeclareOpInterfaceMethods<DestinationStyleOpInterface,
                                  ["getDpsInitsMutable"]>,
        HipDpsOpInterface,
        DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface,
                                  ["reifyResultShapes", "reifyDimOfResult"]>
      ],
      traits)> {
  code reifyDimClassDefinition = [{
    ::llvm::FailureOr<::mlir::OpFoldResult> $cppClass::reifyDimOfResult(
        ::mlir::OpBuilder &b, int resultIndex, int dim) {
      return ::llvm::cast<::mlir::hip::HipDpsOp>(getOperation())
          .reifyDimOfResult(b, resultIndex, dim);
    }
  }];
  let extraClassDefinition = reifyDimClassDefinition;
  let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
}

// Adds the single-result DPS inference convention: tensor mode returns the
// typed outs operand, while memref mode has no SSA result.
class Hip_DpsOp_WithInfer<string mnemonic,
                          string outsAccessor = "Output",
                          list<Trait> traits = []> :
    Hip_DpsOp<mnemonic, !listconcat([
        DeclareOpInterfaceMethods<InferTypeOpInterface,
                                  ["inferReturnTypes"]>
      ], traits)> {
  code inferClassDefinition = !strconcat([{
      ::llvm::LogicalResult $cppClass::inferReturnTypes(
          ::mlir::MLIRContext *,
          ::std::optional<::mlir::Location>,
          ::mlir::ValueRange operands,
          ::mlir::DictionaryAttr attrs,
          ::mlir::OpaqueProperties props,
          ::mlir::RegionRange regions,
          ::llvm::SmallVectorImpl<::mlir::Type> &results) {
        $cppClass::Adaptor adaptor(operands, attrs, props, regions);
        ::mlir::Value outs = adaptor.get}], outsAccessor, [{();
        if (auto t = ::llvm::dyn_cast<::mlir::RankedTensorType>(
                outs.getType()))
          results.push_back(t);
        return ::mlir::success();
      }
    }]);
  let extraClassDefinition = reifyDimClassDefinition # inferClassDefinition;
}

// Uses the shared HipDpsOpInterface implementation to lift the runtime shape
// from destination operands.
class Hip_DpsOp_AutoReify<string mnemonic, list<Trait> traits = []> :
    Hip_DpsOp<mnemonic, traits> {
  code reifyClassDefinition = [{
    ::mlir::LogicalResult $cppClass::reifyResultShapes(
        ::mlir::OpBuilder &b,
        ::mlir::ReifiedRankedShapedTypeDims &reified) {
      return ::llvm::cast<::mlir::hip::HipDpsOp>(getOperation())
          .reifyResultShapes(b, reified);
    }
  }];
  let extraClassDefinition = reifyDimClassDefinition # reifyClassDefinition;
}

// Combines the two common generated methods without exposing independent
// behavior switches to leaf operations.
class Hip_DpsOp_AutoReifyInfer<string mnemonic,
                               string outsAccessor = "Output",
                               list<Trait> traits = []> :
    Hip_DpsOp_WithInfer<mnemonic, outsAccessor, traits> {
  code reifyClassDefinition = [{
    ::mlir::LogicalResult $cppClass::reifyResultShapes(
        ::mlir::OpBuilder &b,
        ::mlir::ReifiedRankedShapedTypeDims &reified) {
      return ::llvm::cast<::mlir::hip::HipDpsOp>(getOperation())
          .reifyResultShapes(b, reified);
    }
  }];
  let extraClassDefinition =
      reifyDimClassDefinition # reifyClassDefinition # inferClassDefinition;
}

/// Shape behavior supplied by a dedicated semantic implementation.
class Hip_DpsOp_Semantic<string mnemonic,
                         list<Trait> traits = [],
                         string outsAccessor = "Output"> :
    Hip_DpsOp_WithInfer<mnemonic, outsAccessor, traits>;

// Parameterized sub-base for ONNX-style reduction ops (reduce_sum,
// reduce_max, reduce_prod). Auto-emits a `reifyResultShapes` body
// that calls `mlir::hip::reifyReductionShape`. The axes operand must
// have a structurally-proven constant source: inline arith.constant in
// tensor form, or constant memref.global after bufferization. Conversion
// materializes normalized, sorted axes. Verification requires one
// contiguous span because the runtime flattens that span into a single
// reduction dimension. Runtime axes are never supported. Each leaf op
// contributes only its TableGen declaration; no per-op `.cpp` thunk is
// needed.
//
// Contract on the leaf op: must declare operands `$data` (input
// tensor), `$axes` (rank-0 or rank-1 i64 tensor), `$output` (DPS init),
// integer attributes `$keepdims` and `$noop_with_empty_axes`, and the
// `$normalized_axes` dense i64 array validated against `$axes`.
class Hip_DpsOp_Reduction<string mnemonic, list<Trait> traits = []> :
    Hip_DpsOp_WithInfer<mnemonic, /*outsAccessor=*/"Output", traits> {
  code reifyClassDefinition = [{
    ::mlir::LogicalResult $cppClass::reifyResultShapes(
        ::mlir::OpBuilder &b,
        ::mlir::ReifiedRankedShapedTypeDims &reified) {
      return ::mlir::hip::reifyReductionShape(
          b, getLoc(), getData(), getAxes(),
          getKeepdims(), getNoopWithEmptyAxes(),
          getOperation(), reified);
    }
  }];
  code verifyClassDefinition = [{
    ::mlir::LogicalResult $cppClass::verify() {
      return ::mlir::hip::verifyReductionDpsOp(
          getOperation(), getData(), getAxes(), getKeepdims(),
          getNoopWithEmptyAxes());
    }
  }];
  let extraClassDefinition =
      reifyDimClassDefinition # reifyClassDefinition # inferClassDefinition #
      verifyClassDefinition;
  let hasVerifier = 1;
}

// Parameterized sub-base for elementwise NumPy-broadcast ops (add,
// mul, sub, div, min, mod, equal, less, and, where, miopen.add).
// Auto-emits a `reifyResultShapes` body that calls
// `mlir::hip::reifyBroadcastShapeFor` with the broadcast input
// operands — output element type is independent of this reify path
// and is taken from the `outs` operand by the converter.
//
// `operandGetters` is the list of ODS-generated accessor stems for
// the broadcast inputs, in right-alignment order. The auto-emitted
// body builds a `{getXX(), getYY(), ...}` initialiser list from
// these names. Most ops pass `["Lhs", "Rhs"]`; ops with non-default
// operand naming pass that naming (`["A", "B"]` for `miopen.add`,
// `["Condition", "X", "Y"]` for `where`).
//
// `outsAccessor` defaults to `"Output"`; pass the leaf op's actual
// outs accessor stem (e.g. `"C"` for `miopen.add`) so the inherited
// auto-emitted `inferReturnTypes` reads the correct outs type.
class Hip_DpsOp_Broadcast<string mnemonic, list<string> operandGetters,
                          string outsAccessor = "Output",
                          list<Trait> traits = []> :
    Hip_DpsOp_WithInfer<mnemonic, outsAccessor, traits> {
  code reifyClassDefinition = !strconcat([{
    ::mlir::LogicalResult $cppClass::reifyResultShapes(
        ::mlir::OpBuilder &b,
        ::mlir::ReifiedRankedShapedTypeDims &reified) {
      return ::mlir::hip::reifyBroadcastShapeFor(
          b, getLoc(), {
  }],
  !interleave(!foreach(g, operandGetters, "get" # g # "()"), ", "),
  [{}, getOperation(), reified);
    }
  }]);
  code verifyClassDefinition = !strconcat([{
    ::mlir::LogicalResult $cppClass::verify() {
      return ::mlir::hip::verifyBroadcastDpsOp(getOperation(), {
  }],
  !interleave(!foreach(g, operandGetters, "get" # g # "()"), ", "),
  [{});
    }
  }]);
  let extraClassDefinition =
      reifyDimClassDefinition # reifyClassDefinition # inferClassDefinition #
      verifyClassDefinition;
  let hasVerifier = 1;
}

// Accepts both ranked tensor (pre-bufferization) and memref (post-bufferization).
def Hip_TensorOrMemRef : AnyTypeOf<[AnyRankedTensor, AnyMemRef],
                                    "ranked tensor or memref">;

def Hip_AllocOp : Hip_Op<"alloc",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Allocate device memory";
  let description = [{
    Allocates device memory using HIP and returns a memref.

    Example:
    ```mlir
    %x = hip.alloc(%ctx, %N) : memref<?x128xf32, 1>
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Variadic<Index>:$dynamicSizes
  );
  let results = (outs AnyMemRef:$memref);

  let assemblyFormat = [{
    `(` $ctx (`,` $dynamicSizes^ )? `)` attr-dict `:` type($memref)
  }];
}

def Hip_GetPoolOp : Hip_Op<"get_pool",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Get a runtime-owned GPU memory pool (grow-on-demand)";
  let description = [{
    Returns a reference to one of the runtime-managed GPU memory pools.  The
    runtime guarantees that the returned buffer is at least %pool_size bytes.
    If the current pool is smaller, the runtime grows it (realloc)
    transparently.

    The optional `domain_id` attribute selects which pool to access (default
    0).  hip-pool-allocs partitions a function's pooled allocs into
    independent dominance domains and emits one hip.get_pool per domain;
    domains are independent so growing one does not affect any other.  Old
    single-pool textual MLIR (without the attribute) parses as
    `domain_id = 0`, and the printer omits the attribute when it equals the
    default — single-domain output is bit-identical to pre-multi-domain IR.

    Example:
    ```mlir
    %pool0 = hip.get_pool(%ctx, %size_a) : memref<?xi8>
    %pool1 = hip.get_pool(%ctx, %size_b) {domain_id = 1 : i64} : memref<?xi8>
    ```
  }];

  let arguments = (ins Hip_ContextType:$ctx, Index:$pool_size,
                       DefaultValuedAttr<I64Attr, "0">:$domain_id);
  let results = (outs AnyMemRef:$pool);
  let assemblyFormat = "`(` $ctx `,` $pool_size `)` attr-dict `:` type($pool)";
}

def Hip_GetHostScratchOp : Hip_Op<"get_host_scratch",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Get the runtime-owned host-mapped scratch buffer (grow-on-demand)";
  let description = [{
    Returns a reference to the runtime-managed host-mapped scratch buffer.
    Memory is hipHostMalloc(hipHostMallocMapped) — host-writable AND
    GPU-readable. Used by hip-materialize-host-scalars to redirect tiny
    host-fed memref.alloc ops away from the GPU pool (which on some targets
    is real device memory and SEGVs on host stores). Grow-on-demand semantics
    mirror hip.get_pool.

    Example:
    ```mlir
    %scratch = hip.get_host_scratch(%ctx, %size) : memref<?xi8>
    ```
  }];

  let arguments = (ins Hip_ContextType:$ctx, Index:$scratch_size);
  let results = (outs AnyMemRef:$scratch);
  let assemblyFormat = "`(` $ctx `,` $scratch_size `)` attr-dict `:` type($scratch)";
}

def Hip_AllocOutputOp : Hip_Op<"alloc_output",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Obtain a graph-output buffer from the EP output allocator";
  let description = [{
    Allocates the buffer for graph output `out_idx` via the EP-supplied output
    allocator, at the point in the graph where the output shape is known.
    Operands: runtime context + one Index per dynamic dim of the result memref
    (same operand convention as hip.alloc). The `out_idx` attribute identifies
    which graph output this buffer is.

    Unlike hip.alloc / hip.get_pool / hip.get_host_scratch, this op intentionally
    does NOT carry an Allocate memory effect: the buffer is EP/runtime-owned (it
    is a graph output), so hip-pool-allocs must not pool it. A custom pipeline
    that enables ownership-based buffer deallocation must not free it either.
    It DOES carry a Write effect so later DCE/CSE cannot drop or merge it -- see
    AllocOutputOp::getEffects in HipDialect.cpp for the rationale.

    Example:
    ```mlir
    %out = hip.alloc_output(%ctx, %M, %N) {out_idx = 0 : i64} : memref<?x?xf16>
    ```
  }];

  let arguments = (ins Hip_ContextType:$ctx, Variadic<Index>:$dynamicSizes,
                       I64Attr:$out_idx);
  let results = (outs AnyMemRef:$memref);
  let assemblyFormat = [{
    `(` $ctx (`,` $dynamicSizes^ )? `)` attr-dict `:` type($memref)
  }];
  let hasVerifier = 1;
}

// ===== Control flow ==========================================================

def Hip_LoopOp : Hip_Op<"loop",
    [AttrSizedOperandSegments,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     DeclareOpInterfaceMethods<DestinationStyleOpInterface,
                               ["getDpsInitsMutable"]>,
     DeclareOpInterfaceMethods<InferTypeOpInterface,
                               ["inferReturnTypes"]>,
     DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Outlined-body counted/conditional loop";
  let description = [{
    Runtime-driven loop with an outlined body function. Each iteration
    invokes `$body_func` with the iteration index, the current condition,
    the loop-carried values, and the captured outer-graph values. The
    body updates the loop-carried values and returns a new condition.

    Operand layout:
      - `$ctx`: `!hip.context` (always first).
      - `$max_trip_count`: `index` — the trip count `M` (ONNX Loop input 0).
      - `$cond_init`: optional `i1` — initial condition (ONNX Loop input 1).
      - `$v_init`: variadic, length must equal `$num_loop_carried` — initial
        values for loop-carried tensors/memrefs; types must match the op's
        result types.
      - `$captures`: variadic — outer-graph values used inside the body
        that are not loop-carried (e.g. constant tensors A, B).

    Result types match `$v_init` types and correspond to the loop-carried
    values after the final iteration.

    If `$cond_is_passthrough` is set, the outlining pass has verified that
    the body's yielded `cond_out` is SSA-equal to its `cond_in` block
    argument. The LLVM lowering then picks the fast-path runtime function
    `hipdnn_ep_run_counted_loop` (no per-iter device->host cond sync).
    Otherwise the slow-path `hipdnn_ep_run_loop` is used (matches ORT CUDA
    EP / MIGraphX behavior).

    Example:
    ```mlir
    %v_final = hip.loop(%ctx, %M, %cond_init)
                   iter_args(%acc_init : tensor<16x16xf32>)
                   captures(%A, %B : tensor<?x16x16xf32>,
                                     tensor<16x16xf32>)
                   -> (tensor<16x16xf32>)
                   body @loop_body_n0
                   {num_loop_carried = 1 : i32, cond_is_passthrough}
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Index:$max_trip_count,
    Optional<I1>:$cond_init,
    Variadic<Hip_TensorOrMemRef>:$v_init,
    Variadic<AnyType>:$captures,
    FlatSymbolRefAttr:$body_func,
    I32Attr:$num_loop_carried,
    UnitAttr:$cond_is_passthrough
  );

  // v_final is non-empty only at the tensor level (DPS convention).
  // After bufferization + loop-body out-param promotion, the results are
  // eliminated and v_init carries the writes via its memref descriptors.
  let results = (outs Variadic<AnyRankedTensor>:$v_final);

  let assemblyFormat = [{
    `(` $ctx `,` $max_trip_count (`,` $cond_init^)? `)`
    ( `iter_args` `(` $v_init^ `:` type($v_init) `)` )?
    ( `captures` `(` $captures^ `:` type($captures) `)` )?
    ( `->` `(` type($v_final)^ `)` )?
    `body` $body_func
    attr-dict
  }];

  let hasVerifier = 1;
}

def Hip_IfOp : Hip_Op<"if",
    [AttrSizedOperandSegments,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     DeclareOpInterfaceMethods<DestinationStyleOpInterface,
                               ["getDpsInitsMutable"]>,
     DeclareOpInterfaceMethods<InferTypeOpInterface,
                               ["inferReturnTypes"]>,
     DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Outlined then/else conditional";
  let description = [{
    Runtime-driven conditional with outlined then/else body functions.
    Exactly one branch executes per invocation; its outputs are written
    into the DPS `o_init` buffers (tensor mode) or returned via out-params
    (memref mode post-bufferization).

    Operand layout:
      - `$ctx`: `!hip.context` (always first).
      - `$cond`: `i1` — branch selector (true -> then, false -> else).
      - `$o_init`: variadic, length must equal `$num_outputs` — destination
        buffers for branch outputs; types must match the op's result types.
      - `$captures`: variadic — outer-graph values used inside either
        branch that are not subgraph block arguments.

    Result types match `$o_init` types in tensor mode.

    Example:
    ```mlir
    %out = hip.if(%ctx, %cond)
               outs(%out_init : tensor<5xf32>)
               then @main_if_then_n0 else @main_if_else_n0
               {num_outputs = 1 : i32}
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    I1:$cond,
    Variadic<Hip_TensorOrMemRef>:$o_init,
    Variadic<AnyType>:$captures,
    FlatSymbolRefAttr:$then_func,
    FlatSymbolRefAttr:$else_func,
    I32Attr:$num_outputs
  );

  let results = (outs Variadic<AnyRankedTensor>:$outputs);

  let assemblyFormat = [{
    `(` $ctx `,` $cond `)`
    ( `outs` `(` $o_init^ `:` type($o_init) `)` )?
    ( `captures` `(` $captures^ `:` type($captures) `)` )?
    ( `->` `(` type($outputs)^ `)` )?
    `then` $then_func `else` $else_func
    attr-dict
  }];

  let hasVerifier = 1;
}

def Hip_FreeOp : Hip_Op<"free",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Free device memory";
  let description = [{
    Frees device memory allocated with hip.alloc.

    Example:
    ```mlir
    hip.free(%ctx, %x) : memref<?x128xf32, 1>
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    AnyMemRef:$memref
  );

  let assemblyFormat = [{
    `(` $ctx `,` $memref `)` attr-dict `:` type($memref)
  }];
}

// ===== Constant Management ===================================================

def Hip_ConstantOp : Hip_Op<"constant", [Pure]> {
  let summary = "Policy-neutral constant tensor carrier";
  let description = [{
    Carries a statically-shaped tensor constant from a producer to
    `hip-externalize-constants`. The carrier has exactly one source:

      * `value`: an inline `DenseElementsAttr`; or
      * `location`, `offset`, and `size`: a file-backed byte range; or
      * `memory_address` and `size`: a process-local host-memory byte range.

    Producers that need a stable artifact order set `serialization_order`.
    Unordered carriers follow ordered carriers in stable module walk order.
    `symbol_name_hint` contributes only to the generated private symbol name;
    `source_name` is copied into the human-readable artifact manifest.

    Memory-address sources are process-local and non-serializable. They are
    accepted only when the production externalizer is given a FileSystem while
    the producer-owned storage is live. Textual pass invocation rejects them
    before any address is dereferenced.

    This op deliberately has no fold method. Generic canonicalization may
    reason about its purity, but only `hip-externalize-constants` may choose
    between inline and external storage.
  }];

  let arguments = (ins
    OptionalAttr<ElementsAttr>:$value,
    OptionalAttr<StrAttr>:$location,
    OptionalAttr<I64Attr>:$offset,
    OptionalAttr<I64Attr>:$memory_address,
    OptionalAttr<I64Attr>:$size,
    OptionalAttr<StrAttr>:$symbol_name_hint,
    OptionalAttr<StrAttr>:$source_name,
    OptionalAttr<I64Attr>:$serialization_order
  );
  let results = (outs AnyRankedTensor:$result);

  let extraClassDeclaration = [{
    enum class SourceKind { Inline, Memory, File };

    /// Classify the source of a verified carrier.
    SourceKind getSourceKind();
  }];

  let assemblyFormat = [{
    attr-dict `:` type($result)
  }];

  let builders = [
    OpBuilder<(ins "::mlir::RankedTensorType":$result,
                   "::mlir::DenseElementsAttr":$value), [{
      build($_builder, $_state, result, value, /*location=*/nullptr,
            /*offset=*/nullptr, /*memory_address=*/nullptr, /*size=*/nullptr,
            /*symbol_name_hint=*/nullptr, /*source_name=*/nullptr,
            /*serialization_order=*/nullptr);
    }]>,
    OpBuilder<(ins "::mlir::RankedTensorType":$result,
                   "::mlir::StringAttr":$location,
                   "::mlir::IntegerAttr":$offset,
                   "::mlir::IntegerAttr":$size), [{
      build($_builder, $_state, result, /*value=*/nullptr, location, offset,
            /*memory_address=*/nullptr, size, /*symbol_name_hint=*/nullptr,
            /*source_name=*/nullptr, /*serialization_order=*/nullptr);
    }]>,
    OpBuilder<(ins "::mlir::RankedTensorType":$result,
                   "::mlir::IntegerAttr":$memory_address,
                   "::mlir::IntegerAttr":$size), [{
      build($_builder, $_state, result, /*value=*/nullptr,
            /*location=*/nullptr, /*offset=*/nullptr, memory_address, size,
            /*symbol_name_hint=*/nullptr, /*source_name=*/nullptr,
            /*serialization_order=*/nullptr);
    }]>
  ];

  let hasVerifier = 1;
}

def Hip_GetConstantOp : Hip_Op<"get_constant", [
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Retrieve constant from GPU memory";
  let description = [{
    Retrieves a pre-uploaded constant from state->gpu_constants[index].

    Arguments:
    - ctx: HIP execution context (contains gpu_constants array)
    - index: Index in gpu_constants array (0, 1, 2, ...)

    Returns:
    - memref: GPU memory reference to the constant

    Example:
    ```mlir
    %weight = hip.get_constant(%ctx, %index_0) : memref<64x3x3x3xf32, 1>
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    I64:$index
  );
  let results = (outs AnyMemRef:$result);

  let assemblyFormat = [{
    `(` $ctx `,` $index `)` attr-dict `:` type($result)
  }];
}

// ===== Region ops (structural grouping, no runtime) =========================

def Hip_MiopenGraphOp : Hip_Op<"miopen.graph", [SingleBlock, NoTerminator]> {
  let summary = "MIOpen backend dispatch region";
  let description = [{
    Groups operations that should be dispatched to the MIOpen library.
    Purely structural -- inlined and erased during lowering.
  }];
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$body attr-dict";
}

def Hip_HipblasltGraphOp : Hip_Op<"hipblaslt.graph", [SingleBlock, NoTerminator]> {
  let summary = "hipBLASLt backend dispatch region";
  let description = [{
    Groups operations that should be dispatched to the hipBLASLt library.
    Purely structural -- inlined and erased during lowering.
  }];
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$body attr-dict";
}

// ===== MIOpen DNN Operations =================================================

def Hip_ConvOp :
    Hip_DpsOp_Semantic<"conv", /*traits=*/[OpStateOpInterface]> {
  let summary = "2D convolution operation using MIOpen (in-place)";
  let description = [{
    Performs 2D convolution using MIOpen (miopenConvolutionForward).

    Uses destination-passing style: output buffer is provided as argument,
    and the operation writes the result in-place.

    Attributes:
    - kernel_shape: Array of kernel dimensions [H, W]
    - strides: Array of stride values [H, W]
    - pads: Array of padding values [top, left, bottom, right]
    - dilations: Array of dilation values [H, W]
    - group: Number of groups for grouped convolution

    Example:
    ```mlir
    hip.conv(%ctx) ins(%input, %weights, %bias :
                       memref<1x3x224x224xf32, 1>,
                       memref<64x3x3x3xf32, 1>,
                       memref<64xf32, 1>)
                   outs(%output : memref<1x64x112x112xf32, 1>)
                   {kernel_shape = [3, 3], strides = [1, 1],
                    pads = [1, 1, 1, 1], dilations = [1, 1], group = 1}
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$input,
    Hip_TensorOrMemRef:$weights,
    Optional<Hip_TensorOrMemRef>:$bias,
    Hip_TensorOrMemRef:$output,
    I64ArrayAttr:$kernel_shape,
    I64ArrayAttr:$strides,
    I64ArrayAttr:$pads,
    I64ArrayAttr:$dilations,
    I64Attr:$group
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $input `,` $weights (`,` $bias^)? `:`
    type($input) `,` type($weights) (`,` type($bias)^)? `)`
    `outs` `(` $output `:` type($output) `)`
    attr-dict (`:` type($result_tensors)^ )?
  }];

  let extraClassDeclaration = [{
    ::mlir::Value generateOpStateInit(::mlir::OpBuilder &builder,
                                      ::mlir::Location loc,
                                      ::mlir::Value statePtr, int32_t slot) {
      return ::mlir::hip::emitOpStateConstruct(
          builder, loc, statePtr, slot, "hipdnn_ep_op_state_construct_conv",
          {});
    }
  }];

  let hasVerifier = 1;
}

def Hip_ConvTransposeOp :
    Hip_DpsOp_Semantic<"conv_transpose",
                       /*traits=*/[OpStateOpInterface],
                       /*outsAccessor=*/"Output"> {
  let summary = "2D transposed convolution / deconvolution using MIOpen "
                "(in-place)";
  let description = [{
    Performs 2D transposed convolution (a.k.a. deconvolution) using MIOpen's
    `miopenTranspose` convolution mode (`miopenConvolutionForward` with a
    transpose-mode descriptor).

    Uses destination-passing style: output buffer is provided as argument,
    operation writes result in-place.

    Tensor layouts (NCHW), matching ONNX ConvTranspose:
    - input:  [N, C, H, W]            (C = input channels)
    - weights:[C, M/group, kH, kW]    (note: input channels come first, unlike
                                       the forward Conv weight layout [M, C, ...])
    - bias:   [M]                     (optional)
    - output: [N, M, H', W']          (M = weights.dim(1) * group)

    Attributes:
    - kernel_shape:    Array of kernel dimensions [kH, kW]
    - strides:         Array of stride values [sH, sW]
    - pads:            Array of padding values [top, left, bottom, right]
    - dilations:       Array of dilation values [dH, dW]
    - output_padding:  Array of extra output padding [oH, oW] (ONNX "adjs")
    - group:           Number of groups for grouped transposed convolution

    Example:
    ```mlir
    hip.conv_transpose(%ctx) ins(%input, %weights, %bias :
                       memref<1x1x3x3xf32, 1>,
                       memref<1x2x3x3xf32, 1>,
                       memref<2xf32, 1>)
                   outs(%output : memref<1x2x5x5xf32, 1>)
                   {kernel_shape = [3, 3], strides = [1, 1],
                    pads = [0, 0, 0, 0], dilations = [1, 1],
                    output_padding = [0, 0], group = 1}
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$input,
    Hip_TensorOrMemRef:$weights,
    Optional<Hip_TensorOrMemRef>:$bias,
    Hip_TensorOrMemRef:$output,
    I64ArrayAttr:$kernel_shape,
    I64ArrayAttr:$strides,
    I64ArrayAttr:$pads,
    I64ArrayAttr:$dilations,
    I64ArrayAttr:$output_padding,
    I64Attr:$group
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $input `,` $weights (`,` $bias^)? `:`
    type($input) `,` type($weights) (`,` type($bias)^)? `)`
    `outs` `(` $output `:` type($output) `)`
    attr-dict (`:` type($result_tensors)^ )?
  }];

  let extraClassDeclaration = [{
    ::mlir::Value generateOpStateInit(::mlir::OpBuilder &builder,
                                      ::mlir::Location loc,
                                      ::mlir::Value statePtr, int32_t slot) {
      return ::mlir::hip::emitOpStateConstruct(
          builder, loc, statePtr, slot, "hipdnn_ep_op_state_construct_conv",
          {});
    }
  }];

  let hasVerifier = 1;
}

// ===== hipBLASLt ops =========================================================

// Semantic MatMul-family ops recover result dimensions from their inputs rather
// than echoing the outs shape, so reification is hand-written in
// `HipReifyResultShapesImpl.cpp`. Their family base still emits the common
// single-result `inferReturnTypes` body from the typed outs operand.
def Hip_MatmulOp :
    Hip_DpsOp_Semantic<"matmul", /*traits=*/[OpStateOpInterface]> {
  let summary = "Matrix multiplication (supports batched N-D x N-D)";
  let description = [{
    Performs matrix multiplication: output = A @ B
    Supports batched operation (leading dimensions are batch dimensions
    with NumPy-style broadcast).

    Uses destination-passing style: output buffer is provided as argument.

    Shape contract:
      A:      [..., M, K]
      B:      [..., K, N]
      output: [broadcast(A.batch, B.batch), M, N]

    `?` (kDynamic) is supported for M, N, and structurally representable batch
    dimensions; dynamic contraction K is unsupported. `reifyResultShapes`
    lifts supported dynamic dims into `tensor.dim` / `memref.dim` of the
    corresponding operand. See
    [docs/design/hip-shape-inference.md](docs/design/hip-shape-inference.md)
    for the dim-source contract and the consuming pipeline.

    Optional `transA` / `transB` (0/1) apply a last-dimension swap on the
    corresponding operand before multiply — the compile-time fusion of
    `Transpose(perm=[..,r,r-2])` into MatMul for attention Q @ K^T. When
    unset they default to 0 (standard onnx.MatMul layout: A [...,M,K],
    B [...,K,N]).

    Example (2D):
    ```mlir
    hip.matmul(%ctx) ins(%A, %B : memref<128x4096xf32, 1>,
                                  memref<4096x1024xf32, 1>)
                     outs(%output : memref<128x1024xf32, 1>)
    ```

    Example (3D batched):
    ```mlir
    hip.matmul(%ctx) ins(%A, %B : memref<1x128x4096xf16, 1>,
                                  memref<4096x1024xf16, 1>)
                     outs(%output : memref<1x128x1024xf16, 1>)
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$A,
    Hip_TensorOrMemRef:$B,
    Hip_TensorOrMemRef:$output,
    DefaultValuedAttr<I64Attr, "0">:$transA,
    DefaultValuedAttr<I64Attr, "0">:$transB
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $A `,` $B `:` type($A) `,` type($B) `)`
    `outs` `(` $output `:` type($output) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];

  // OpStateOpInterface: construct this matmul instance's per-session state
  // (MatmulState), which holds a shared_ptr to the device-wide hipBLASLt algo
  // table. No compile-time params — the table fills lazily per call. The
  // construct fn stores the state into op_states[slot] itself.
  let extraClassDeclaration = [{
    ::mlir::Value generateOpStateInit(::mlir::OpBuilder &builder,
                                      ::mlir::Location loc,
                                      ::mlir::Value statePtr, int32_t slot) {
      return ::mlir::hip::emitOpStateConstruct(
          builder, loc, statePtr, slot, "hipdnn_ep_op_state_construct_matmul",
          {});
    }
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Quantized Operations
//===----------------------------------------------------------------------===//

def Hip_MatMulNBitsOp :
    Hip_DpsOp_Semantic<"matmul_nbits",
                       /*traits=*/[AttrSizedOperandSegments,
                                   OpStateOpInterface]> {
  let summary = "Quantized N-bit matrix multiplication (com.microsoft MatMulNBits)";
  let description = [{
    Performs matrix multiplication with N-bit quantized weights.
    Dequantizes packed int4/int8 weights using per-block scale factors,
    then computes Y = A @ dequantized_weight^T.

    Implements the com.microsoft MatMulNBits operator (inputs 3-6).

    Weight packing: B is stored as uint8 with shape (N, k_blocks, blob_size)
    where k_blocks = ceil(K / block_size), blob_size = block_size * bits / 8.
    For 4-bit: lower nibble = first value, upper nibble = second value.

    Dequantization: dequantized = (quantized - zero_point) * scale
    Default zero_point = 2^(bits-1) (= 8 for 4-bit).

    Optional inputs per spec:
      zero_points: per-block zero points for dequantization
      g_idx: GPTQ group ordering indices (deprecated)
      bias: output bias vector [N]

    Uses destination-passing style: output buffer is provided as argument.

    Example:
    ```mlir
    hip.matmul_nbits(%handle) ins(%A, %B, %scales :
        memref<1x128x2880xf16, 1>, memref<5120x90x16xui8, 1>,
        memref<5120x90xf16, 1>)
        outs(%output : memref<1x128x5120xf16, 1>)
        {K = 2880, N = 5120, bits = 4, block_size = 32, accuracy_level = 4}
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$handle,
    Hip_TensorOrMemRef:$A,                      // activation [..., K]
    Hip_TensorOrMemRef:$B,                      // packed uint8 [N, k_blocks, blob_size]
    Hip_TensorOrMemRef:$scales,                 // dequant scales [N, k_blocks]
    Optional<Hip_TensorOrMemRef>:$zero_points,  // dequant zero points (optional)
    Optional<Hip_TensorOrMemRef>:$g_idx,        // GPTQ group indices (optional, deprecated)
    Optional<Hip_TensorOrMemRef>:$bias,         // output bias [N] (optional)
    Hip_TensorOrMemRef:$output,                 // result [..., N]
    I64Attr:$K,
    I64Attr:$N,
    I64Attr:$bits,
    I64Attr:$block_size,
    I64Attr:$accuracy_level,
    I64Attr:$zp_elem_size  // zero_points element size: 1=uint8 packed, 2=fp16
  );

  let assemblyFormat = [{
    `(` $handle `)` `ins` `(` $A `,` $B `,` $scales `:`
    type($A) `,` type($B) `,` type($scales) `)`
    (`zero_points` `(` $zero_points^ `:` type($zero_points) `)`)?
    (`g_idx` `(` $g_idx^ `:` type($g_idx) `)`)?
    (`bias` `(` $bias^ `:` type($bias) `)`)?
    `outs` `(` $output `:` type($output) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];

  // OpStateOpInterface: construct this MatMulNBits instance's per-session
  // zero_points unpack cache (MatmulNbitsState). No compile-time params — the
  // cache fills lazily per call. The construct fn stores the state into
  // op_states[slot] itself.
  let extraClassDeclaration = [{
    ::mlir::Value generateOpStateInit(::mlir::OpBuilder &builder,
                                      ::mlir::Location loc,
                                      ::mlir::Value statePtr, int32_t slot) {
      return ::mlir::hip::emitOpStateConstruct(
          builder, loc, statePtr, slot,
          "hipdnn_ep_op_state_construct_matmul_nbits", {});
    }
  }];
  let hasVerifier = 1;
}

def Hip_QMoEOp : Hip_DpsOp_WithInfer<"qmoe", "Output", [AttrSizedOperandSegments]> {
  let summary = "Quantized Mixture-of-Experts (com.microsoft QMoE)";
  let description = [{
    Implements the com.microsoft QMoE operator (ONNX v1.25+ compatible): quantized
    Mixture-of-Experts with top-k expert routing and optional SwiGLU activation.

    Routes each token to its top-k experts, performs quantized linear
    projections (fc1 gate+up, fc2 down) per expert with int4 dequantization,
    applies activation (SwiGLU by default), and combines expert outputs.

    SwiGLU formula (when swiglu_fusion=1, interleaved in fc1 output):
      g, l = interleaved_split(fc1_out)
      G = clamp(g, max=swiglu_limit)
      L = clamp(l, -swiglu_limit, swiglu_limit)
      result = G * sigmoid(activation_alpha * G) * (L + activation_beta)

    Weight packing follows MatMulNBits convention: uint8 packed,
    dequantized = (quantized - 8) * scale for 4-bit weights.

    Optional inputs per spec:
      fc1_experts_bias, fc2_experts_bias: bias vectors (optional)
      fc3_*: unfused SwiGLU second GEMM weights/scales/bias (optional)
      fc1/fc2/fc3_zero_points: per-block zero points (optional)
      router_weights: alternative aggregation weights (optional, ONNX v1.25+, takes precedence over router_probs)

    Uses destination-passing style: output buffer is provided as argument.

    Example:
    ```mlir
    hip.qmoe(%handle) ins(%input, %router,
                           %fc1w, %fc1s, %fc2w, %fc2s :
        memref<1x128x2880xf16, 1>, memref<128x32xf16, 1>,
        memref<32x5760x1440xui8, 1>, memref<32x5760x90xf16, 1>,
        memref<32x2880x1440xui8, 1>, memref<32x2880x90xf16, 1>)
        fc1_bias(%fc1b : memref<32x5760xf16, 1>)
        fc2_bias(%fc2b : memref<32x2880xf16, 1>)
        outs(%output : memref<1x128x2880xf16, 1>)
        {expert_weight_bits = 4, k = 4, block_size = 32, ...}
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$handle,
    Hip_TensorOrMemRef:$input,                  // [batch, seq, hidden]
    Hip_TensorOrMemRef:$router_probs,           // [num_tokens, num_experts]
    Hip_TensorOrMemRef:$fc1_experts_weights,    // [num_experts, fusion*inter, hidden/pack]
    Hip_TensorOrMemRef:$fc1_scales,             // [num_experts, fusion*inter, hidden/bs]
    Hip_TensorOrMemRef:$fc2_experts_weights,    // [num_experts, hidden, inter/pack]
    Hip_TensorOrMemRef:$fc2_scales,             // [num_experts, hidden, inter/bs]
    Optional<Hip_TensorOrMemRef>:$fc1_experts_bias,       // [num_experts, fusion*inter]
    Optional<Hip_TensorOrMemRef>:$fc2_experts_bias,       // [num_experts, hidden]
    Optional<Hip_TensorOrMemRef>:$fc3_experts_weights,    // [num_experts, inter, hidden/pack]
    Optional<Hip_TensorOrMemRef>:$fc3_scales,             // [num_experts, inter, hidden/bs]
    Optional<Hip_TensorOrMemRef>:$fc3_experts_bias,       // [num_experts, inter]
    Optional<Hip_TensorOrMemRef>:$fc1_zero_points,        // fc1 dequant zero points
    Optional<Hip_TensorOrMemRef>:$fc2_zero_points,        // fc2 dequant zero points
    Optional<Hip_TensorOrMemRef>:$fc3_zero_points,        // fc3 dequant zero points
    Optional<Hip_TensorOrMemRef>:$router_weights,         // [num_tokens, num_experts] alternative routing weights (ONNX v1.25+)
    Hip_TensorOrMemRef:$output,                 // DPS init [batch, seq, hidden]
    I64Attr:$expert_weight_bits,
    I64Attr:$k,
    I64Attr:$block_size,
    I64Attr:$normalize_routing_weights,
    I64Attr:$swiglu_fusion,
    I64Attr:$use_sparse_mixer,
    F32Attr:$activation_alpha,
    F32Attr:$activation_beta,
    F32Attr:$swiglu_limit,
    StrAttr:$activation_type
  );

  let assemblyFormat = [{
    `(` $handle `)` `ins` `(`
        $input `,` $router_probs `,`
        $fc1_experts_weights `,` $fc1_scales `,`
        $fc2_experts_weights `,` $fc2_scales `:`
        type($input) `,` type($router_probs) `,`
        type($fc1_experts_weights) `,` type($fc1_scales) `,`
        type($fc2_experts_weights) `,` type($fc2_scales)
    `)`
    (`fc1_bias` `(` $fc1_experts_bias^ `:` type($fc1_experts_bias) `)`)?
    (`fc2_bias` `(` $fc2_experts_bias^ `:` type($fc2_experts_bias) `)`)?
    (`fc3_weights` `(` $fc3_experts_weights^ `:` type($fc3_experts_weights) `)`)?
    (`fc3_scales` `(` $fc3_scales^ `:` type($fc3_scales) `)`)?
    (`fc3_bias` `(` $fc3_experts_bias^ `:` type($fc3_experts_bias) `)`)?
    (`fc1_zero_points` `(` $fc1_zero_points^ `:` type($fc1_zero_points) `)`)?
    (`fc2_zero_points` `(` $fc2_zero_points^ `:` type($fc2_zero_points) `)`)?
    (`fc3_zero_points` `(` $fc3_zero_points^ `:` type($fc3_zero_points) `)`)?
    (`router_weights` `(` $router_weights^ `:` type($router_weights) `)`)?
    `outs` `(` $output `:` type($output) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_GatherBlockQuantizedOp : Hip_DpsOp_AutoReify<"gather_block_quantized"> {
  let summary = "Gather + block-wise dequantize "
                "(com.microsoft GatherBlockQuantized)";
  let description = [{
    Implements the com.microsoft GatherBlockQuantized operator: gather rows
    from a block-wise quantized constant `data` along `gather_axis` using
    integer `indices`, and dequantize the gathered rows using per-block
    `scales` and optional `zero_points`.

    `data` is a constant tensor of rank `r >= 1` quantized along
    `quantize_axis` with the given `block_size`. For 4-bit, two values are
    packed per byte (low nibble first, high nibble second); for 8-bit one
    value per byte. `scales` (and `zero_points`, when present) share the
    layout of `data` except along `quantize_axis`, where their extent is
    `data.shape[quantize_axis] / block_size`. When `zero_points` is absent,
    the default is `0` for int4/uint4 and `2^(bits-1)` for uint8.

    Output rank is `q + (r - 1)` where `q = rank(indices)`. Element type of
    `output` matches `scales` (T2: f32 / f16 / bf16). For uint8 `data`,
    `gather_axis` must be 0.

    Optional unit attribute `unsigned_quant_storage`: set when
    convert-onnx-to-hip identifies UINT4/UINT8
    packed byte storage (MLIR cannot represent ui4). Lowering passes
    HIPDNN_EP_DATATYPE_UINT8 to the runtime instead of the signless-i8 default
    (INT8 / signed int4). Absent for signed INT4 or explicit ui8/i8 storage.

    Uses destination-passing style: the `output` buffer is provided as an
    argument. The runtime is responsible for fetching the per-element
    quantized value (handling sub-byte unpacking when `bits == 4`),
    dequantizing it via `(q - zp) * scale`, and writing the result.

    Example:
    ```mlir
    hip.gather_block_quantized(%ctx)
        ins(%data, %indices, %scales :
            memref<2048x96xui8, 1>,
            memref<8xi64, 1>,
            memref<2048x12xf16, 1>)
        zero_points(%zp : memref<2048x12xui8, 1>)
        outs(%out : memref<8x96xf16, 1>)
        {bits = 4, block_size = 16, gather_axis = 0, quantize_axis = 1}
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$data,                   // packed quantized weights
    Hip_TensorOrMemRef:$indices,                // i32/i64 indices
    Hip_TensorOrMemRef:$scales,                 // dequant scales (T2)
    Optional<Hip_TensorOrMemRef>:$zero_points,  // dequant zero points
    Hip_TensorOrMemRef:$output,                 // DPS init / dequantized output
    I64Attr:$bits,
    I64Attr:$block_size,
    I64Attr:$gather_axis,
    I64Attr:$quantize_axis,
    UnitAttr:$unsigned_quant_storage
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $data `,` $indices `,` $scales `:`
        type($data) `,` type($indices) `,` type($scales) `)`
    (`zero_points` `(` $zero_points^ `:` type($zero_points) `)`)?
    `outs` `(` $output `:` type($output) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

// ===== MIOpen ops ============================================================

def Hip_RmsNormOp : Hip_DpsOp_WithInfer<"rms_norm", "Output"> {
  let summary = "RMS normalization";
  let description = [{
    Applies simplified (RMS) layer normalization.

    Formula: output = input / sqrt(mean(input^2) + epsilon) * scale

    Uses destination-passing style: output buffer is provided as argument.

    Example:
    ```mlir
    hip.rms_norm(%ctx)
        ins(%input, %scale : memref<1x128x4096xf16, 1>, memref<4096xf16, 1>)
        outs(%output : memref<1x128x4096xf16, 1>)
        {axis = -1 : i64, epsilon = 9.99999974e-06 : f32, stash_type = 1 : i64}
    ```

    Supports dynamic shapes:
    ```mlir
    hip.rms_norm(%ctx)
        ins(%input, %scale : memref<?x?xf16, 1>, memref<?xf16, 1>)
        outs(%output : memref<?x?xf16, 1>)
        {axis = -1 : i64, epsilon = 1.0e-05 : f32, stash_type = 1 : i64}
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$input,
    Hip_TensorOrMemRef:$scale,
    Hip_TensorOrMemRef:$output,
    I64Attr:$axis,
    F32Attr:$epsilon,
    I64Attr:$stash_type
  );

  // result_tensors inherited from Hip_DpsOp
  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $input `,` $scale `:` type($input) `,` type($scale) `)`
    `outs` `(` $output `:` type($output) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];

  let hasVerifier = 1;
}

def Hip_LayerNormOp :
    Hip_DpsOp<"layer_norm", /*traits=*/[AttrSizedOperandSegments]> {
  let summary = "Standard layer normalization (ONNX LayerNormalization opset 17)";
  let description = [{
    Full ONNX LayerNormalization (opset 17+) with mean subtraction and an
    optional bias term, fused into a single runtime call.

    Stage 1 (in stash_type precision):
      Mean       = ReduceMean(X, axes=[axis ... rank-1])
      D          = X - Mean
      Var        = ReduceMean(D * D, axes=[axis ... rank-1])
      InvStdDev  = 1 / sqrt(Var + epsilon)
      Normalized = D * InvStdDev   (cast back to T)

    Stage 2 (in T precision):
      NormalizedScaled = Normalized * Scale
      Y                = NormalizedScaled + B   (only when B is present)

    Inputs (2-3, MS spec order):
      input  (required, T) - N-D tensor to normalize
      scale  (required, T) - unidirectional broadcastable scale
      bias   (optional, T) - unidirectional broadcastable bias

    DPS output buffers (1-3, Variadic outputs):
      outputs[0] = output     (required, T) - normalized result, same shape as input
      outputs[1] = mean       (optional, U) - saved mean (training-only)
      outputs[2] = inv_std    (optional, U) - saved inverse standard deviation
                                              (training-only)

    Attributes:
      axis       - first normalization dimension (negative = from rank-1, default -1)
      epsilon    - small value added to variance (default 1e-5)
      stash_type - internal computation precision (1 = float, default)

    The runtime treats all dims < axis as the "row" dimension and all dims >= axis
    as the "feature" dimension being normalized. mean / inv_std outputs may be
    nullptr in the runtime call when the corresponding optional output buffer
    is omitted.

    Example (inference, output only):
    ```mlir
    hip.layer_norm(%ctx)
        ins(%input, %scale, %bias :
            memref<1x128x4096xf16, 1>, memref<4096xf16, 1>, memref<4096xf16, 1>)
        outs(%output : memref<1x128x4096xf16, 1>)
        {axis = -1 : i64, epsilon = 9.99999974e-06 : f32, stash_type = 1 : i64}
    ```

    Example (training, all 3 outputs, no bias):
    ```mlir
    hip.layer_norm(%ctx)
        ins(%input, %scale :
            memref<1x128x4096xf16, 1>, memref<4096xf16, 1>)
        outs(%output, %mean, %inv_std :
            memref<1x128x4096xf16, 1>,
            memref<1x128x1xf32, 1>, memref<1x128x1xf32, 1>)
        {axis = -1 : i64, epsilon = 1.0e-05 : f32, stash_type = 1 : i64}
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,

    // ONNX spec inputs 1-2 (required)
    Hip_TensorOrMemRef:$input,
    Hip_TensorOrMemRef:$scale,

    // ONNX spec input 3 (optional)
    Optional<Hip_TensorOrMemRef>:$bias,

    // DPS output buffers (1-3): [output] | [output, mean] | [output, mean, inv_std]
    Variadic<Hip_TensorOrMemRef>:$outputs,

    // Attributes (mirror ONNX LayerNormalization)
    DefaultValuedAttr<I64Attr, "-1">:$axis,
    DefaultValuedAttr<F32Attr, "9.99999974E-6">:$epsilon,
    DefaultValuedAttr<I64Attr, "1">:$stash_type
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(`
        $input `,` $scale
        ( `,` $bias^ )?
        `:` type($input) `,` type($scale)
        ( `,` type($bias)^ )?
    `)`
    `outs` `(` $outputs `:` type($outputs) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
  let hasVerifier = 1;
}

def Hip_InstanceNormOp : Hip_DpsOp_AutoReifyInfer<"instance_norm"> {
  let summary = "ONNX InstanceNormalization (per-instance per-channel mean/var)";
  let description = [{
    Carries out instance normalization as described in
    https://arxiv.org/abs/1607.08022 / ONNX InstanceNormalization:

      y = scale * (x - mean) / sqrt(variance + epsilon) + B

    Mean and variance are computed over the spatial axes of each (N, C)
    slice. Input rank is at least 3: (N, C, D1, ..., Dn). Scale and B are
    1-D tensors of length C.

    Example:
    ```mlir
    hip.instance_norm(%ctx)
        ins(%input, %scale, %bias :
            memref<2x3x8x8xf32, 1>, memref<3xf32, 1>, memref<3xf32, 1>)
        outs(%output : memref<2x3x8x8xf32, 1>)
        {epsilon = 1.000000e-05 : f32}
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$input,
    Hip_TensorOrMemRef:$scale,
    Hip_TensorOrMemRef:$bias,
    Hip_TensorOrMemRef:$output,
    DefaultValuedAttr<F32Attr, "1.000000e-05">:$epsilon
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(`
        $input `,` $scale `,` $bias `:`
        type($input) `,` type($scale) `,` type($bias)
    `)`
    `outs` `(` $output `:` type($output) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
  let hasVerifier = 1;
}

def Hip_SkipRmsNormOp :
    Hip_DpsOp<"skip_rms_norm", [AttrSizedOperandSegments]> {
  let summary = "Fused skip + RMS normalization (com.microsoft.SkipSimplifiedLayerNormalization)";
  let description = [{
    Full alignment with Microsoft ONNX Runtime SkipSimplifiedLayerNormalization
    specification (v1).

    Performs fused skip connection addition followed by RMS normalization:
      input_skip_bias_sum = input + skip [+ bias]
      output = RMSNorm(input_skip_bias_sum) * gamma

    RMSNorm(x) = x / sqrt(mean(x^2) + epsilon)

    Inputs (3-4):
      1. input   (required, T) - 3D or 2D input tensor
      2. skip    (required, T) - skip connection tensor (same shape as input)
      3. gamma   (required, T) - 1D normalization weight (hidden_size)
      4. bias    (optional, T) - 1D bias (hidden_size)

    DPS output buffers (1-2, passed as Variadic `outputs`):
      outputs[0] = output             (required, T) - normalized result
      outputs[1] = input_skip_bias_sum (optional, T) - residual sum (input + skip [+ bias])

    Uses destination-passing style: output buffers are provided as arguments.

    Example (without bias, 1 output):
    ```mlir
    hip.skip_rms_norm(%ctx)
        ins(%input, %skip, %gamma : memref<1x128x4096xf16, 1>,
                                     memref<1x128x4096xf16, 1>,
                                     memref<4096xf16, 1>)
        outs(%output : memref<1x128x4096xf16, 1>)
        {epsilon = 1.0e-05 : f32}
    ```

    Example (with bias, 2 outputs):
    ```mlir
    hip.skip_rms_norm(%ctx)
        ins(%input, %skip, %gamma, %bias :
            memref<1x128x4096xf16, 1>, memref<1x128x4096xf16, 1>,
            memref<4096xf16, 1>, memref<4096xf16, 1>)
        outs(%output, %input_skip_bias_sum : memref<1x128x4096xf16, 1>,
                                              memref<1x128x4096xf16, 1>)
        {epsilon = 1.0e-05 : f32}
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,

    // MS spec inputs 1-3 (required)
    Hip_TensorOrMemRef:$input,
    Hip_TensorOrMemRef:$skip,
    Hip_TensorOrMemRef:$gamma,

    // MS spec input 4 (optional)
    Optional<Hip_TensorOrMemRef>:$bias,

    // DPS output buffers: [output] or [output, input_skip_bias_sum]
    Variadic<Hip_TensorOrMemRef>:$outputs,

    // Attributes
    F32Attr:$epsilon
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(`
        $input `,` $skip `,` $gamma
        ( `,` $bias^ )?
        `:` type($input) `,` type($skip) `,` type($gamma)
        ( `,` type($bias)^ )?
    `)`
    `outs` `(` $outputs `:` type($outputs) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];

  let hasVerifier = 1;
}

def Hip_RopeOp : Hip_DpsOp_WithInfer<"rope"> {
  let summary = "Rotary position embedding (RoPE)";
  let description = [{
    Applies rotary position embedding to the input tensor.

    Uses destination-passing style: output buffer is provided as argument.

    `position_ids` is OPTIONAL, mirroring the two forms of the ONNX
    RotaryEmbedding operator:
    - Present (com.microsoft.RotaryEmbedding / native with position_ids):
      cos_cache/sin_cache are a 2D lookup table `[max_position, rotary_dim/2]`
      indexed by `position_ids[b, s]`.
    - Absent (native ai.onnx RotaryEmbedding opset>=23 without position_ids):
      cos_cache/sin_cache are already position-expanded to
      `[batch, seq, rotary_dim/2]`, so the runtime indexes them directly by
      the flattened token position `b*seq + s`. (HF exports precompute the
      per-token cos/sin via a Gather, leaving the RotaryEmbedding node with
      no position_ids operand.)

    Attributes:
    - interleaved: Rotary embedding interleaved format (0 or 1)
    - num_heads: Number of attention heads
    - rotary_embedding_dim: Dimension of rotary embedding

    Example:
    ```mlir
    hip.rope(%ctx)
        ins(%input, %position_ids, %cos_cache, %sin_cache :
            memref<1x128x4096xf16, 1>, memref<1x128xi64, 1>,
            memref<131072x64xf16, 1>, memref<131072x64xf16, 1>)
        outs(%output : memref<1x128x4096xf16, 1>)
        {interleaved = 0 : i64, num_heads = 32 : i64, rotary_embedding_dim = 128 : i64}
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$input,
    Optional<Hip_TensorOrMemRef>:$position_ids,
    Hip_TensorOrMemRef:$cos_cache,
    Hip_TensorOrMemRef:$sin_cache,
    Hip_TensorOrMemRef:$output,
    I64Attr:$interleaved,
    I64Attr:$num_heads,
    I64Attr:$rotary_embedding_dim
  );

  // position_ids is the only optional operand; MLIR resolves its presence from
  // the operand count (no AttrSizedOperandSegments needed). Each optional group
  // must LEAD with the `,` literal: a group's first parsable element has to be a
  // literal / variable / custom directive, so a bare `type(...)` directive
  // cannot open the type-list group. Anchoring on $position_ids omits it from
  // both the value and type lists when absent; the present-form IR prints
  // identically to before.
  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(`
        $input (`,` $position_ids^)? `,` $cos_cache `,` $sin_cache `:`
        type($input) (`,` type($position_ids)^)? `,` type($cos_cache) `,` type($sin_cache)
    `)`
    `outs` `(` $output `:` type($output) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_MiopenAddOp : Hip_DpsOp_Broadcast<"miopen.add",
                                          /*operandGetters=*/["A", "B"],
                                          /*outsAccessor=*/"C"> {
  let summary = "Element-wise add via miopenOpTensor(miopenTensorOpAdd)";
  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$A,
    Hip_TensorOrMemRef:$B,
    Hip_TensorOrMemRef:$C
  );
  // result_tensors inherited from Hip_DpsOp
  let hasCustomAssemblyFormat = 1;
}

def Hip_MulOp : Hip_DpsOp_Broadcast<"mul", /*operandGetters=*/["Lhs", "Rhs"],
                                    /*outsAccessor=*/"Output",
                                    /*traits=*/[OpStateOpInterface]> {
  let summary = "Elementwise multiplication";

  // OpStateOpInterface: hip.mul/add/min/max all lower to wrap_elementwise.
  // The op-state slot itself is unused (OpTensorState is an empty struct);
  // it exists only because every OpStateOpInterface op must construct one.
  let extraClassDeclaration = [{
    ::mlir::Value generateOpStateInit(::mlir::OpBuilder &builder,
                                      ::mlir::Location loc,
                                      ::mlir::Value statePtr, int32_t slot) {
      return ::mlir::hip::emitOpStateConstruct(
          builder, loc, statePtr, slot, "hipdnn_ep_op_state_construct_optensor",
          {});
    }
  }];
  let description = [{
    Performs elementwise multiplication: output = lhs * rhs

    Uses destination-passing style: output buffer is provided as argument.

    Example:
    ```mlir
    hip.mul(%ctx) ins(%lhs, %rhs : memref<1x128x14336xf16, 1>,
                                   memref<1x128x14336xf16, 1>)
                  outs(%output : memref<1x128x14336xf16, 1>)
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$lhs,
    Hip_TensorOrMemRef:$rhs,
    Hip_TensorOrMemRef:$output
  );

  // result_tensors inherited from Hip_DpsOp
  let hasCustomAssemblyFormat = 1;
}

def Hip_AddOp : Hip_DpsOp_Broadcast<"add", /*operandGetters=*/["Lhs", "Rhs"],
                                    /*outsAccessor=*/"Output",
                                    /*traits=*/[OpStateOpInterface]> {
  let summary = "Elementwise addition";

  // OpStateOpInterface: shared OpTensorState (see hip.mul).
  let extraClassDeclaration = [{
    ::mlir::Value generateOpStateInit(::mlir::OpBuilder &builder,
                                      ::mlir::Location loc,
                                      ::mlir::Value statePtr, int32_t slot) {
      return ::mlir::hip::emitOpStateConstruct(
          builder, loc, statePtr, slot, "hipdnn_ep_op_state_construct_optensor",
          {});
    }
  }];
  let description = [{
    Performs elementwise addition: output = lhs + rhs

    Uses destination-passing style: output buffer is provided as argument.

    Example:
    ```mlir
    hip.add(%ctx) ins(%lhs, %rhs : memref<1x128x32xf16, 1>,
                                   memref<1x128x32xf16, 1>)
                  outs(%output : memref<1x128x32xf16, 1>)
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$lhs,
    Hip_TensorOrMemRef:$rhs,
    Hip_TensorOrMemRef:$output
  );

  // result_tensors inherited from Hip_DpsOp
  let hasCustomAssemblyFormat = 1;
}

def Hip_MinOp : Hip_DpsOp_Broadcast<"min", /*operandGetters=*/["Lhs", "Rhs"],
                                    /*outsAccessor=*/"Output",
                                    /*traits=*/[OpStateOpInterface]> {
  let summary = "Elementwise minimum";

  // OpStateOpInterface: shared OpTensorState (see hip.mul).
  let extraClassDeclaration = [{
    ::mlir::Value generateOpStateInit(::mlir::OpBuilder &builder,
                                      ::mlir::Location loc,
                                      ::mlir::Value statePtr, int32_t slot) {
      return ::mlir::hip::emitOpStateConstruct(
          builder, loc, statePtr, slot, "hipdnn_ep_op_state_construct_optensor",
          {});
    }
  }];
  let description = [{
    Performs elementwise minimum: output = min(lhs, rhs)

    Supports multidirectional (NumPy-style) broadcasting via the runtime's
    wrap_elementwise dispatch.

    Uses destination-passing style: output buffer is provided as argument.

    Example:
    ```mlir
    hip.min(%ctx) ins(%lhs, %rhs : memref<3xf32, 1>,
                                   memref<3xf32, 1>)
                  outs(%output : memref<3xf32, 1>)
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$lhs,
    Hip_TensorOrMemRef:$rhs,
    Hip_TensorOrMemRef:$output
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $lhs `,` $rhs `:` type($lhs) `,` type($rhs) `)`
    `outs` `(` $output `:` type($output) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_MaxOp : Hip_DpsOp_Broadcast<"max",
                                    /*operandGetters=*/["Lhs", "Rhs"],
                                    /*outsAccessor=*/"Output",
                                    /*traits=*/[OpStateOpInterface]> {
  let summary = "Elementwise maximum";

  // OpStateOpInterface: shared OpTensorState (see hip.mul).
  let extraClassDeclaration = [{
    ::mlir::Value generateOpStateInit(::mlir::OpBuilder &builder,
                                      ::mlir::Location loc,
                                      ::mlir::Value statePtr, int32_t slot) {
      return ::mlir::hip::emitOpStateConstruct(
          builder, loc, statePtr, slot, "hipdnn_ep_op_state_construct_optensor",
          {});
    }
  }];
  let description = [{
    Performs elementwise maximum: output = max(lhs, rhs)

    Supports multidirectional (NumPy-style) broadcasting via the runtime's
    wrap_elementwise dispatch.

    Uses destination-passing style: output buffer is provided as argument.

    Example:
    ```mlir
    hip.max(%ctx) ins(%lhs, %rhs : memref<3xf32, 1>,
                                   memref<3xf32, 1>)
                  outs(%output : memref<3xf32, 1>)
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$lhs,
    Hip_TensorOrMemRef:$rhs,
    Hip_TensorOrMemRef:$output
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $lhs `,` $rhs `:` type($lhs) `,` type($rhs) `)`
    `outs` `(` $output `:` type($output) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_MiopenSoftmaxOp : Hip_DpsOp_AutoReifyInfer<"miopen.softmax"> {
  let summary = "Softmax via miopenSoftmaxForward_V2 (row-wise, last dim)";
  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$input,
    Hip_TensorOrMemRef:$output
  );
  // result_tensors inherited from Hip_DpsOp
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

// ===== Custom HIP kernel ops (no library equivalent) =========================

def Hip_TransposeOp : Hip_DpsOp_WithInfer<"transpose"> {
  let summary = "N-D transpose with arbitrary permutation (ONNX Transpose)";
  let description = [{
    Permutes the dimensions of the input tensor according to the `perm`
    attribute. Output dim i corresponds to input dim perm[i]. Implements
    the ONNX Transpose operator.

    The `perm` attribute is a list of integers and must be a permutation of
    [0, 1, ..., rank-1]. Its length must equal the rank of the input. When
    the ONNX `perm` attribute is omitted, the converter materializes the
    default reverse permutation `[rank-1, ..., 0]` here.

    Uses destination-passing style: output buffer is provided as argument.

    Example:
    ```mlir
    hip.transpose(%ctx) ins(%input : memref<1x2x3xf32, 1>)
                        outs(%output : memref<2x3x1xf32, 1>)
                        {perm = [1, 2, 0]}
    ```
  }];
  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$input,
    Hip_TensorOrMemRef:$output,
    I64ArrayAttr:$perm
  );
  // result_tensors inherited from Hip_DpsOp
  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $input `:` type($input) `)`
    `outs` `(` $output `:` type($output) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
  let hasVerifier = 1;
}

def Hip_GatherOp : Hip_DpsOp_WithInfer<"gather"> {
  let summary = "Gather elements along an axis";
  let description = [{
    Gathers elements from data tensor along the given axis using indices.
    Implements the ONNX Gather operator.

    Uses destination-passing style: output buffer is provided as argument.

    Example:
    ```mlir
    hip.gather(%ctx) ins(%data, %indices : memref<3x4xf32, 1>, memref<2xi64, 1>)
                     outs(%output : memref<2x4xf32, 1>) {axis = 0 : i64}
    ```
  }];
  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$data,
    Hip_TensorOrMemRef:$indices,
    Hip_TensorOrMemRef:$output,
    DefaultValuedAttr<I64Attr, "0">:$axis
  );
  // result_tensors inherited from Hip_DpsOp
  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $data `,` $indices `:` type($data) `,` type($indices) `)`
    `outs` `(` $output `:` type($output) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_OneHotOp : Hip_DpsOp_WithInfer<"one_hot"> {
  let summary = "Expand indices into a one-hot tensor along an axis";
  let description = [{
    Implements ONNX OneHot: inserts a `depth`-sized axis into the output
    and writes `on_value` at index positions from `indices`, filling all
    other slots with `off_value` from the 2-element `values` tensor.

    Example:
    ```mlir
    hip.one_hot(%ctx)
        ins(%indices, %depth, %values :
            memref<2x2xi64, 1>, memref<i64, 1>, memref<2xf32, 1>)
        outs(%output : memref<2x2x10xf32, 1>) {axis = -1 : i64}
    ```
  }];
  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$indices,
    Hip_TensorOrMemRef:$depth,
    Hip_TensorOrMemRef:$values,
    Hip_TensorOrMemRef:$output,
    DefaultValuedAttr<I64Attr, "-1">:$axis
  );
  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $indices `,` $depth `,` $values `:` type($indices) `,` type($depth) `,` type($values) `)`
    `outs` `(` $output `:` type($output) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_CompressOp : Hip_DpsOp_WithInfer<"compress"> {
  let summary = "Select slices or elements where a 1-D condition is true";
  let description = [{
    Implements ONNX Compress: along `axis`, keep only slices whose
    corresponding `condition` entry is true, in condition order. When
    `flatten` is true the input is treated as a 1-D tensor before selection
    (no `axis` attribute on the ONNX op).

    The selected extent is data-dependent (the number of true entries), and
    the `outs` destination defines it: the kernel writes exactly that many
    slices, densely, starting at the destination's first slice. The condition
    LENGTH is only an upper bound, so the ONNX->HIP converter sizes a dynamic
    destination extent from the real count -- scanned on device and read back
    with `hip.readback_dim` before the destination is allocated -- rather than
    from the condition length. Reification therefore reports the destination's
    extent, which is also what sizes `hip.alloc_output` for a graph output.

    Example:
    ```mlir
    hip.compress(%ctx)
        ins(%input, %condition : memref<3x2xf32, 1>, memref<3xi1, 1>)
        outs(%output : memref<2x2xf32, 1>) {axis = 0 : i64, flatten = false}
    ```
  }];
  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$input,
    Hip_TensorOrMemRef:$condition,
    Hip_TensorOrMemRef:$output,
    DefaultValuedAttr<I64Attr, "0">:$axis,
    DefaultValuedAttr<BoolAttr, "false">:$flatten
  );
  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $input `,` $condition `:` type($input) `,` type($condition) `)`
    `outs` `(` $output `:` type($output) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_ScatterElementsOp : Hip_DpsOp_AutoReifyInfer<"scatter_elements"> {
  let summary = "Scatter updates along an axis using same-rank indices";
  let description = [{
    Implements ONNX ScatterElements: produce an `output` tensor with the
    same shape as `data` by copying `data`, then writing `updates` at
    positions specified by `indices` along `axis`. Supports the same
    `reduction` modes as ScatterND (`none`, `add`, `mul`, `min`, `max`).

    Example:
    ```mlir
    hip.scatter_elements(%ctx)
        ins(%data, %indices, %updates :
            memref<2x2xf32, 1>, memref<2x2xi32, 1>, memref<2x2xf32, 1>)
        outs(%output : memref<2x2xf32, 1>) {axis = 1 : i64, reduction = "none"}
    ```
  }];
  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$data,
    Hip_TensorOrMemRef:$indices,
    Hip_TensorOrMemRef:$updates,
    Hip_TensorOrMemRef:$output,
    DefaultValuedAttr<I64Attr, "0">:$axis,
    DefaultValuedAttr<StrAttr, "\"none\"">:$reduction
  );
  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $data `,` $indices `,` $updates `:` type($data) `,` type($indices) `,` type($updates) `)`
    `outs` `(` $output `:` type($output) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_GatherElementsOp : Hip_DpsOp_WithInfer<"gather_elements"> {
  let summary = "Gather elements along an axis using same-rank indices";
  let description = [{
    Gathers values from the data tensor at positions specified by the
    indices tensor along the given axis. Output shape matches indices shape.
    Implements the ONNX GatherElements operator.

    Example:
    ```mlir
    hip.gather_elements(%ctx)
        ins(%data, %indices : memref<2x2xf32, 1>, memref<2x2xi32, 1>)
        outs(%output : memref<2x2xf32, 1>) {axis = 1 : i64}
    ```
  }];
  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$data,
    Hip_TensorOrMemRef:$indices,
    Hip_TensorOrMemRef:$output,
    DefaultValuedAttr<I64Attr, "0">:$axis
  );
  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $data `,` $indices `:` type($data) `,` type($indices) `)`
    `outs` `(` $output `:` type($output) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_TopKOp : Hip_DpsOp<"top_k"> {
  let summary = "Retrieve top-K values and indices along an axis";
  let description = [{
    Implements the ONNX TopK operator: along `axis`, select the top `K`
    elements from the input and return their values and indices.

    Inputs:
      x - data tensor
      k - scalar or 1-element int64 tensor giving K

    DPS outputs (always 2):
      values - same element type as x
      indices - int64

    Example:
    ```mlir
    hip.top_k(%ctx)
        ins(%x, %k : memref<3x4xf32, 1>, memref<i64, 1>)
        outs(%values, %indices : memref<3x2xf32, 1>, memref<3x2xi64, 1>)
        {axis = 1 : i64, largest = true, sorted = true}
    ```
  }];
  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$x,
    Hip_TensorOrMemRef:$k,
    Hip_TensorOrMemRef:$values,
    Hip_TensorOrMemRef:$indices,
    DefaultValuedAttr<I64Attr, "-1">:$axis,
    DefaultValuedAttr<BoolAttr, "true">:$largest,
    DefaultValuedAttr<BoolAttr, "true">:$sorted
  );
  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $x `,` $k `:` type($x) `,` type($k) `)`
    `outs` `(` $values `,` $indices `:` type($values) `,` type($indices) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_RangeOp : Hip_DpsOp_WithInfer<"range"> {
  let summary = "Generate 1-D range tensor";
  let description = [{
    Implements ONNX Range semantics:
      output[i] = start + i * delta, limit is exclusive.

    start/limit/delta are scalar tensors (rank-0), output is rank-1.
    Uses destination-passing style: output buffer is provided as argument.
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$start,
    Hip_TensorOrMemRef:$limit,
    Hip_TensorOrMemRef:$delta,
    Hip_TensorOrMemRef:$output
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $start `,` $limit `,` $delta `:`
    type($start) `,` type($limit) `,` type($delta) `)`
    `outs` `(` $output `:` type($output) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_SiluOp : Hip_DpsOp_AutoReifyInfer<"silu"> {
  let summary = "SiLU activation: output = input * sigmoid(input)";
  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$input,
    Hip_TensorOrMemRef:$output
  );
  // result_tensors inherited from Hip_DpsOp
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def Hip_SigmoidOp : Hip_DpsOp_AutoReifyInfer<"sigmoid", "Y"> {
  let summary = "Sigmoid activation";
  let description = [{
    Applies sigmoid activation: y = 1 / (1 + exp(-x))

    Lowered via the unified unary elementwise path (wrap_sigmoid).

    Uses destination-passing style: output buffer is provided as argument.

    Example:
    ```mlir
    hip.sigmoid(%ctx) ins(%x : memref<1x128x512xf32, 1>)
                      outs(%y : memref<1x128x512xf32, 1>)
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$x,
    Hip_TensorOrMemRef:$y
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $x `:` type($x) `)`
    `outs` `(` $y `:` type($y) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_TanhOp : Hip_DpsOp_AutoReifyInfer<"tanh", "Y"> {
  let summary = "Tanh activation";
  let description = [{
    Applies hyperbolic-tangent activation: y = tanh(x)

    Lowered via the unified unary elementwise path (wrap_tanh).

    Uses destination-passing style: output buffer is provided as argument.

    Example:
    ```mlir
    hip.tanh(%ctx) ins(%x : memref<1x128x512xf32, 1>)
                   outs(%y : memref<1x128x512xf32, 1>)
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$x,
    Hip_TensorOrMemRef:$y
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $x `:` type($x) `)`
    `outs` `(` $y `:` type($y) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_SoftplusOp : Hip_DpsOp_AutoReifyInfer<"softplus", "Y"> {
  let summary = "Softplus activation: y = ln(exp(x) + 1)";

  let description = [{
    Applies the softplus activation function element-wise.
    Implements the standard ONNX Softplus operator.

    Formula: Y = ln(exp(X) + 1)

    Uses destination-passing style: output buffer is provided as argument.

    Example:
    ```mlir
    hip.softplus(%ctx) ins(%x : memref<1x128x512xf32, 1>)
                       outs(%y : memref<1x128x512xf32, 1>)
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$x,
    Hip_TensorOrMemRef:$y
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $x `:` type($x) `)`
    `outs` `(` $y `:` type($y) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_GeluOp : Hip_DpsOp_AutoReifyInfer<"gelu"> {
  let summary = "GELU activation (Gaussian Error Linear Unit)";
  let description = [{
    Applies the GELU activation function element-wise.
    Implements the ONNX Gelu operator with support for both exact and approximate modes.

    Approximate mode (approximate="tanh"):
      Y ≈ 0.5 * X * (1 + tanh(sqrt(2/π) * (X + 0.044715 * X³)))

    Exact mode (approximate="none", default):
      Y = X * 0.5 * (1.0 + erf(X / sqrt(2.0)))

    Uses destination-passing style: output buffer is provided as argument.

    Example:
    ```mlir
    hip.gelu(%ctx) ins(%x : memref<1x128x768xf16, 1>)
                   outs(%y : memref<1x128x768xf16, 1>)
                   {approximate = "tanh"}
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$input,
    Hip_TensorOrMemRef:$output,
    DefaultValuedAttr<StrAttr, "\"none\"">:$approximate
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $input `:` type($input) `)`
    `outs` `(` $output `:` type($output) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_BiasGeluOp : Hip_DpsOp_AutoReifyInfer<"bias_gelu"> {
  let summary = "Fused bias-add + GELU (erf) activation";
  let description = [{
    Computes Gelu(data + bias) where bias is a 1D tensor broadcast along the
    last dimension of data. Implements com.microsoft.BiasGelu with the exact
    (erf-based) Gelu formula.

    Example:
    ```mlir
    hip.bias_gelu(%ctx) ins(%data, %bias : tensor<1x128x768xf16>, tensor<768xf16>)
                        outs(%out : tensor<1x128x768xf16>)
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$data,
    Hip_TensorOrMemRef:$bias,
    Hip_TensorOrMemRef:$output
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $data `,` $bias `:` type($data) `,` type($bias) `)`
    `outs` `(` $output `:` type($output) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_FastGeluOp : Hip_DpsOp_AutoReifyInfer<"fast_gelu"> {
  let summary = "Fast GELU (tanh approx) with optional 1D bias broadcast";
  let description = [{
    Computes FastGelu(x [+ broadcast(bias)]) using the tanh approximation.
    Implements com.microsoft.FastGelu. When bias is omitted, applies FastGelu
    to the input directly.

    Example:
    ```mlir
    hip.fast_gelu(%ctx) ins(%x : tensor<1x128x768xf16>)
                        outs(%y : tensor<1x128x768xf16>)
    hip.fast_gelu(%ctx) ins(%x : tensor<1x128x768xf16>)
                        bias(%b : tensor<768xf16>)
                        outs(%y : tensor<1x128x768xf16>)
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$input,
    Optional<Hip_TensorOrMemRef>:$bias,
    Hip_TensorOrMemRef:$output
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $input `:` type($input) `)`
    (`bias` `(` $bias^ `:` type($bias) `)`)?
    `outs` `(` $output `:` type($output) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_LeakyReluOp : Hip_DpsOp_AutoReify<"leaky_relu"> {
  let summary = "LeakyRelu activation: y = x >= 0 ? x : alpha * x";
  let description = [{
    Applies LeakyRelu activation element-wise.
    Implements the ONNX LeakyRelu operator.

    Formula: Y = X if X >= 0, else alpha * X

    Uses destination-passing style: output buffer is provided as argument.

    Example:
    ```mlir
    hip.leaky_relu(%ctx) ins(%x : memref<3x4xf32, 1>)
                         outs(%y : memref<3x4xf32, 1>)
                         {alpha = 0.01 : f64}
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$input,
    Hip_TensorOrMemRef:$output,
    DefaultValuedAttr<F64Attr, "0.01">:$alpha
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $input `:` type($input) `)`
    `outs` `(` $output `:` type($output) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_ReciprocalOp : Hip_DpsOp_AutoReifyInfer<"reciprocal", "Y"> {
  let summary = "Element-wise reciprocal: y = 1/x";
  let description = [{
    Computes the reciprocal of each element.
    Implements the standard ONNX Reciprocal operator.

    Formula: Y = 1 / X

    Uses destination-passing style: output buffer is provided as argument.

    Example:
    ```mlir
    hip.reciprocal(%ctx) ins(%x : memref<1x128x512xf32, 1>)
                         outs(%y : memref<1x128x512xf32, 1>)
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$x,
    Hip_TensorOrMemRef:$y
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $x `:` type($x) `)`
    `outs` `(` $y `:` type($y) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_SqrtOp : Hip_DpsOp_AutoReifyInfer<"sqrt", "Y"> {
  let summary = "Element-wise square root: y = sqrt(x)";
  let description = [{
    Computes the square root of each element.
    Formula: Y = sqrt(X) = X^0.5

    Negative inputs return NaN (per ONNX Sqrt specification).

    Uses destination-passing style: output buffer is provided as argument.

    Example:
    ```mlir
    hip.sqrt(%ctx) ins(%x : memref<1x128x512xf32, 1>)
                   outs(%y : memref<1x128x512xf32, 1>)
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$x,
    Hip_TensorOrMemRef:$y
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $x `:` type($x) `)`
    `outs` `(` $y `:` type($y) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_DivOp : Hip_DpsOp_Broadcast<"div", /*operandGetters=*/["Lhs", "Rhs"]> {
  let summary = "Elementwise division";
  let description = [{
    Performs elementwise division: output = lhs / rhs

    Supports multidirectional (Numpy-style) broadcasting.

    Uses destination-passing style: output buffer is provided as argument.

    Example:
    ```mlir
    hip.div(%ctx) ins(%lhs, %rhs : memref<3x4xf32, 1>,
                                   memref<3x4xf32, 1>)
                  outs(%output : memref<3x4xf32, 1>)
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$lhs,
    Hip_TensorOrMemRef:$rhs,
    Hip_TensorOrMemRef:$output
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $lhs `,` $rhs `:` type($lhs) `,` type($rhs) `)`
    `outs` `(` $output `:` type($output) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_EqualOp : Hip_DpsOp_Broadcast<"equal", /*operandGetters=*/["Lhs", "Rhs"]> {
  let summary = "Element-wise equality comparison: c = (a == b)";
  let description = [{
    Compares two input tensors element-wise for equality.
    Output is a boolean tensor (i1) with the same broadcast shape.
    Implements the standard ONNX Equal operator.

    Supports multidirectional (Numpy-style) broadcasting.

    Uses destination-passing style: output buffer is provided as argument.

    Example:
    ```mlir
    hip.equal(%ctx) ins(%a, %b : memref<3x4xf32, 1>, memref<3x4xf32, 1>)
                    outs(%c : memref<3x4xi1, 1>)
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$lhs,
    Hip_TensorOrMemRef:$rhs,
    Hip_TensorOrMemRef:$output
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $lhs `,` $rhs `:` type($lhs) `,` type($rhs) `)`
    `outs` `(` $output `:` type($output) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_AndOp : Hip_DpsOp_Broadcast<"and", /*operandGetters=*/["Lhs", "Rhs"]> {
  let summary = "Element-wise logical AND: c = a && b";
  let description = [{
    Computes element-wise logical AND of two boolean tensors (i1).
    Implements the standard ONNX And operator.

    Supports multidirectional (NumPy-style) broadcasting between `lhs`
    and `rhs`. Both inputs and the output have boolean (i1) element type.

    Uses destination-passing style: output buffer is provided as argument.

    Example:
    ```mlir
    hip.and(%ctx) ins(%a, %b : memref<3x4xi1, 1>, memref<3x4xi1, 1>)
                  outs(%c : memref<3x4xi1, 1>)
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$lhs,
    Hip_TensorOrMemRef:$rhs,
    Hip_TensorOrMemRef:$output
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $lhs `,` $rhs `:` type($lhs) `,` type($rhs) `)`
    `outs` `(` $output `:` type($output) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_OrOp : Hip_DpsOp_Broadcast<"or", /*operandGetters=*/["Lhs", "Rhs"]> {
  let summary = "Element-wise logical OR: c = a || b";
  let description = [{
    Computes element-wise logical OR of two boolean tensors (i1).
    Implements the standard ONNX Or operator.

    Supports multidirectional (NumPy-style) broadcasting between `lhs`
    and `rhs`. Both inputs and the output have boolean (i1) element type.

    Uses destination-passing style: output buffer is provided as argument.

    Example:
    ```mlir
    hip.or(%ctx) ins(%a, %b : memref<3x4xi1, 1>, memref<3x4xi1, 1>)
                 outs(%c : memref<3x4xi1, 1>)
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$lhs,
    Hip_TensorOrMemRef:$rhs,
    Hip_TensorOrMemRef:$output
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $lhs `,` $rhs `:` type($lhs) `,` type($rhs) `)`
    `outs` `(` $output `:` type($output) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_NotOp : Hip_DpsOp_AutoReifyInfer<"not", "Y"> {
  let summary = "Element-wise logical NOT: y = !x";
  let description = [{
    Computes the logical NOT of each element (boolean/i1).
    Implements the standard ONNX Not operator.

    Formula: Y = !X

    Uses destination-passing style: output buffer is provided as argument.

    Example:
    ```mlir
    hip.not(%ctx) ins(%x : memref<3x4xi1, 1>)
                  outs(%y : memref<3x4xi1, 1>)
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$x,
    Hip_TensorOrMemRef:$y
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $x `:` type($x) `)`
    `outs` `(` $y `:` type($y) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_CosOp : Hip_DpsOp_AutoReifyInfer<"cos", "Y"> {
  let summary = "Element-wise cosine: y = cos(x)";
  let description = [{
    Computes the cosine of each element.
    Implements the standard ONNX Cos operator.

    Formula: Y = cos(X)

    Uses destination-passing style: output buffer is provided as argument.

    Example:
    ```mlir
    hip.cos(%ctx) ins(%x : memref<1x128x512xf32, 1>)
                  outs(%y : memref<1x128x512xf32, 1>)
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$x,
    Hip_TensorOrMemRef:$y
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $x `:` type($x) `)`
    `outs` `(` $y `:` type($y) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_ErfOp : Hip_DpsOp_AutoReifyInfer<"erf", "Y"> {
  let summary = "Element-wise error function: y = erf(x)";
  let description = [{
    Computes the Gauss error function of each element.
    Implements the standard ONNX Erf operator.

    Formula: Y = erf(X)

    Uses destination-passing style: output buffer is provided as argument.

    Example:
    ```mlir
    hip.erf(%ctx) ins(%x : memref<1x128x200x200xf32, 1>)
                  outs(%y : memref<1x128x200x200xf32, 1>)
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$x,
    Hip_TensorOrMemRef:$y
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $x `:` type($x) `)`
    `outs` `(` $y `:` type($y) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_SinOp : Hip_DpsOp_AutoReifyInfer<"sin", "Y"> {
  let summary = "Element-wise sine: y = sin(x)";
  let description = [{
    Computes the sine of each element.
    Implements the standard ONNX Sin operator.

    Formula: Y = sin(X)

    Uses destination-passing style: output buffer is provided as argument.

    Example:
    ```mlir
    hip.sin(%ctx) ins(%x : memref<1x128x512xf32, 1>)
                  outs(%y : memref<1x128x512xf32, 1>)
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$x,
    Hip_TensorOrMemRef:$y
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $x `:` type($x) `)`
    `outs` `(` $y `:` type($y) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_CeilOp : Hip_DpsOp_AutoReifyInfer<"ceil", "Y"> {
  let summary = "Element-wise ceiling: y = ceil(x)";
  let description = [{
    Rounds each element upward to the nearest integer.
    Implements the standard ONNX Ceil operator.

    Formula: Y = ceil(X)

    Lowered via the unified unary elementwise path (wrap_ceil).

    Uses destination-passing style: output buffer is provided as argument.

    Example:
    ```mlir
    hip.ceil(%ctx) ins(%x : memref<3x4xf32, 1>)
                   outs(%y : memref<3x4xf32, 1>)
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$x,
    Hip_TensorOrMemRef:$y
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $x `:` type($x) `)`
    `outs` `(` $y `:` type($y) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_RoundOp : Hip_DpsOp_AutoReifyInfer<"round", "Y"> {
  let summary = "Element-wise round-ties-to-even: y = round(x)";
  let description = [{
    Rounds each element to the nearest integer; halfway cases go to the
    nearest even integer (ONNX Round).

    Formula: Y = round_ties_to_even(X)

    Lowered via the unified unary elementwise path (wrap_round).

    Uses destination-passing style: output buffer is provided as argument.

    Example:
    ```mlir
    hip.round(%ctx) ins(%x : memref<3x4xf32, 1>)
                    outs(%y : memref<3x4xf32, 1>)
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$x,
    Hip_TensorOrMemRef:$y
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $x `:` type($x) `)`
    `outs` `(` $y `:` type($y) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_AtanOp : Hip_DpsOp_AutoReifyInfer<"atan", "Y"> {
  let summary = "Element-wise arctangent: y = atan(x)";
  let description = [{
    Computes the arctangent of each element.
    Implements the standard ONNX Atan operator.

    Formula: Y = atan(X)

    Lowered via the unified unary elementwise path (wrap_atan).

    Uses destination-passing style: output buffer is provided as argument.

    Example:
    ```mlir
    hip.atan(%ctx) ins(%x : memref<3x4xf32, 1>)
                   outs(%y : memref<3x4xf32, 1>)
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$x,
    Hip_TensorOrMemRef:$y
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $x `:` type($x) `)`
    `outs` `(` $y `:` type($y) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_FloorOp : Hip_DpsOp_AutoReifyInfer<"floor", "Y"> {
  let summary = "Element-wise floor: y = floor(x)";
  let description = [{
    Rounds each element downward to the nearest integer.
    Implements the standard ONNX Floor operator.

    Formula: Y = floor(X)

    Lowered via the unified unary elementwise path (wrap_floor).

    Uses destination-passing style: output buffer is provided as argument.

    Example:
    ```mlir
    hip.floor(%ctx) ins(%x : memref<3x4xf32, 1>)
                    outs(%y : memref<3x4xf32, 1>)
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$x,
    Hip_TensorOrMemRef:$y
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $x `:` type($x) `)`
    `outs` `(` $y `:` type($y) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_ExpOp : Hip_DpsOp_AutoReify<"exp"> {
  let summary = "Element-wise natural exponential: y = exp(x)";
  let description = [{
    Computes the natural exponential of each element.
    Implements the standard ONNX Exp operator.

    Formula: Y = exp(X)

    Lowered via the unified unary elementwise path (wrap_exp).

    Uses destination-passing style: output buffer is provided as argument.

    Example:
    ```mlir
    hip.exp(%ctx) ins(%x : memref<1x128x512xf32, 1>)
                  outs(%y : memref<1x128x512xf32, 1>)
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$x,
    Hip_TensorOrMemRef:$y
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $x `:` type($x) `)`
    `outs` `(` $y `:` type($y) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_LogOp : Hip_DpsOp_AutoReify<"log"> {
  let summary = "Element-wise natural logarithm: y = log(x)";
  let description = [{
    Computes the natural logarithm of each element.
    Implements the standard ONNX Log operator.

    Formula: Y = log(X)

    Lowered via the unified unary elementwise path (wrap_log).

    Uses destination-passing style: output buffer is provided as argument.

    Example:
    ```mlir
    hip.log(%ctx) ins(%x : memref<1x128x512xf32, 1>)
                  outs(%y : memref<1x128x512xf32, 1>)
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$x,
    Hip_TensorOrMemRef:$y
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $x `:` type($x) `)`
    `outs` `(` $y `:` type($y) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_AbsOp : Hip_DpsOp_AutoReifyInfer<"abs", "Y"> {
  let summary = "Element-wise absolute value: y = abs(x)";
  let description = [{
    Computes the absolute value of each element.
    Implements the standard ONNX Abs operator.

    Formula: Y = abs(X)

    Lowered via the unified unary elementwise path (wrap_abs).

    Uses destination-passing style: output buffer is provided as argument.

    Example:
    ```mlir
    hip.abs(%ctx) ins(%x : memref<3x4xf32, 1>)
                  outs(%y : memref<3x4xf32, 1>)
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$x,
    Hip_TensorOrMemRef:$y
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $x `:` type($x) `)`
    `outs` `(` $y `:` type($y) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_NegOp : Hip_DpsOp_AutoReifyInfer<"neg", "Y"> {
  let summary = "Element-wise negation: y = -x";
  let description = [{
    Negates each element of the input tensor.
    Implements the standard ONNX Neg operator.

    Formula: Y = -X

    Lowered via the unified unary elementwise path (wrap_neg).

    Uses destination-passing style: output buffer is provided as argument.

    Example:
    ```mlir
    hip.neg(%ctx) ins(%x : memref<3x4xf32, 1>)
                  outs(%y : memref<3x4xf32, 1>)
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$x,
    Hip_TensorOrMemRef:$y
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $x `:` type($x) `)`
    `outs` `(` $y `:` type($y) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_SubOp : Hip_DpsOp_Broadcast<"sub", /*operandGetters=*/["Lhs", "Rhs"]> {
  let summary = "Elementwise subtraction";
  let description = [{
    Performs elementwise subtraction: output = lhs - rhs

    Both inputs must have the same shape. Broadcasting is not currently supported
    and must be handled before conversion to HIP dialect.

    Uses destination-passing style: output buffer is provided as argument.

    Example:
    ```mlir
    hip.sub(%ctx) ins(%lhs, %rhs : memref<1x1xi64, 1>, memref<1x1xi64, 1>)
                  outs(%output : memref<1x1xi64, 1>)
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$lhs,
    Hip_TensorOrMemRef:$rhs,
    Hip_TensorOrMemRef:$output
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $lhs `,` $rhs `:` type($lhs) `,` type($rhs) `)`
    `outs` `(` $output `:` type($output) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_WhereOp : Hip_DpsOp_Broadcast<"where",
                                      /*operandGetters=*/["Condition", "X", "Y"]> {
  let summary = "Elementwise selection: output = condition ? x : y";
  let description = [{
    Implements ONNX Where: output[i] = condition[i] ? x[i] : y[i].

    Supports multidirectional (NumPy-style) broadcasting between condition,
    x and y. Each operand's shape is broadcast to the output shape; dims of
    1 are broadcast against larger dims of the other operands.

    The condition tensor must have a boolean (i1) element type. The x and y
    tensors must share the same element type, which is also the output type.

    Uses destination-passing style: output buffer is provided as argument.

    Example:
    ```mlir
    hip.where(%ctx) ins(%cond, %x, %y :
                        memref<1x4xi1, 1>,
                        memref<2x1xf32, 1>,
                        memref<2x4xf32, 1>)
                    outs(%out : memref<2x4xf32, 1>)
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$condition,
    Hip_TensorOrMemRef:$x,
    Hip_TensorOrMemRef:$y,
    Hip_TensorOrMemRef:$output
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $condition `,` $x `,` $y `:`
        type($condition) `,` type($x) `,` type($y) `)`
    `outs` `(` $output `:` type($output) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_CastOp : Hip_DpsOp_AutoReifyInfer<"cast"> {
  let summary = "Type conversion operation";
  let description = [{
    Converts input tensor from one data type to another.

    Input and output may have different element types but the same shape.

    Uses destination-passing style: output buffer is provided as argument.

    Example:
    ```mlir
    hip.cast(%ctx) ins(%input : memref<128x256xf32, 1>)
                   outs(%output : memref<128x256xf16, 1>) {to = 10 : i64}
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$input,
    Hip_TensorOrMemRef:$output,
    I64Attr:$to
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $input `:` type($input) `)`
    `outs` `(` $output `:` type($output) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_ReduceSumOp : Hip_DpsOp_Reduction<"reduce_sum"> {
  let summary = "Reduce tensor by summing along axes";
  let description = [{
    Computes the sum of input tensor elements along the specified axes.
    Implements the ONNX ReduceSum operator (opset 13+).

    Uses destination-passing style: output buffer is provided as argument.

    Conversion materializes normalized constant axes: absent or empty axes
    become the all-axis span when noop_with_empty_axes=0 and remain empty when
    noop_with_empty_axes=1.

    Example:
    ```mlir
    hip.reduce_sum(%ctx) ins(%data, %axes : memref<1x128xi64, 1>, memref<i64, 1>)
                         outs(%output : memref<1x1xi64, 1>) {keepdims = 1 : i64, noop_with_empty_axes = 0 : i64}
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$data,
    Hip_TensorOrMemRef:$axes,
    Hip_TensorOrMemRef:$output,
    DefaultValuedAttr<I64Attr, "1">:$keepdims,
    DefaultValuedAttr<I64Attr, "0">:$noop_with_empty_axes,
    DenseI64ArrayAttr:$normalized_axes
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $data `,` $axes `:` type($data) `,` type($axes) `)`
    `outs` `(` $output `:` type($output) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_ReduceMeanOp : Hip_DpsOp_Reduction<"reduce_mean"> {
  let summary = "Reduce tensor by averaging along axes";
  let description = [{
    Computes the mean of input tensor elements along the specified axes.
    Implements the ONNX ReduceMean operator (opset 18+).

    Uses destination-passing style: output buffer is provided as argument.

    The division by the reduced-element count is performed inside the runtime
    kernel (`reduce_size = num_input / num_output`), so extents in a
    compile-time-resolved contiguous reduction span may be dynamic.
    Result dims are refined post-conversion by `--hip-infer-shapes` through the
    upstream MLIR `ReifyRankedShapedTypeOpInterface` / `InferTypeOpInterface`
    that `Hip_DpsOp_Reduction` implements (no ONNX-level shape refinement
    needed).

    Conversion materializes normalized constant axes: absent or empty axes
    become the all-axis span when noop_with_empty_axes=0 and remain empty when
    noop_with_empty_axes=1.

    Example:
    ```mlir
    hip.reduce_mean(%ctx) ins(%data, %axes : memref<1x128xf16, 1>, memref<i64, 1>)
                          outs(%output : memref<1x1xf16, 1>) {keepdims = 1 : i64, noop_with_empty_axes = 0 : i64}
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$data,
    Hip_TensorOrMemRef:$axes,
    Hip_TensorOrMemRef:$output,
    DefaultValuedAttr<I64Attr, "1">:$keepdims,
    DefaultValuedAttr<I64Attr, "0">:$noop_with_empty_axes,
    DenseI64ArrayAttr:$normalized_axes
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $data `,` $axes `:` type($data) `,` type($axes) `)`
    `outs` `(` $output `:` type($output) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_ReduceL2Op : Hip_DpsOp_Reduction<"reduce_l2"> {
  let summary = "Reduce tensor by L2 norm along axes";
  let description = [{
    Computes the L2 norm (Euclidean norm) of input tensor elements along the
    specified axes: output = sqrt(sum(x^2)). Implements the ONNX ReduceL2
    operator (opset 18+).

    Uses destination-passing style: output buffer is provided as argument.
    The sum-of-squares and sqrt are performed inside the runtime kernel
    (`reduce_size = num_input / num_output`), so extents in a
    compile-time-resolved contiguous reduction span may be dynamic.

    Conversion materializes normalized constant axes: absent or empty axes
    become the all-axis span when noop_with_empty_axes=0 and remain empty when
    noop_with_empty_axes=1.

    Example:
    ```mlir
    hip.reduce_l2(%ctx) ins(%data, %axes : memref<128x3x256x32xf16, 1>, memref<1xi64, 1>)
                         outs(%output : memref<128x3x256x1xf16, 1>) {keepdims = 1 : i64, noop_with_empty_axes = 0 : i64}
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$data,
    Hip_TensorOrMemRef:$axes,
    Hip_TensorOrMemRef:$output,
    DefaultValuedAttr<I64Attr, "1">:$keepdims,
    DefaultValuedAttr<I64Attr, "0">:$noop_with_empty_axes,
    DenseI64ArrayAttr:$normalized_axes
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $data `,` $axes `:` type($data) `,` type($axes) `)`
    `outs` `(` $output `:` type($output) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_ReduceMaxOp : Hip_DpsOp_Reduction<"reduce_max"> {
  let summary = "Reduce tensor by taking maximum along axes";
  let description = [{
    Computes the max of input tensor elements along the specified axes.
    Implements the ONNX ReduceMax operator (opset 18+).

    Uses destination-passing style: output buffer is provided as argument.

    Conversion materializes normalized constant axes: absent or empty axes
    become the all-axis span when noop_with_empty_axes=0 and remain empty when
    noop_with_empty_axes=1.

    Example:
    ```mlir
    hip.reduce_max(%ctx) ins(%data, %axes : memref<1x128xi64, 1>, memref<i64, 1>)
                         outs(%output : memref<1x1xi64, 1>) {keepdims = 1 : i64, noop_with_empty_axes = 0 : i64}
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$data,
    Hip_TensorOrMemRef:$axes,
    Hip_TensorOrMemRef:$output,
    DefaultValuedAttr<I64Attr, "1">:$keepdims,
    DefaultValuedAttr<I64Attr, "0">:$noop_with_empty_axes,
    DenseI64ArrayAttr:$normalized_axes
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $data `,` $axes `:` type($data) `,` type($axes) `)`
    `outs` `(` $output `:` type($output) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_ReduceMinOp : Hip_DpsOp_Reduction<"reduce_min"> {
  let summary = "Reduce tensor by taking minimum along axes";
  let description = [{
    Computes the min of input tensor elements along the specified axes.
    Implements the ONNX ReduceMin operator (opset 18+).

    Uses destination-passing style: output buffer is provided as argument.

    Conversion materializes normalized constant axes: absent or empty axes
    become the all-axis span when noop_with_empty_axes=0 and remain empty when
    noop_with_empty_axes=1.

    Example:
    ```mlir
    hip.reduce_min(%ctx) ins(%data, %axes : memref<1x128xi64, 1>, memref<i64, 1>)
                         outs(%output : memref<1x1xi64, 1>) {keepdims = 1 : i64, noop_with_empty_axes = 0 : i64}
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$data,
    Hip_TensorOrMemRef:$axes,
    Hip_TensorOrMemRef:$output,
    DefaultValuedAttr<I64Attr, "1">:$keepdims,
    DefaultValuedAttr<I64Attr, "0">:$noop_with_empty_axes,
    DenseI64ArrayAttr:$normalized_axes
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $data `,` $axes `:` type($data) `,` type($axes) `)`
    `outs` `(` $output `:` type($output) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

// ===== Causal Conv with State ================================================

def Hip_CausalConvWithStateOp :
    Hip_DpsOp<"causal_conv_with_state",
              [AttrSizedOperandSegments, OpStateOpInterface]> {
  let summary = "Stateful causal depthwise convolution";
  let description = [{
    Causal depthwise convolution with carry state for incremental decoding.

    Used by Gated DeltaNet and Mamba-style state-space architectures as a
    preprocessing step. Replaces the 3-op pattern (Concat + Conv + Slice)
    with a single fused operation.

    The convolution is causal (looks only at current and past positions along
    the last spatial dimension) and depthwise (each channel is convolved
    independently with its own kernel).

    Input layout is channels-first, (batch_size, channels, ...), unless
    `channels_last` is set, in which case 1D input is (batch_size, seq_len,
    channels). Weight layout: (channels, 1, k_1, ...) for depthwise
    convolution. The carry state stores the last (k-1) positions along the
    causal axis for incremental decode, and is (batch, channels, k-1) in both
    layouts -- only input and output are permuted, so a layout change never
    invalidates a carried state.

    `channels_last` exists because the surrounding graph is channels-last: a
    channels-first convolution has to be bracketed by a Transpose pair, and at
    the Qwen3.6 footprint that pair costs more than the convolution it serves.
    The canonicalizer sets the attribute while absorbing both Transposes.
    Only ndim=1 is supported channels-last.

    Example (1D):
    ```mlir
    hip.causal_conv_with_state(%ctx)
        ins(%input, %weight, %bias, %past_state :
            tensor<1x64x128xf16>, tensor<64x1x4xf16>,
            tensor<64xf16>, tensor<1x64x3xf16>)
        outs(%output, %present_state :
             tensor<1x64x128xf16>, tensor<1x64x3xf16>)
        {activation = "silu", ndim = 1 : i64}
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,

    // Required inputs
    Hip_TensorOrMemRef:$input,           // (batch, channels, ...)
    Hip_TensorOrMemRef:$weight,          // (channels, 1, k_1, ...)

    // Optional inputs
    Optional<Hip_TensorOrMemRef>:$bias,        // (channels)
    Optional<Hip_TensorOrMemRef>:$past_state,  // (batch, channels, k-1)

    // DPS output buffers
    Hip_TensorOrMemRef:$output,          // same shape as input
    Hip_TensorOrMemRef:$present_state,   // (batch, channels, k-1)

    // Attributes
    DefaultValuedAttr<StrAttr, "\"none\"">:$activation,  // "none", "silu", "swish"
    DefaultValuedAttr<I64Attr, "1">:$ndim,               // 1, 2, or 3
    DefaultValuedAttr<BoolAttr, "false">:$channels_last  // input/output (B, L, C)
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(`
        $input `,` $weight
        (`,` $bias^)?
        (`,` $past_state^)?
        `:`
        type($input) `,` type($weight)
        (`,` type($bias)^)?
        (`,` type($past_state)^)?
    `)`
    `outs` `(`
        $output `,` $present_state
        `:`
        type($output) `,` type($present_state)
    `)`
    attr-dict (`:` type($result_tensors)^)?
  }];

  // Absorbs a surrounding [0,2,1] Transpose pair into `channels_last`.
  // See lib/Dialect/IR/HipCausalConvCanonicalize.cpp.
  let hasCanonicalizer = 1;

  // OpStateOpInterface: construct this instance's per-session MIOpen
  // descriptor/algo cache (CausalConvState). No compile-time params — the
  // cache fills lazily per shape. The construct fn stores the state into
  // op_states[slot] itself.
  let extraClassDeclaration = [{
    ::mlir::Value generateOpStateInit(::mlir::OpBuilder &builder,
                                      ::mlir::Location loc,
                                      ::mlir::Value statePtr, int32_t slot) {
      return ::mlir::hip::emitOpStateConstruct(
          builder, loc, statePtr, slot,
          "hipdnn_ep_op_state_construct_causal_conv_with_state", {});
    }
  }];
  let hasVerifier = 1;
}

// ===== Group Query Attention =================================================

def Hip_GqaOp : Hip_DpsOp_AutoReify<"gqa", [AttrSizedOperandSegments, OpStateOpInterface]> {
  let summary = "Group Query Attention - Full MS spec implementation";
  let description = [{
    Complete implementation of com.microsoft.GroupQueryAttention (v1) specification.

    Supported features:
    - Grouped-query/Multi-query attention (GQA/MQA)
    - KV cache (BNSH format, supports past/present buffer sharing)
    - RoPE (rotary position embedding, supports cos/sin cache + position_ids)
    - Attention bias (ALiBi, T5 relative position bias, etc.)
    - Local / sliding-window attention
    - KV cache quantization (float8/int8/int4, per-tensor/per-channel)
    - Smooth softmax (numerical stability optimization)
    - Debug output (QK matrix visualization)

    Cache format: BNSH (batch_size, num_heads, seq_len, head_size)

    MS spec reference:
    https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#commicrosoftgroupqueryattention

    Example:
    ```mlir
    hip.gqa(%ctx)
        ins(%query, %key, %value, %past_key, %past_value,
            %seqlens_k, %total_seq_len :
            tensor<1x128x4096xf16>, ...)
        outs(%output, %present_key, %present_value :
             tensor<1x128x4096xf16>, ...)
        {num_heads = 32, kv_num_heads = 8, scale = 0.0884}
    ```
  }];

  let arguments = (ins
    // Internal context (HIP-specific, not ONNX spec)
    Hip_ContextType:$ctx,

    // === MS GQA Standard Inputs (spec order 1-14) ===

    // Input 1: query (required)
    Hip_TensorOrMemRef:$query,

    // Inputs 2-3: key/value (optional - supports packed QKV)
    Optional<Hip_TensorOrMemRef>:$key,
    Optional<Hip_TensorOrMemRef>:$value,

    // Inputs 4-5: past KV cache (optional - first inference has no past)
    Optional<Hip_TensorOrMemRef>:$past_key,
    Optional<Hip_TensorOrMemRef>:$past_value,

    // Inputs 6-7: sequence length info (required)
    Hip_TensorOrMemRef:$seqlens_k,        // 1D (batch_size)
    Hip_TensorOrMemRef:$total_seq_len,    // Scalar

    // Inputs 8-10: RoPE related (optional)
    Optional<Hip_TensorOrMemRef>:$cos_cache,      // 2D (max_seq_len, head_size/2)
    Optional<Hip_TensorOrMemRef>:$sin_cache,      // 2D (max_seq_len, head_size/2)
    Optional<Hip_TensorOrMemRef>:$position_ids,   // 2D (batch, seq_len), int64

    // Input 11: attention bias (optional - ALiBi etc.)
    Optional<Hip_TensorOrMemRef>:$attention_bias, // 4D (batch/1, heads/1, seq_q, seq_kv)

    // Input 12: smooth softmax factor (optional)
    Optional<Hip_TensorOrMemRef>:$head_sink,      // 1D (num_heads)

    // Inputs 13-14: KV cache quantization scale (optional)
    Optional<Hip_TensorOrMemRef>:$k_scale,        // Broadcastable to present_key shape
    Optional<Hip_TensorOrMemRef>:$v_scale,        // Broadcastable to present_value shape

    // === DPS-style output buffers ===
    Hip_TensorOrMemRef:$output,        // 3D (batch, seq_len, hidden_size)
    Hip_TensorOrMemRef:$present_key,   // 4D BNSH format
    Hip_TensorOrMemRef:$present_value, // 4D BNSH format
    Optional<Hip_TensorOrMemRef>:$output_qk,  // Optional QK matrix output (debug)

    // === MS GQA Standard Attributes ===

    // Required attributes
    I64Attr:$num_heads,           // Query head count
    I64Attr:$kv_num_heads,        // KV head count (GQA core)

    // Optional attributes (with defaults)
    DefaultValuedAttr<F32Attr, "0.0">:$scale,  // 0.0 = auto-compute 1/√head_size at runtime (matches ORT sentinel)
    DefaultValuedAttr<I64Attr, "0">:$do_rotary,
    DefaultValuedAttr<I64Attr, "0">:$rotary_interleaved,
    DefaultValuedAttr<F32Attr, "0.0">:$softcap,
    DefaultValuedAttr<I64Attr, "-1">:$local_window_size,
    DefaultValuedAttr<I64Attr, "0">:$smooth_softmax,
    DefaultValuedAttr<I64Attr, "0">:$qk_output,  // 0=no output, 1=before softmax, 2=after softmax

    // Quantization related attributes
    DefaultValuedAttr<StrAttr, "\"NONE\"">:$k_quant_type,  // "NONE"/"PER_TENSOR"/"PER_CHANNEL"
    DefaultValuedAttr<StrAttr, "\"NONE\"">:$v_quant_type,
    DefaultValuedAttr<I64Attr, "8">:$kv_cache_bit_width,  // 8 or 4

    // Whisper / bidirectional-attention enabler: when true, the causal mask
    // step in the runtime is skipped. Default false preserves all existing
    // Llama / gpt-oss callers' behaviour.
    DefaultValuedAttr<BoolAttr, "false">:$no_causal
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(`
        $query
        ( `,` $key^ )?
        ( `,` $value^ )?
        ( `,` $past_key^ )?
        ( `,` $past_value^ )?
        `,` $seqlens_k `,` $total_seq_len
        ( `,` $cos_cache^ )?
        ( `,` $sin_cache^ )?
        ( `,` $position_ids^ )?
        ( `,` $attention_bias^ )?
        ( `,` $head_sink^ )?
        ( `,` $k_scale^ )?
        ( `,` $v_scale^ )?
        `:`
        type($query)
        ( `,` type($key)^ )?
        ( `,` type($value)^ )?
        ( `,` type($past_key)^ )?
        ( `,` type($past_value)^ )?
        `,` type($seqlens_k) `,` type($total_seq_len)
        ( `,` type($cos_cache)^ )?
        ( `,` type($sin_cache)^ )?
        ( `,` type($position_ids)^ )?
        ( `,` type($attention_bias)^ )?
        ( `,` type($head_sink)^ )?
        ( `,` type($k_scale)^ )?
        ( `,` type($v_scale)^ )?
    `)`
    `outs` `(`
        $output `,` $present_key `,` $present_value
        ( `,` $output_qk^ )?
        `:`
        type($output) `,` type($present_key) `,` type($present_value)
        ( `,` type($output_qk)^ )?
    `)`
    attr-dict (`:` type($result_tensors)^)?
  }];

  // OpStateOpInterface: construct this GQA instance's per-session hipBLASLt
  // GEMM descriptor/algo cache (GqaState). No compile-time params — the cache
  // fills lazily per GEMM shape. The construct fn stores the state into
  // op_states[slot] itself.
  let extraClassDeclaration = [{
    ::mlir::Value generateOpStateInit(::mlir::OpBuilder &builder,
                                      ::mlir::Location loc,
                                      ::mlir::Value statePtr, int32_t slot) {
      return ::mlir::hip::emitOpStateConstruct(
          builder, loc, statePtr, slot, "hipdnn_ep_op_state_construct_gqa", {});
    }
  }];

  let hasVerifier = 1;
}

// ===== hipDNN Graph Outlining + Execution ====================================

def Hip_YieldOp : Hip_Op<"yield", [Pure, Terminator,
    HasParent<"HipDNNGraphOutlineOp">]> {
  let summary = "Yield results from a hipDNN graph outline region";
  let description = [{
    Terminates the body of a `hip.hipdnn_graph_outline` operation, forwarding
    the computed results back to the parent op.

    Example:
    ```mlir
    hip.yield %result : tensor<1x1x8x8xf32>
    ```
  }];

  let arguments = (ins Variadic<AnyRankedTensor>:$values);

  let builders = [
    OpBuilder<(ins), [{
      build($_builder, $_state, ::mlir::ValueRange{});
    }]>
  ];

  let assemblyFormat = "($values^ `:` type($values))? attr-dict";
}

def Hip_HipDNNGraphOutlineOp : Hip_Op<"hipdnn_graph_outline", [
    SingleBlockImplicitTerminator<"YieldOp">,
    IsolatedFromAbove
]> {
  let summary = "Outlined ONNX subgraph for hipDNN graph compilation";
  let description = [{
    Wraps one or more ONNX operations that are candidates for hipDNN graph
    compilation. The region captures only tensor operands as block arguments
    and is isolated from above.

    Created by the OutlineOnnxToHipDNN pass (handle-free, CLI-testable).
    Consumed by the CompileHipDNNGraphs pass, which either:
    - Compiles the region via hipDNN and replaces this op with hip.hipdnn_graph
    - Un-outlines (inlines the region back) on compilation failure

    Example:
    ```mlir
    %0 = hip.hipdnn_graph_outline
        ins(%x, %w : tensor<1x1x8x8xf32>, tensor<1x1x3x3xf32>)
        -> tensor<1x1x8x8xf32> {
      ^bb0(%arg0: tensor<1x1x8x8xf32>, %arg1: tensor<1x1x3x3xf32>):
        %1 = "onnx.Conv"(%arg0, %arg1) {auto_pad = "NOTSET", group = 1 : si64,
             kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]}
             : (tensor<1x1x8x8xf32>, tensor<1x1x3x3xf32>) -> tensor<1x1x8x8xf32>
        hip.yield %1 : tensor<1x1x8x8xf32>
    }
    ```
  }];

  let arguments = (ins Variadic<AnyRankedTensor>:$inputs);
  let results = (outs Variadic<AnyRankedTensor>:$results);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{
    `ins` `(` $inputs `:` type($inputs) `)` `->` type($results)
    $body attr-dict
  }];
}

// ===== hipDNN Graph Execution ================================================

def Hip_HipDNNGraphOp : Hip_DpsOp_AutoReify<"hipdnn_graph", [
    AttrSizedOperandSegments
]> {
  let summary = "Execute a pre-compiled hipDNN graph";
  let description = [{
    Dispatches to a pre-compiled hipDNN graph identified by graph_id.
    The graph was compiled during the CompileHipDNNGraphs pass and stored
    in the graph registry attached to RuntimeState.

    At runtime, looks up the graph by ID and calls graph->Execute() with
    a variant_pack built from the provided tensor UIDs and GPU pointers.

    Tensor UIDs (input_uids, output_uids) are assigned during graph building
    and baked into the IR as compile-time constants.

    Example:
    ```mlir
    hip.hipdnn_graph(%ctx) graph_id(0)
        ins(%input, %weights : tensor<1x3x224x224xf32>, tensor<64x3x7x7xf32>)
        outs(%output : tensor<1x64x112x112xf32>)
        {input_uids = [0 : i64, 1 : i64], output_uids = [2 : i64]}
        : tensor<1x64x112x112xf32>
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Variadic<Hip_TensorOrMemRef>:$inputs,
    Variadic<Hip_TensorOrMemRef>:$outputs,
    I32Attr:$graph_id,
    I64ArrayAttr:$input_uids,
    I64ArrayAttr:$output_uids
  );

  let assemblyFormat = [{
    `(` $ctx `)` `graph_id` `(` $graph_id `)`
    `ins` `(` $inputs `:` type($inputs) `)`
    `outs` `(` $outputs `:` type($outputs) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

// ===== Multi-Head Attention =================================================

def Hip_MultiHeadAttentionOp : Hip_DpsOp_AutoReify<"multi_head_attention", [AttrSizedOperandSegments, OpStateOpInterface]> {
  let summary = "Multi-Head Attention (com.microsoft.MultiHeadAttention v1)";
  let description = [{
    Implements the com.microsoft.MultiHeadAttention operator (opset v1).

    Multi-Head Self/Cross Attention with optional bias from input projection.
    Supports packed QKV in `query`, packed KV in `key`, separate Q/K/V,
    optional key padding mask, optional additive attention bias, KV cache
    (past_key/past_value) with optional buffer-sharing
    (past_sequence_length), unidirectional (causal) masking, and an optional
    cache_indirection buffer for beam-search decoding.

    Inputs (1-10, MS spec order):
      1. query                 (required, T) - either:
                                  * (batch, seq_q, hidden)         standard
                                  * (batch, seq_kv, num_heads, 3, head_size)
                                                                   packed QKV
      2. key                   (optional, T) - either:
                                  * (batch, seq_kv, hidden)
                                  * (batch, seq_kv, num_heads, 2, head_size)
                                                                   packed KV
                                  * (batch, num_heads, seq_kv, head_size)
                                                                   past_key
      3. value                 (optional, T) - (batch, seq_kv, v_hidden) or
                                                past_value layout
      4. bias                  (optional, T) - 1-D (hidden + hidden + v_hidden)
      5. key_padding_mask      (optional, M) - 1-D / 2-D / 3-D mask
      6. attention_bias        (optional, T) - bias added to QxK'
      7. past_key              (optional, T) - past KV cache for key
      8. past_value            (optional, T) - past KV cache for value
      9. past_sequence_length  (optional, M) - used with buffer sharing
     10. cache_indirection     (optional, M) - beam-search cache routing

    DPS output buffers (1-4):
      output         (required, T) - (batch, seq_q, v_hidden)
      present_key    (optional, T) - present KV cache for key
      present_value  (optional, T) - present KV cache for value
      qk             (optional, QK) - debug QK matrix

    Attributes:
      num_heads          (required) - number of attention heads
      mask_filter_value  (optional, default -10000.0) - value to fill masked
                                                       positions
      scale              (optional, default 0.0) - 0.0 means
                                                  1/sqrt(head_size) at runtime
      unidirectional     (optional, default 0) - 1 = causal mask

    Example (basic self-attention with separate Q/K/V):
    ```mlir
    hip.multi_head_attention(%ctx)
        ins(%query, %key, %value :
            memref<1x128x4096xf16, 1>, memref<1x128x4096xf16, 1>,
            memref<1x128x4096xf16, 1>)
        outs(%output : memref<1x128x4096xf16, 1>)
        {num_heads = 32 : i64, scale = 0.0 : f32,
         mask_filter_value = -1.000000e+04 : f32, unidirectional = 0 : i64}
    ```
  }];

  let arguments = (ins
    // Internal context (HIP-specific, not ONNX spec)
    Hip_ContextType:$ctx,

    // === MS MultiHeadAttention spec inputs 1-10 ===

    // Input 1: query (required)
    Hip_TensorOrMemRef:$query,

    // Inputs 2-3: key/value (optional - packed QKV in query when both absent)
    Optional<Hip_TensorOrMemRef>:$key,
    Optional<Hip_TensorOrMemRef>:$value,

    // Input 4: bias (optional - input-projection bias)
    Optional<Hip_TensorOrMemRef>:$bias,

    // Input 5: key_padding_mask (optional)
    Optional<Hip_TensorOrMemRef>:$key_padding_mask,

    // Input 6: attention_bias (optional - added to Q*K')
    Optional<Hip_TensorOrMemRef>:$attention_bias,

    // Inputs 7-8: past_key / past_value (optional - KV cache)
    Optional<Hip_TensorOrMemRef>:$past_key,
    Optional<Hip_TensorOrMemRef>:$past_value,

    // Input 9: past_sequence_length (optional - used with buffer sharing)
    Optional<Hip_TensorOrMemRef>:$past_sequence_length,

    // Input 10: cache_indirection (optional - beam-search cache routing)
    Optional<Hip_TensorOrMemRef>:$cache_indirection,

    // === DPS output buffers (1-4) ===
    Hip_TensorOrMemRef:$output,
    Optional<Hip_TensorOrMemRef>:$present_key,
    Optional<Hip_TensorOrMemRef>:$present_value,
    Optional<Hip_TensorOrMemRef>:$qk,

    // === Attributes ===
    I64Attr:$num_heads,
    DefaultValuedAttr<F32Attr, "-1.000000e+04">:$mask_filter_value,
    DefaultValuedAttr<F32Attr, "0.0">:$scale,
    DefaultValuedAttr<I64Attr, "0">:$unidirectional
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(`
        $query
        ( `,` $key^ )?
        ( `,` $value^ )?
        ( `,` $bias^ )?
        ( `,` $key_padding_mask^ )?
        ( `,` $attention_bias^ )?
        ( `,` $past_key^ )?
        ( `,` $past_value^ )?
        ( `,` $past_sequence_length^ )?
        ( `,` $cache_indirection^ )?
        `:`
        type($query)
        ( `,` type($key)^ )?
        ( `,` type($value)^ )?
        ( `,` type($bias)^ )?
        ( `,` type($key_padding_mask)^ )?
        ( `,` type($attention_bias)^ )?
        ( `,` type($past_key)^ )?
        ( `,` type($past_value)^ )?
        ( `,` type($past_sequence_length)^ )?
        ( `,` type($cache_indirection)^ )?
    `)`
    `outs` `(`
        $output
        ( `,` $present_key^ )?
        ( `,` $present_value^ )?
        ( `,` $qk^ )?
        `:`
        type($output)
        ( `,` type($present_key)^ )?
        ( `,` type($present_value)^ )?
        ( `,` type($qk)^ )?
    `)`
    attr-dict (`:` type($result_tensors)^)?
  }];

  // OpStateOpInterface: construct this MHA instance's per-session hipBLASLt
  // GEMM descriptor/algo cache (MhaState). No compile-time params — the cache
  // fills lazily per GEMM shape. The construct fn stores the state into
  // op_states[slot] itself.
  let extraClassDeclaration = [{
    ::mlir::Value generateOpStateInit(::mlir::OpBuilder &builder,
                                      ::mlir::Location loc,
                                      ::mlir::Value statePtr, int32_t slot) {
      return ::mlir::hip::emitOpStateConstruct(
          builder, loc, statePtr, slot,
          "hipdnn_ep_op_state_construct_multi_head_attention", {});
    }
  }];
}

def Hip_LinearAttentionOp : Hip_DpsOp_AutoReify<"linear_attention", [AttrSizedOperandSegments]> {
  let summary = "Linear attention with recurrent state (com.microsoft.LinearAttention)";
  let description = [{
    Unified linear attention operator for autoregressive decoding (T=1) and
    prefill (T>1).

    All inputs use 3D packed format [B, T, H*D]; q_num_heads and kv_num_heads
    are always required. The op internally unpacks to 4D for computation.

    The update_rule attribute selects the recurrence type:
      "linear":       S_t = S_{t-1} + k_t (x) v_t
      "gated":        S_t = exp(g_t) * S_{t-1} + k_t (x) v_t
      "delta":        S_t = S_{t-1} + b_t * k_t (x) (v_t - S_{t-1}^T k_t)
      "gated_delta":  S_t = exp(g_t) * S_{t-1} + b_t * k_t (x) (v_t - exp(g_t) * S_{t-1}^T k_t)

    Head counts are three-way (subject to divisibility constraints):
      - H_q   = q_num_heads             (query heads, attribute)
      - H_kv  = kv_num_heads            (KV state heads, attribute)
      - n_k   = key.shape[-1] / d_k     (key heads, derived at runtime)
    Constraints:
      - n_k divides H_kv  (multiple KV heads may share one K head when n_k<H_kv)
      - Either H_q % H_kv == 0  (standard GQA, H_q >= H_kv)
        or     H_kv % H_q == 0  (inverse GQA,   H_q <  H_kv)

    Output head count H_o = max(H_q, H_kv):
      - Standard GQA (H_q >= H_kv): output packed in Q-head order.
      - Inverse GQA  (H_q <  H_kv): output packed in KV-head order, i.e. each
        KV state produces its own output slot.

    Inputs (3-6):
      1. query       (required, T) - [B, T, H_q * d_k]
      2. key         (required, T) - [B, T, n_k * d_k], n_k | H_kv
      3. value       (required, T) - [B, T, H_kv * d_v]
      4. past_state  (optional, S) - [B, H_kv, d_k, d_v]
      5. decay       (optional, T) - [B, T, H_kv * d_k] or [B, T, H_kv]
      6. beta        (optional, T) - [B, T, H_kv] or [B, T, 1]

    DPS output buffers:
      output        (required, T) - [B, T, max(H_q, H_kv) * d_v]
      present_state (required, S) - [B, H_kv, d_k, d_v]

    Example:
    ```mlir
    hip.linear_attention(%ctx)
        ins(%query, %key, %value :
            memref<1x1x4096xf16, 1>, memref<1x1x1024xf16, 1>,
            memref<1x1x1024xf16, 1>)
        past_state(%past : memref<1x8x128x128xf16, 1>)
        decay(%g : memref<1x1x1024xf16, 1>)
        beta(%b : memref<1x1x8xf16, 1>)
        outs(%output, %state : memref<1x1x4096xf16, 1>,
                                memref<1x8x128x128xf16, 1>)
        {q_num_heads = 32 : i64, kv_num_heads = 8 : i64,
         scale = 0.0 : f32, update_rule = "gated_delta"}
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,

    // Required inputs (spec order 1-3)
    Hip_TensorOrMemRef:$query,
    Hip_TensorOrMemRef:$key,
    Hip_TensorOrMemRef:$value,

    // Optional inputs (spec order 4-6)
    Optional<Hip_TensorOrMemRef>:$past_state,
    Optional<Hip_TensorOrMemRef>:$decay,
    Optional<Hip_TensorOrMemRef>:$beta,

    // DPS output buffers
    Hip_TensorOrMemRef:$output,
    Hip_TensorOrMemRef:$present_state,

    // Required attributes
    I64Attr:$q_num_heads,
    I64Attr:$kv_num_heads,

    // Optional attributes (with defaults)
    DefaultValuedAttr<F32Attr, "0.0">:$scale,
    DefaultValuedAttr<I64Attr, "0">:$chunk_size,
    DefaultValuedAttr<StrAttr, "\"gated_delta\"">:$update_rule
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(`
        $query `,` $key `,` $value `:`
        type($query) `,` type($key) `,` type($value)
    `)`
    (`past_state` `(` $past_state^ `:` type($past_state) `)`)?
    (`decay` `(` $decay^ `:` type($decay) `)`)?
    (`beta` `(` $beta^ `:` type($beta) `)`)?
    `outs` `(` $output `,` $present_state `:`
              type($output) `,` type($present_state) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_GemmOp :
    Hip_DpsOp_Semantic<"gemm", /*traits=*/[OpStateOpInterface],
                       /*outsAccessor=*/"Output"> {
  let summary = "General matrix multiplication (ONNX Gemm)";
  let description = [{
    Computes Y = alpha * A' * B' + beta * C, where:
      A' = transpose(A) if transA else A
      B' = transpose(B) if transB else B
    Input C is optional and broadcastable to (M, N).
    Implements ONNX Gemm operator (opset 13).
    Uses destination-passing style: output buffer is provided as argument.
    Example:
    ```mlir
    hip.gemm(%ctx) ins(%A, %B, %C :
        memref<128x256xf32, 1>, memref<256x512xf32, 1>, memref<512xf32, 1>)
        outs(%Y : memref<128x512xf32, 1>)
        {alpha = 1.0 : f32, beta = 1.0 : f32, transA = 0 : i64, transB = 0 : i64}
    ```
  }];
  let arguments = (ins
    Hip_ContextType:$ctx,

    // inputs
    Hip_TensorOrMemRef:$input_a,
    Hip_TensorOrMemRef:$input_b,
    Optional<Hip_TensorOrMemRef>:$input_c,  // optional

    //outputs
    Hip_TensorOrMemRef:$output,

    // attributes
    DefaultValuedAttr<F32Attr, "1.0">:$alpha,
    DefaultValuedAttr<F32Attr, "1.0">:$beta,
    DefaultValuedAttr<I64Attr, "0">:$transA,
    DefaultValuedAttr<I64Attr, "0">:$transB
  );
  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(`
        $input_a `,` $input_b (`,` $input_c^)?
        `:` type($input_a) `,` type($input_b) (`,` type($input_c)^)?
    `)`
    `outs` `(` $output `:` type($output) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];

  // OpStateOpInterface: construct this instance's per-session GemmState, which
  // holds a shared_ptr to the device-wide hipBLASLt algo table shared across
  // sessions via WeakStore.
  let extraClassDeclaration = [{
    ::mlir::Value generateOpStateInit(::mlir::OpBuilder &builder,
                                      ::mlir::Location loc,
                                      ::mlir::Value statePtr, int32_t slot) {
      return ::mlir::hip::emitOpStateConstruct(
          builder, loc, statePtr, slot, "hipdnn_ep_op_state_construct_gemm",
          {});
    }
  }];

  // Verifies the DPS contract plus the shared `inferGemmShape` rule, so the
  // `outs` shape, the converter destination, and `reifyResultShapes` are all
  // held to one ONNX Gemm shape function.
  let hasVerifier = 1;
}

def Hip_CumSumOp : Hip_DpsOp_AutoReifyInfer<"cumsum", "Y"> {
  let summary = "Cumulative sum along an axis";
  let description = [{
    Computes the cumulative sum of input tensor elements along the given axis.
    Implements the standard ONNX CumSum operator (opset 14+).

    The `axis` input is a 0-D scalar tensor selecting the reduction axis. The
    `exclusive` attribute (0 / 1) toggles inclusive vs exclusive accumulation,
    and `reverse` (0 / 1) reverses the accumulation direction along the axis.

    Uses destination-passing style: output buffer is provided as argument.

    Example:
    ```mlir
    hip.cumsum(%ctx) ins(%x, %axis : memref<3x4xf32, 1>, memref<i64, 1>)
                     outs(%y : memref<3x4xf32, 1>)
                     {exclusive = 0 : i64, reverse = 0 : i64}
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$x,
    Hip_TensorOrMemRef:$axis,
    Hip_TensorOrMemRef:$y,
    DefaultValuedAttr<I64Attr, "0">:$exclusive,
    DefaultValuedAttr<I64Attr, "0">:$reverse
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $x `,` $axis `:` type($x) `,` type($axis) `)`
    `outs` `(` $y `:` type($y) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_PadOp : Hip_DpsOp_WithInfer<"pad", "Output", [AttrSizedOperandSegments]> {
  let summary = "Pad a tensor with constant / reflect / edge / wrap";
  let description = [{
    Pads input tensor along the axes selected by `axes` (or all axes if
    `axes` is empty) according to the `pads` tensor.
    Implements the standard ONNX Pad operator (opset 25+).

    `pads` is a 1-D int64 tensor of shape `[2 * num_axes]` formatted as
    `[x1_begin, ..., xN_begin, x1_end, ..., xN_end]`. `constant_value` is
    an optional scalar used in `constant` mode (default 0). `axes` is an
    optional 1-D int64 tensor selecting which axes the pads apply to;
    when omitted (rank-0 / size-0), all axes are padded.

    The `mode` attribute is one of `"constant"`, `"reflect"`, `"edge"`,
    `"wrap"`.

    Uses destination-passing style: output buffer is provided as argument.

    Example:
    ```mlir
    hip.pad(%ctx) ins(%data, %pads : memref<3x2xf32, 1>, memref<4xi64, 1>)
                  cval(%cval : memref<f32, 1>)
                  axes(%axes : memref<2xi64, 1>)
                  outs(%out : memref<3x4xf32, 1>) {mode = "constant"}
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$data,
    Hip_TensorOrMemRef:$pads,
    Optional<Hip_TensorOrMemRef>:$constant_value,
    Optional<Hip_TensorOrMemRef>:$axes,
    Hip_TensorOrMemRef:$output,
    DefaultValuedAttr<StrAttr, "\"constant\"">:$mode
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $data `,` $pads `:` type($data) `,` type($pads) `)`
    (`cval` `(` $constant_value^ `:` type($constant_value) `)`)?
    (`axes` `(` $axes^ `:` type($axes) `)`)?
    `outs` `(` $output `:` type($output) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_TileOp : Hip_DpsOp_WithInfer<"tile"> {
  let summary = "Tile a tensor by repeating along each dimension";
  let description = [{
    Constructs a new tensor by repeating `input` along each dimension by the
    counts specified in `repeats`. Implements the standard ONNX Tile operator.

    `repeats` is a 1-D int64 tensor with the same length as `input`'s rank;
    `output_dim[i] = input_dim[i] * repeats[i]`.

    Uses destination-passing style: output buffer is provided as argument.

    Example:
    ```mlir
    hip.tile(%ctx) ins(%x, %repeats : memref<2x2xf32, 1>, memref<2xi64, 1>)
                   outs(%y : memref<2x4xf32, 1>)
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$input,
    Hip_TensorOrMemRef:$repeats,
    Hip_TensorOrMemRef:$output
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $input `,` $repeats `:` type($input) `,` type($repeats) `)`
    `outs` `(` $output `:` type($output) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_ExpandOp : Hip_DpsOp_WithInfer<"expand"> {
  let summary = "Broadcast a tensor to a target shape";
  let description = [{
    Broadcasts `input` to the target `shape` following NumPy broadcast rules.
    Implements the standard ONNX Expand operator.

    `shape` is a 1-D int64 tensor describing the target shape (which is
    broadcast against the input shape, so output rank may exceed `shape`'s
    length).

    Uses destination-passing style: output buffer is provided as argument.

    Example:
    ```mlir
    hip.expand(%ctx) ins(%x, %shape : memref<3x1xf32, 1>, memref<3xi64, 1>)
                     outs(%y : memref<2x3x6xf32, 1>)
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$input,
    Hip_TensorOrMemRef:$shape,
    Hip_TensorOrMemRef:$output
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $input `,` $shape `:` type($input) `,` type($shape) `)`
    `outs` `(` $output `:` type($output) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_GlobalPoolOp : Hip_DpsOp_AutoReify<"global_pool"> {
  let summary = "Global pooling over spatial dimensions (average / max / lp)";
  let description = [{
    Unified lowering for ONNX GlobalAveragePool / GlobalMaxPool / GlobalLpPool.
    For an input of shape `(N, C, D_1, D_2, ..., D_k)`, produces an output of
    shape `(N, C, 1, 1, ..., 1)` (same rank as input). Each output element is
    a per-`(n, c)` reduction across all spatial elements; `mode` selects the
    reduction:

      * mode=0 (AVERAGE): Y = mean(X[n, c, *...])
      * mode=1 (MAX)    : Y = max(X[n, c, *...])
      * mode=2 (LP)     : Y = pow(sum(pow(|X[n, c, *...]|, p)), 1/p)
                          (LP norm; `p` defaults to 2)

    Both input and output rank must be >= 3 (i.e. at least one spatial dim);
    the dimensions at positions 0 (N) and 1 (C) are passthrough, and every
    spatial dimension in the output is 1.

    The `p` attribute is only meaningful when `mode == 2` (LP); for AVERAGE
    and MAX it is ignored by the runtime.

    Uses destination-passing style: the output buffer is provided.

    Example:
    ```mlir
    hip.global_pool(%ctx) ins(%x : memref<1x3x5x5xf32, 1>)
                          outs(%y : memref<1x3x1x1xf32, 1>)
                          {mode = 0 : i64, p = 2 : i64}
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$input,
    Hip_TensorOrMemRef:$output,
    I64Attr:$mode,
    DefaultValuedAttr<I64Attr, "2">:$p
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $input `:` type($input) `)`
    `outs` `(` $output `:` type($output) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_ReduceProdOp : Hip_DpsOp_Reduction<"reduce_prod"> {
  let summary = "Reduce tensor by multiplying elements along axes";
  let description = [{
    Computes the product of input tensor elements along the specified axes.
    Implements the ONNX ReduceProd operator (opset 18+).

    Uses destination-passing style: output buffer is provided as argument.

    Conversion materializes normalized constant axes: absent or empty axes
    become the all-axis span when noop_with_empty_axes=0 and remain empty when
    noop_with_empty_axes=1.

    Example:
    ```mlir
    hip.reduce_prod(%ctx) ins(%data, %axes : memref<3x2x2xf32, 1>, memref<0xi64, 1>)
                          outs(%output : memref<1x1x1xf32, 1>) {keepdims = 1 : i64, noop_with_empty_axes = 0 : i64}
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$data,
    Hip_TensorOrMemRef:$axes,
    Hip_TensorOrMemRef:$output,
    DefaultValuedAttr<I64Attr, "1">:$keepdims,
    DefaultValuedAttr<I64Attr, "0">:$noop_with_empty_axes,
    DenseI64ArrayAttr:$normalized_axes
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $data `,` $axes `:` type($data) `,` type($axes) `)`
    `outs` `(` $output `:` type($output) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_LessOp : Hip_DpsOp_Broadcast<"less", /*operandGetters=*/["Lhs", "Rhs"]> {
  let summary = "Element-wise less-than comparison: c = (a < b)";
  let description = [{
    Compares two input tensors element-wise. Output is a boolean tensor (i1)
    with the broadcast shape. Implements the standard ONNX Less operator.

    Supports multidirectional (NumPy-style) broadcasting.

    Uses destination-passing style: output buffer is provided as argument.

    Example:
    ```mlir
    hip.less(%ctx) ins(%a, %b : memref<3x4xf32, 1>, memref<3x4xf32, 1>)
                   outs(%c : memref<3x4xi1, 1>)
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$lhs,
    Hip_TensorOrMemRef:$rhs,
    Hip_TensorOrMemRef:$output
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $lhs `,` $rhs `:` type($lhs) `,` type($rhs) `)`
    `outs` `(` $output `:` type($output) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_SliceOp : Hip_DpsOp_WithInfer<"slice", "Output", [AttrSizedOperandSegments]> {
  let summary = "Slice a tensor along multiple axes (ONNX Slice fallback)";
  let description = [{
    Native fallback for ONNX Slice when the OnnxToHip pass cannot decompose
    the op to `tensor.extract_slice` (e.g. non-constant slice parameters or
    negative `steps`). Implements the standard ONNX Slice operator semantics:

      output[k] = data[ start[k0]:end[k0]:step[k0],
                         start[k1]:end[k1]:step[k1], ... ]

    `starts` / `ends` are 1-D integer tensors (int32 or int64), one entry per
    axis listed in `axes`. `axes` defaults to `[0, ..., r-1]` when absent and
    `steps` defaults to all-ones. Per-axis negative indices and negative
    steps follow ONNX clamping rules.

    The common case (compile-time constant indices, positive unit stride) is
    handled by an upstream decompose pattern that lowers `onnx.Slice` to
    `tensor.extract_slice` — that path produces a zero-cost view without any
    runtime call. This op exists only to give the fallback path a stable
    DPS-shaped representation to bufferize and lower from.

    Uses destination-passing style: output buffer is provided as argument.

    Example:
    ```mlir
    hip.slice(%ctx) ins(%data, %starts, %ends : memref<3x4xf32, 1>,
                                                  memref<2xi64, 1>,
                                                  memref<2xi64, 1>)
                    axes(%axes : memref<2xi64, 1>)
                    steps(%steps : memref<2xi64, 1>)
                    outs(%out : memref<3x2xf32, 1>)
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$data,
    Hip_TensorOrMemRef:$starts,
    Hip_TensorOrMemRef:$ends,
    Optional<Hip_TensorOrMemRef>:$axes,
    Optional<Hip_TensorOrMemRef>:$steps,
    Hip_TensorOrMemRef:$output
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $data `,` $starts `,` $ends `:`
        type($data) `,` type($starts) `,` type($ends) `)`
    (`axes` `(` $axes^ `:` type($axes) `)`)?
    (`steps` `(` $steps^ `:` type($steps) `)`)?
    `outs` `(` $output `:` type($output) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_GatherNDOp : Hip_DpsOp_WithInfer<"gather_nd"> {
  let summary = "GatherND: gather slices using N-D index tuples";
  let description = [{
    Gathers slices of `data` indexed by the trailing dimension of `indices`.
    Implements the standard ONNX GatherND operator.

    `indices` is an integer tensor whose last dimension defines the rank of
    each index tuple, optionally with `batch_dims` leading batch dims that
    must match `data`. Output rank is
    `q + r - indices_shape[-1] - 1 - batch_dims`, where `q = rank(indices)`
    and `r = rank(data)`.

    Uses destination-passing style: output buffer is provided as argument.

    Example:
    ```mlir
    hip.gather_nd(%ctx) ins(%data, %indices : memref<2x2xf32, 1>, memref<2x2xi64, 1>)
                        outs(%out : memref<2xf32, 1>) {batch_dims = 0 : i64}
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$data,
    Hip_TensorOrMemRef:$indices,
    Hip_TensorOrMemRef:$output,
    DefaultValuedAttr<I64Attr, "0">:$batch_dims
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $data `,` $indices `:` type($data) `,` type($indices) `)`
    `outs` `(` $output `:` type($output) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_ScatterNDOp : Hip_DpsOp_AutoReifyInfer<"scatter_nd"> {
  let summary = "ScatterND: scatter updates into a data tensor using N-D index tuples";
  let description = [{
    Implements the standard ONNX ScatterND operator: produce an `output`
    tensor with the same shape as `data` whose values are `data` copied,
    then `updates` written back at positions specified by `indices`.

    `indices` is an integer tensor whose last dimension `k` (with
    `k <= rank(data)`) defines the rank of each index tuple. When
    `k == rank(data)` each update entry is a scalar; otherwise each entry
    is a contiguous slice of trailing shape `data.shape[k:]`.

    `reduction` selects how an existing value at the target position is
    combined with the new update value:

      * `"none"` (default) — overwrite (`a = b`)
      * `"add"` — accumulate (`a = a + b`)
      * `"mul"` — multiply (`a = a * b`)
      * `"min"` — minimum (`a = min(a, b)`)
      * `"max"` — maximum (`a = max(a, b)`)

    This is the inverse of GatherND.

    Uses destination-passing style: the `output` buffer is provided as an
    argument, and the runtime is responsible for both the initial
    `data → output` copy and the subsequent scatter writes.

    Example:
    ```mlir
    hip.scatter_nd(%ctx) ins(%data, %indices, %updates :
                              memref<4x4x4xf32, 1>,
                              memref<2x1xi64, 1>,
                              memref<2x4x4xf32, 1>)
                          outs(%out : memref<4x4x4xf32, 1>)
                          {reduction = "none"}
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$data,
    Hip_TensorOrMemRef:$indices,
    Hip_TensorOrMemRef:$updates,
    Hip_TensorOrMemRef:$output,
    DefaultValuedStrAttr<StrAttr, "none">:$reduction
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $data `,` $indices `,` $updates `:`
        type($data) `,` type($indices) `,` type($updates) `)`
    `outs` `(` $output `:` type($output) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_SignOp : Hip_DpsOp_AutoReifyInfer<"sign", "Y"> {
  let summary = "Element-wise sign: y = sign(x)";
  let description = [{
    Computes the element-wise sign of the input. Returns +1 / 0 / -1 for
    positive / zero / negative elements respectively. Implements the standard
    ONNX Sign operator.

    Lowered via the unified unary elementwise path (wrap_sign).

    Uses destination-passing style: output buffer is provided as argument.

    Example:
    ```mlir
    hip.sign(%ctx) ins(%x : memref<3x4xf32, 1>)
                   outs(%y : memref<3x4xf32, 1>)
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$x,
    Hip_TensorOrMemRef:$y
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $x `:` type($x) `)`
    `outs` `(` $y `:` type($y) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_ModOp : Hip_DpsOp_Broadcast<"mod", /*operandGetters=*/["Lhs", "Rhs"]> {
  let summary = "Element-wise modulo / fmod operation";
  let description = [{
    Performs element-wise binary modulo. Implements the standard ONNX Mod
    operator. The semantics depend on the `fmod` attribute:

    * `fmod = 0` (default): integer modulo; sign follows divisor (Python `%`).
    * `fmod = 1`: floating-point fmod; sign follows dividend (C `fmod`).

    Supports multidirectional (NumPy-style) broadcasting.

    Uses destination-passing style: output buffer is provided as argument.

    Example:
    ```mlir
    hip.mod(%ctx) ins(%a, %b : memref<3x4xi32, 1>, memref<3x4xi32, 1>)
                  outs(%c : memref<3x4xi32, 1>) {fmod = 0 : i64}
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$lhs,
    Hip_TensorOrMemRef:$rhs,
    Hip_TensorOrMemRef:$output,
    DefaultValuedAttr<I64Attr, "0">:$fmod
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $lhs `,` $rhs `:` type($lhs) `,` type($rhs) `)`
    `outs` `(` $output `:` type($output) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_SizeOp : Hip_DpsOp_AutoReifyInfer<"size", "Y"> {
  let summary = "Total element count of a tensor (ONNX Size, dynamic-shape path)";
  let description = [{
    Implements the runtime path of the standard ONNX Size operator
    (opset 1+) for inputs whose shape is not fully static.

    Statically-shaped Size ops are folded to a single `arith.constant` at
    OnnxToHip time (see SizeConversion.cpp::SizeToConstant), so this op
    is only emitted when at least one input dimension is `?`. The
    HipToLLVM lowering computes `prod(input.shape)` as a runtime i64
    value (using compile-time constants for static dims and descriptor
    `sizes[]` for dynamic dims) and stores it into the rank-0 i64
    output buffer.

    Input  x: ranked tensor / memref of any element type, any rank,
              with at least one dynamic dim.
    Output y: rank-0 i64 tensor / memref (`tensor<i64>` / `memref<i64, 1>`).

    Uses destination-passing style: the output buffer is provided as
    argument.

    Example:
    ```mlir
    hip.size(%ctx) ins(%x : memref<?x?xf16, 1>)
                    outs(%y : memref<i64, 1>)
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$x,
    Hip_TensorOrMemRef:$y
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $x `:` type($x) `)`
    `outs` `(` $y `:` type($y) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_NonZeroOp : Hip_DpsOp_AutoReify<"nonzero"> {
  let summary = "Indices of non-zero elements (ONNX NonZero)";
  let description = [{
    Implements the standard ONNX NonZero operator (opset 13+):
    returns the indices of the elements that are non-zero in row-major
    order (by dimension), behaving similarly to `numpy.nonzero`.

    Input `x` is a ranked tensor of shape `[D0, D1, ..., D(R-1)]`. Output
    `y` has shape `[R, N]` of element type `i64`, where `R` is the input
    rank and `N` is the (data-dependent) number of non-zero elements.
    The `N` dimension is represented as `?` in MLIR; the compiler
    over-allocates `y` using `numel(x)` as the upper-bound *capacity*
    (`N` columns beyond the true count are left undefined).

    The true count `N` is data-dependent and only known once the kernel
    has scanned the input, so the op also takes a second destination
    `count` (a device `i32` scalar buffer) that the kernel writes with
    the actual number of non-zero elements. Downstream code reads it back
    to the host via `hip.readback_dim` and slices `y` to `[R, N]` so the
    real extent (not the capacity) flows to consumers and to the
    ORT-reported output shape. Two destinations means the op is
    multi-result (`y`, `count`); like the other multi-output DPS ops it does
    not use the single-result inference family (converters pass explicit types).

    Uses destination-passing style: both output buffers are provided as
    arguments.

    Example:
    ```mlir
    hip.nonzero(%ctx) ins(%x : memref<3x4xf16, 1>)
                      outs(%y, %count : memref<2x?xi64, 1>, memref<i32, 1>)
                      {input_data_type = 1 : i64}
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$x,
    Hip_TensorOrMemRef:$y,
    Hip_TensorOrMemRef:$count,
    I64Attr:$input_data_type
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $x `:` type($x) `)`
    `outs` `(` $y `,` $count `:` type($y) `,` type($count) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_PoolOp : Hip_DpsOp_AutoReify<"pool"> {
  let summary = "Spatial pooling (ONNX MaxPool / AveragePool / LpPool)";
  let description = [{
    Unified window pooling for ONNX MaxPool / AveragePool / LpPool. For an
    input of shape `(N, C, D_1, D_2, ..., D_k)` (rank = 2 + spatial_rank,
    with spatial_rank in {1, 2, 3}) reduces each sliding window described by
    `kernel_shape`, `strides`, `pads`, and `dilations` into a single output
    element. Output shape is `(N, C, O_1, ..., O_k)`.

    `pool_mode` selects the per-window reduction (values mirror
    HIPDNN_EP_POOL_* in lib/Runtime/hipdnn_ep_runtime.h):

      * pool_mode=0 (AVERAGE): Y = mean over the window
      * pool_mode=1 (MAX)    : Y = max over the window
      * pool_mode=2 (LP)     : Y = pow(sum(pow(|x|, p)), 1/p)

    Pad positions are never read (they fall outside the input bounds). For
    AVERAGE, `count_include_pad` selects the divisor: 0 = number of in-bounds
    window elements, 1 = full kernel volume (pad cells contribute 0 to the
    sum). For LP, `p` is the norm exponent (>= 1). For MAX, both attributes
    are ignored.

    Only MAX may produce a second output `Indices` of element type i64
    holding the flattened input index of each selected element (per the ONNX
    spec, indices are a flat 1-D tensor of size `N * C * D_1 * ... * D_k`,
    ignoring padding). AVERAGE / LP always have exactly one output.

    Attributes:
    - pool_mode:    0 = AVERAGE, 1 = MAX, 2 = LP
    - kernel_shape: spatial kernel dims (length = spatial_rank)
    - strides:      spatial strides (length = spatial_rank)
    - pads:         [x1_begin, ..., xk_begin, x1_end, ..., xk_end]
    - dilations:    spatial dilations (length = spatial_rank)
    - ceil_mode:    0 = floor (default), 1 = ceil  (already absorbed into
                    output shape at compile time; passed through for
                    runtime debugging)
    - storage_order: 0 = row-major (default). 1 (column-major) is not
                    supported and rejected at conversion time. MAX only.
    - count_include_pad: AVERAGE divisor selector (default 0). AVERAGE only.
    - p:            LP norm exponent (default 2). LP only.

    Uses destination-passing style: the output buffer (and optional indices
    buffer) are provided as Variadic `outputs`.

    Example (2D MaxPool, with indices):
    ```mlir
    hip.pool(%ctx) ins(%x : memref<1x3x32x32xf16, 1>)
                   outs(%y, %idx : memref<1x3x16x16xf16, 1>,
                                   memref<1x3x16x16xi64, 1>)
                   {pool_mode = 1, kernel_shape = [2, 2], strides = [2, 2],
                    pads = [0, 0, 0, 0], dilations = [1, 1],
                    ceil_mode = 0, storage_order = 0,
                    count_include_pad = 0, p = 2}
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$input,
    Variadic<Hip_TensorOrMemRef>:$outputs,
    I64Attr:$pool_mode,
    I64ArrayAttr:$kernel_shape,
    I64ArrayAttr:$strides,
    I64ArrayAttr:$pads,
    I64ArrayAttr:$dilations,
    I64Attr:$ceil_mode,
    I64Attr:$storage_order,
    DefaultValuedAttr<I64Attr, "0">:$count_include_pad,
    DefaultValuedAttr<I64Attr, "2">:$p
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $input `:` type($input) `)`
    `outs` `(` $outputs `:` type($outputs) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_ResizeOp : Hip_DpsOp_AutoReify<"resize"> {
  let summary = "Spatial resize / interpolation (ONNX Resize)";
  let description = [{
    Implements a subset of ONNX Resize that covers the common image / volume
    upsample-and-downsample patterns.  Resamples the trailing spatial axes of
    an `(N, C, D_1, ..., D_k)` input onto an `(N, C, O_1, ..., O_k)` output
    grid.  The leading two dims (N, C) must be unchanged at runtime — i.e.
    we only resize spatial axes.

    Supported attribute set (anything outside is rejected at conversion):
      * mode in {"nearest", "linear"}              — encoded as i64
        - 0 = nearest, 1 = linear (N-linear: bilinear / trilinear)
      * coordinate_transformation_mode in
          {"half_pixel", "asymmetric", "align_corners"}  — encoded as i64
        - 0 = half_pixel, 1 = asymmetric, 2 = align_corners
      * nearest_mode in {"round_prefer_floor"}      — encoded as i64
        - 0 = round_prefer_floor (only used when mode=nearest)

    Per-axis scale is recovered at runtime from the input/output spatial
    extents (`scale = in_dim / out_dim`), so neither `scales` nor `sizes`
    appear as operands here — they were resolved at compile time into the
    output type that the upstream importer produced.

    Uses destination-passing style: the output buffer is provided.

    Example (2D bilinear upsample with half_pixel):
    ```mlir
    hip.resize(%ctx) ins(%x : memref<1x3x16x16xf16, 1>)
                     outs(%y : memref<1x3x32x32xf16, 1>)
                     {mode = 1, coord_transform = 0, nearest_mode = 0}
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$input,
    Hip_TensorOrMemRef:$output,
    I64Attr:$mode,
    I64Attr:$coord_transform,
    I64Attr:$nearest_mode
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $input `:` type($input) `)`
    `outs` `(` $output `:` type($output) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}

def Hip_GridSampleOp : Hip_DpsOp_AutoReifyInfer<"grid_sample"> {
  let summary = "ONNX GridSample (4D NCHW bilinear / nearest)";
  let description = [{
    Samples a 4-D input `X` of shape (N, C, H_in, W_in) at normalized
    locations from `grid` of shape (N, H_out, W_out, 2).  The last grid
    axis is (x, y) in [-1, 1].  Output Y has shape (N, C, H_out, W_out).

    Attribute encoding (decoded at ONNX->HIP conversion):
      * mode: 0 = nearest, 1 = bilinear (ONNX "bilinear" / "linear")
      * padding_mode: 0 = zeros, 1 = border, 2 = reflection
      * align_corners: 0 or 1 (ONNX default 0)

    Cubic / bicubic interpolation and 5-D (volumetric) sampling are
    rejected at conversion.

    Example:
    ```mlir
    hip.grid_sample(%ctx)
        ins(%x, %grid : memref<1x3x8x8xf32, 1>, memref<1x4x4x2xf32, 1>)
        outs(%y : memref<1x3x4x4xf32, 1>)
        {mode = 1, padding_mode = 0, align_corners = 0}
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$input,
    Hip_TensorOrMemRef:$grid,
    Hip_TensorOrMemRef:$output,
    I64Attr:$mode,
    I64Attr:$padding_mode,
    I64Attr:$align_corners
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(`
        $input `,` $grid `:` type($input) `,` type($grid)
    `)`
    `outs` `(` $output `:` type($output) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
  let hasVerifier = 1;
}

// hip.readback_dim : materialize a kernel-computed runtime extent as a host
// `index`. It synchronizes the stream so the producing kernel has finished,
// then reads back a single device-resident integer `scalar`.
//
// This is the host-side half of the data-dependent dynamic-shape lowering: a
// kernel writes a runtime extent (a count, valid-length, selected-element
// tally, ...) into a small device buffer, and `readback_dim` turns it into a
// host `index` that can size a `tensor.extract_slice`, `tensor.empty`, or
// `hip.alloc_output`.
//
// It is deliberately not `Pure` and not speculatable: it performs a
// device->host copy plus a stream synchronization, and must remain below the
// producing kernel in program order. Non-speculatability is also what stops
// `hip-hoist-alloc-size-arith` from lifting an extent-sized allocation above
// that kernel into the function entry block.
//
// Before (tensor form, after convert-onnx-to-hip):
//   %n = hip.readback_dim(%ctx, %count : tensor<i32>) -> index
// After bufferize:
//   %n = hip.readback_dim(%ctx, %count : memref<i32, 1>) -> index
def Hip_ReadbackDimOp : Hip_Op<"readback_dim",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Synchronize and read a device i32 scalar into a host index";
  let description = [{
    Reads the single integer stored in the device-resident `scalar` buffer
    back to the host as an `index`, after synchronizing the GPU stream so the
    kernel that wrote `scalar` is guaranteed to have completed.

    Has device->host copy + stream-synchronization side effects; it is not
    speculative and must not be hoisted across the kernel that produces
    `scalar`.
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$scalar
  );
  let results = (outs Index:$dim);

  let assemblyFormat = [{
    `(` $ctx `,` $scalar `:` type($scalar) `)` attr-dict `->` type($dim)
  }];
}

// hip.readback_scalar : synchronize the stream and read a single
// device-resident scalar of ARBITRARY element type back to a host scalar of
// the same type. The general-purpose sibling of `hip.readback_dim` (which is
// specialised to a non-negative i32 extent and returns `index`): this op
// preserves the operand's element type (i64 / f32 / f16 / ...) and is used
// where the host needs the actual value — e.g. the limit/start/delta of a
// data-dependent `onnx.Range` whose trip count is computed host-side from
// signed integers or floats.
//
// Like `hip.readback_dim` it performs a device->host copy plus a stream
// synchronization, so it is NOT `Pure`/speculatable and must stay below the
// kernel that produced `scalar` in program order. A bare `tensor.extract` /
// `memref.load` of a GPU-written scalar is incorrect on targets where the pool
// is true device memory: the host read races / reads stale memory and yields a
// zero count, collapsing the dynamic dimension. (It accidentally works where
// the pool is UMA-mapped host-accessible memory.)
//
// Before (tensor form, after convert-onnx-to-hip):
//   %v = hip.readback_scalar(%ctx, %limit : tensor<i64>) -> i64
// After bufferize:
//   %v = hip.readback_scalar(%ctx, %limit : memref<i64, 1>) -> i64
def Hip_ReadbackScalarOp : Hip_Op<"readback_scalar",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Synchronize and read a device scalar into a host scalar of the "
                "same element type";
  let description = [{
    Reads the single value stored in the device-resident `scalar` buffer back
    to the host as a value of the SAME element type, after synchronizing the
    GPU stream so the kernel that wrote `scalar` is guaranteed to have
    completed.

    Has device->host copy + stream-synchronization side effects; it is not
    speculative and must not be hoisted across the kernel that produces
    `scalar`.
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$scalar
  );
  let results = (outs AnyTypeOf<[AnyInteger, AnyFloat, Index]>:$value);

  let assemblyFormat = [{
    `(` $ctx `,` $scalar `:` type($scalar) `)` attr-dict `->` type($value)
  }];
}
#endif // HIP_OPS
