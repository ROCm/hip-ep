/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===-- HipPasses.td - HIP pass definitions ----------------*- tablegen -*-===//
//
//===----------------------------------------------------------------------===//

#ifndef HIP_PASSES
#define HIP_PASSES

include "mlir/Pass/PassBase.td"

def HipAddContextArgPass : Pass<"hip-add-context-arg", "mlir::ModuleOp"> {
  let summary = "Insert !hip.context as arg 0 into every func.func";
  let description = [{
    Inserts a !hip.context argument at position 0 into every func.func in
    the module. The OnnxToHip conversion patterns assume that the context
    value is always available as the first function argument.

    Skips functions that already have a !hip.context as argument 0.
  }];
  let dependentDialects = [
    "mlir::hip::HipDialect",
    "mlir::func::FuncDialect"
  ];
}

def SimplifyOnnxPass : Pass<"simplify-onnx", "mlir::ModuleOp"> {
  let summary = "Pre-lowering simplification of ONNX-dialect graphs";
  let description = [{
    Runs a small fixed set of ONNX-dialect rewrites that must complete
    BEFORE convert-onnx-to-hip captures the function signatures as input
    metadata. After the per-op rewrites, drops any function argument
    whose only remaining uses were eliminated by them.

    The pass is pure ONNX-dialect: it does not depend on the HIP dialect
    and is intentionally positioned at the head of the pipeline, BEFORE
    hip-add-context-arg, so it operates in the original ONNX function
    index space and could be reused unchanged by any other frontend.

    Pipeline position:
      simplify-onnx -> hip-add-context-arg -> convert-onnx-to-hip

    Per-op rewrites currently implemented:
      * onnx.CastLike -> onnx.Cast
          CastLike's second operand is a *type donor* whose data is never
          read -- the target dtype is statically known from the result
          type. Rewriting to a plain onnx.Cast lets the shared
          CastConversion pattern in convert-onnx-to-hip do the rest, AND
          frees up any function argument that was wired up only to feed
          that type-donor slot.

    Future-friendly: additional simplifiers (e.g. onnx.EyeLike /
    onnx.RandomNormalLike, which are similar shape/type-donor patterns)
    can be added by writing one `static void simplify<Op>(func::FuncOp)`
    helper in SimplifyOnnx.cpp and calling it from the pass dispatcher,
    between the live-before-rewrite snapshot and the shared
    dead-function-argument sweep.

    Invariants:
      * Arguments that were already dead in the input IR are preserved,
        so the existing "metadata captures the original signature"
        contract asserted by test/lit/Pipeline/module-metadata.mlir
        stays intact.
  }];
  let dependentDialects = [
    "mlir::func::FuncDialect"
  ];
}

def ConvertOnnxToHipPass : Pass<"convert-onnx-to-hip", "mlir::ModuleOp"> {
  let summary = "Convert ONNX dialect to HIP dialect (tensor DPS)";
  let description = [{
    Converts ONNX dialect IR (from onnx-mlir) into HIP dialect IR using
    destination-passing style (DPS) with tensor types.

    Transformations:
      1. Constants: onnx.Constant -> arith.constant (small/inline) or
         memref.global extern + memref.get_global + bufferization.to_tensor
         (large/externalized to sidecar .constants.bin + .constants.json)
      2. Op mapping: onnx.{MatMul,Transpose,Mul,Softmax} -> hip.* (tensor DPS)
      3. Cleanup: erase onnx.EntryPoint

    Requires hip-add-context-arg to have run first so that !hip.context is
    available as function argument 0.

    Constant externalization (opt-in via --externalize-min-num-elements):
      Large constants above the threshold are written to a binary sidecar file
      and replaced with extern memref.global ops carrying a hip.external_data
      attribute (offset + size).  The filename is recorded as a
      hip.constants_file attribute on the module.  This follows the onnx-mlir
      pattern (krnl.global + omMMapBinaryFile) adapted to standard MLIR ops.
      Splat constants are always kept inline regardless of size.
  }];
  let options = [
    Option<"externalizeOutputDir", "externalize-output-dir", "std::string",
           /*default=*/"\"\"",
           "Directory for sidecar .constants.bin/.json files (empty = cwd)">,
    Option<"externalizeMinNumElements", "externalize-min-num-elements",
           "int64_t", /*default=*/"0",
           "Minimum number of tensor elements to externalize (0 = disabled)">
  ];
  let dependentDialects = [
    "mlir::hip::HipDialect",
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect",
    "mlir::tensor::TensorDialect",
    "mlir::memref::MemRefDialect",
    "mlir::bufferization::BufferizationDialect"
  ];
}

def ConvertHipToLLVMPass : Pass<"convert-hip-to-llvm", "mlir::ModuleOp"> {
  let summary = "Convert HIP dialect to LLVM dialect";
  let description = [{
    Lowers HIP dialect operations to LLVM dialect by converting each HIP op
    into calls to the HIP runtime C API (hipCreateHandle, hipblasltMatmul,
    etc.). MemRef operands are unpacked into bare pointers for the C calls.
  }];
  let dependentDialects = [
    "mlir::LLVM::LLVMDialect",
    "mlir::memref::MemRefDialect"
  ];
}

def OptimizeMemRefsPass : Pass<"hip-optimize-memrefs", "mlir::func::FuncOp"> {
  let summary = "Optimize memref buffers to reduce allocation count";
  let description = [{
    Performs liveness-based best-fit buffer reuse on straight-line bufferized
    HIP IR.  Replaces memref.alloc ops with earlier compatible buffers whose
    live ranges have ended, reducing peak memory usage.

    Compatibility rules:
      - Static shapes: any slot whose byte-size is >= the needed allocation
        (same element type and memory space).  A memref.reinterpret_cast is
        inserted when shapes differ.
      - Dynamic shapes: the MemRefType must be identical and every dynamic-size
        operand must be the same SSA value.

    Only single-block functions are optimized; multi-block functions are left
    unchanged.
  }];
  let dependentDialects = [
    "mlir::memref::MemRefDialect"
  ];
}

def PromoteStridedHipOperandsPass
    : Pass<"hip-promote-strided-operands", "mlir::func::FuncOp"> {
  let summary = "Materialize contiguous temporaries for non-canonical memref "
                "operands of hip.* ops";
  let description = [{
    Walks every DestinationStyleOpInterface op in the `hip` dialect.  For
    each input (DPS-input) operand whose memref type has a non-identity
    layout (non-zero offset or non-contiguous strides), inserts:

    ```mlir
    %tmp = memref.alloc() : memref<...identity-layout...>
    memref.copy %strided, %tmp : ..., ...
    // consumer is rewritten to read from %tmp instead of %strided
    memref.dealloc %tmp : memref<...identity-layout...>
    ```

    DPS-init (output) operands are left untouched: in this pipeline they are
    always either fresh `memref.alloc` results or function arguments under
    `IdentityLayoutMap`, so they are guaranteed contiguous by construction.

    The pass exists to bridge the contract gap between the IR (which can
    represent strided memrefs faithfully via memref.subview) and the HIP
    runtime call ABI (which receives a single bare pointer with no offset /
    strides channel).  Without this promotion, hip.* runtime calls silently
    read the base of the parent buffer instead of the intended slice.

    Recommended pipeline position:
      hip-optimize-memrefs -> hip-promote-strided-operands -> hip-pool-allocs

    This placement ensures the new transient allocations are routed through
    the HIP memory pool instead of producing extra hipMalloc calls.
  }];
  let dependentDialects = [
    "mlir::memref::MemRefDialect"
  ];
}

def PoolAllocsPass : Pass<"hip-pool-allocs", "mlir::func::FuncOp"> {
  let summary = "Pool memref.alloc ops into a grow-on-demand i8 byte buffer";
  let description = [{
    Packs all memref.alloc ops in a single-block function into one contiguous
    byte pool (memref<?xi8>) obtained via hip.get_pool.  Each original
    allocation is replaced with a memref.view at a computed byte offset.

    The pool size is computed as an SSA value and passed to hip.get_pool,
    which uses grow-on-demand semantics: the runtime grows the pool when the
    requested size exceeds current capacity.

    Static allocations use greedy best-fit offset assignment (inspired by
    greedy best-fit strip packing).  Dynamic allocations
    are bucketed by their byte-size SSA value: non-overlapping lifetimes within
    a bucket share an offset.

    Sub-buffer offsets are aligned to 256 bytes by default (configurable).

    Recommended pipeline position:
      hip-optimize-memrefs (optional) -> hip-pool-allocs -> hip-lower-allocs
  }];
  let options = [
    Option<"alignment", "alignment", "int64_t", /*default=*/"256",
           "Sub-buffer byte offset alignment (must be power of 2)">
  ];
  let dependentDialects = [
    "mlir::hip::HipDialect",
    "mlir::arith::ArithDialect",
    "mlir::memref::MemRefDialect"
  ];
}

def LowerAllocsPass : Pass<"hip-lower-allocs", "mlir::func::FuncOp"> {
  let summary = "Lower memref.alloc to hip.alloc and insert hip.free";
  let description = [{
    Replaces memref.alloc operations with hip.alloc (device allocation via
    hipMalloc) and inserts hip.free for buffers that are not returned from
    the function.  Requires !hip.context as function argument 0.

    This pass bridges one-shot-bufferize output (generic memref allocations)
    to the HIP-specific lowering that produces hipMalloc/hipFree calls.
  }];
  let dependentDialects = [
    "mlir::hip::HipDialect",
    "mlir::memref::MemRefDialect"
  ];
}

def ComposeDimSpecsPass
    : Pass<"hip-compose-dim-specs", "mlir::ModuleOp"> {
  let summary = "Compose per-op output DimSpecs into model-output DimSpecs";
  let description = [{
    For each result dimension of `@main_graph` that is dynamic (`?`), this
    pass walks the producer chain via the (free-function-based)
    `HipShapeOpInterface` (`shape_interface::resolveDimFromValue`) and
    substitutes operand references until the spec bottoms out at one of:
      * `Static` constant
      * `InputDim(input_index, dim_index)` — Category A
      * `InputValueI64(input_index, flat_offset)` — Category B
      * `RuntimeSlot(slot_id)` — Category C (publisher op found upstream)

    The resulting composed DimSpec tree is attached as a func-result
    attribute `hip.dim_spec` on `@main_graph` so `GenerateInterface` can
    pick it up at metadata-blob time without re-walking the IR. A second
    module attribute `hip.dyn_dim_slots_count` records the maximum slot id
    referenced anywhere, plus 1.

    Backward compatibility: if `@main_graph` has no dynamic result dims,
    the pass is a no-op. If a dynamic dim cannot be resolved by the
    walker, the pass emits a remark (not an error) and the result attr
    is left absent — downstream falls back to the legacy "-1 in shape"
    behaviour and the EP will treat the dim as unknown.

    Pipeline position: between `convert-hip-to-llvm` and
    `generate-interface` (today both live inside `buildHipToLLVMPipeline`).
    The pass operates on the bufferized memref IR but does not modify any
    op — it only reads, composes, and attaches attributes.
  }];
  let dependentDialects = [
    "mlir::hip::HipDialect",
    "mlir::func::FuncDialect"
  ];
}

def ReservePropagatorSlotsPass
    : Pass<"hip-reserve-propagator-slots", "mlir::ModuleOp"> {
  let summary = "Reserve runtime slot ids for translucent propagators "
                "whose results have dynamic dims, attaching the "
                "hipdnn.output_slot_ids array-form attribute";
  let description = [{
    For every `Hip_DpsOp` in `@main_graph` whose result type has at
    least one dynamic dim, this pass:

      1. Skips publishers that already carry `slot_id` / `slot_ids` /
         `hipdnn.output_slot_ids` -- those are Cat-C ops (NonZero,
         Range Cat-C, ConstantOfShape Cat-C) that reserved their slot
         ids in the converter.
      2. Skips ops whose every dynamic result dim resolves to a
         non-RuntimeSlot DimSpec (Cat-A / Cat-B / static / pure-
         arithmetic over those) -- such ops do not need a slot to
         publish; their downstream consumer can resolve the dim from
         the operand-relative DimSpec.
      3. For every remaining (result, dim) whose DimSpec contains a
         `RuntimeSlot` leaf (i.e. transitively depends on a Cat-C
         publisher), reserves a fresh slot id from the module-level
         `hipdnn.next_dyn_slot_id` counter and records it in the
         array-form attribute:

           hipdnn.output_slot_ids = [[s_00, s_01, ...], ...]   // i32[numResults][rank]

         with `-1` for entries that don't need a slot. This is what
         downstream passes (lowerings, annotation pass, Phase 3
         coalescing) read to find the propagator's slot grid without
         having to re-walk DimSpec leaves.

    NOTE: the reservation is purely metadata. The propagator's runtime
    wrapper (Phase 2.4) is what actually calls `dyn_pool_alloc_for_slot`
    + `publish_buffer` at run time. Without the wrapper-side change the
    metadata is dead. Pipeline order ensures this pass runs AFTER
    `convert-onnx-to-hip` (so DimSpec builders are present) and BEFORE
    bufferize (so SSA values still have their tensor types and the
    walker can use them directly).
  }];
  let dependentDialects = [
    "mlir::hip::HipDialect",
    "mlir::func::FuncDialect"
  ];
}

def IdentityPropagatorRebindPass
    : Pass<"hip-identity-propagator-rebind", "mlir::ModuleOp"> {
  let summary = "Erase identity-shaped Cat-C propagator ops and rewire "
                "their slot pointers / SSA uses to the upstream input";
  let description = [{
    Phase 4 of the slot-buffer-coalesce design
    (docs/design/slot-buffer-coalesce.md).

    For propagators that are runtime no-ops (e.g. `hip.transpose`
    with the identity permutation, `hip.cast` from f32 -> f32,
    `hip.slice` over the entire input, `hip.tile` with all-ones
    repeats, `hip.expand` to the same shape, `hip.reduce_*` with
    `noop_with_empty_axes=1` and an empty axes set), the kernel
    launch is wasted work AND the propagator's reserved output slot
    buffer can be aliased onto the upstream input's slot buffer (or
    onto the input directly when the input has no slot).

    Algorithm (per `@main_graph` op, top-down walk):
      1. Skip ops without a registered identity predicate
         (`shape_interface::isIdentityOp`).
      2. Evaluate the predicate. If false, skip.
      3. If true:
         a. If the op carries `hipdnn.output_slot_ids` (a Phase 2
            propagator that reserved its own slot), look up the
            input value's producer slot (via the input's producer's
            `hipdnn.output_slot_ids` / `slot_id` / `slot_ids`). When
            BOTH have a slot, record an "input slot -> output slot"
            equivalence so a follow-up renumber phase rewrites all
            consumer references uniformly to the input slot id.
         b. RAUW (replace-all-uses-with) the op's result with the op's
            input.
         c. Erase the op.
      4. After the walk, apply the equivalence map to every slot id
         reference in the module so consumer attrs / DimSpec leaves /
         module-level metadata point at the upstream slot.

    The follow-up pipeline pass (`SlotLifetimeCoalescePass`) sees
    the smaller slot set produced by this rewrite. When the input
    has NO slot (e.g. a propagator immediately downstream of a func
    argument or a static-shape value), step (a) records `-1 ->
    output_slot` which is detected and skipped -- the output slot
    survives, the propagator's kernel still gets elided, and a
    future pass can lift the input into a fresh publisher.

    Backward compat: ops without an identity predicate are
    untouched. Disabled with the `disable-identity-rebind` option
    for A/B testing of kernel-launch counts.

    Pipeline position: BEFORE `hip-annotate-input-dim-slots` and
    therefore BEFORE `hip-slot-lifetime-coalesce`. Identity rebind
    simplifies the input the coalescer sees.
  }];
  let options = [
    Option<"disableIdentityRebind", "disable-identity-rebind", "bool",
           /*default=*/"false",
           "Skip the identity rebind rewrite (for A/B kernel-launch testing)">
  ];
  let dependentDialects = [
    "mlir::hip::HipDialect",
    "mlir::func::FuncDialect"
  ];
}

def SlotLifetimeCoalescePass
    : Pass<"hip-slot-lifetime-coalesce", "mlir::ModuleOp"> {
  let summary = "Coalesce slot ids whose buffers can share storage at runtime";
  let description = [{
    Phase 3 of the slot-buffer-coalesce design
    (docs/design/slot-buffer-coalesce.md).

    Two published slots may share a single dyn-pool buffer iff:
      1. Their published dim DimSpecs are STRUCTURALLY EQUAL after
         `DimSpec::canonicalize()` -- the per-element byte size is
         guaranteed identical at runtime regardless of input values.
      2. Their producer/consumer LIFETIMES in `@main_graph` do NOT
         overlap (last consumer of slot A strictly before first
         producer of slot B, or vice versa).
      3. The two slots have a COMPATIBLE output-binding classification:
         output-bound slots (consumed by a hipdnn.output_index func
         result) live past `inference_compute` return and can only
         share with another output-bound slot of identical lifetime,
         or with an intermediate slot that dies before the output's
         producer runs.

    Algorithm:
      1. Collect every slot id published anywhere in @main_graph
         (scan `slot_id`, `slot_ids`, `hipdnn.output_slot_ids`).
      2. For each slot, derive its canonical DimSpec (from the
         publisher's `output_dim_specs` attribute on the matching
         (result, dim)), define-index, last-use-index, and
         output-bound flag.
      3. Group slots by canonical-DimSpec serialized bytes.
      4. Within each group, first-fit-decreasing bin pack by
         lifetime so non-overlapping slots end up in one bin.
         Output-bound slots inherit lastUseIndex = +inf.
      5. Pick the smallest slot id in each bin as the
         representative; rewrite every other slot id in the bin to
         the representative throughout the module (publishers,
         consumers, DimSpec leaves, both per-op and module-level).
      6. Renumber surviving slot ids to a contiguous 0..K-1 range
         and update `hipdnn.dyn_dim_slots_count` so
         `GenerateInterface` emits the reduced count into the
         FlatBuffers metadata.

    Pipeline position: between `hip-annotate-input-dim-slots` and
    `convert-hip-to-llvm`. Annotation must run first so any later
    consumer-side attributes get rewritten in step 5.

    Backward compat: when no slots are reserved (static-shape
    graphs), the pass is a no-op. Disabled with the
    `disable-coalesce` option for A/B testing of footprint
    impact.
  }];
  let options = [
    Option<"disableCoalesce", "disable-coalesce", "bool", /*default=*/"false",
           "Skip the coalescing rewrite (use only for A/B footprint testing)">
  ];
  let dependentDialects = [
    "mlir::hip::HipDialect",
    "mlir::func::FuncDialect"
  ];
}

def ElideSlotPublisherAllocsPass
    : Pass<"hip-elide-slot-publisher-allocs", "mlir::func::FuncOp"> {
  let summary = "Shrink the DPS-init memref.alloc of Cat-C slot publishers "
                "(NonZero, Range Cat-C, ConstantOfShape Cat-C) to a 0-byte "
                "placeholder so the static pool reserves no bytes for it";
  let description = [{
    Cat-C publishers (NonZero, Range Cat-C, ConstantOfShape Cat-C) allocate
    and publish their OWN exact-size GPU buffer at runtime through the
    `hipdnn_ep_state_dyn_pool_alloc` + `hipdnn_ep_state_publish_buffer`
    pair. The DPS-init `memref.alloc` that bufferize materialised for them
    holds an UPPER BOUND (numel(input) for NonZero, ceildiv-of-range for
    Range, product-of-shape for ConstantOfShape). That buffer is reserved
    by `hip-pool-allocs` in the static pool but is NEVER read or written
    by the publisher's runtime wrapper -- it is dead memory.

    This pass marks each such alloc as a 0-byte placeholder by replacing
    its dynamic upper-bound size operand with `arith.constant 0 : index`.
    The alloc's TYPE is preserved so SSA uses (including the publisher's
    DPS-init operand and any descriptor-based queries) stay well-typed,
    but pool-allocs will then pack a 0-byte dynamic alloc whose bucket
    contributes 0 to the pool footprint.

    Gating: only fires on ops carrying the `hipdnn.elide_dps_init` UnitAttr
    attached by the Cat-C converter. The marker is the contract -- the
    publisher's wrapper has already promised to publish a slot buffer
    via the slot API, and every consumer downstream will either:
      (a) read the runtime-published buffer via `hipdnn.input_slot_buffers`
          (AnnotateInputDimSlotsPass handles this later in the pipeline), or
      (b) consume the alloc descriptor for its rank/dtype only (the
          0-byte size is never the answer to a dim query because dynamic
          dim queries on slot-bound operands are rewired via
          `hipdnn.input_dim_slots`).

    Safety walk: before shrinking, the pass verifies that every non-dealloc
    use of the alloc result is one of:
      * the publisher op itself (DPS-init operand);
      * another hip-dialect op (future consumer that will be annotated);
      * a `memref.dim` op (descriptor-based dim query -- the upper-bound
        value the alloc once held was never the right answer anyway,
        and lowering of the consumer's slot-aware dim query goes through
        `hipdnn_ep_state_read_dim`).
    Any OTHER use (non-hip dialect, non-dim) causes the pass to skip
    that publisher conservatively. This keeps the pass safe in the face
    of unexpected IR shapes.

    Pipeline position:
      bufferize -> dealloc -> CSE/canonicalize ->
      ELIDE-SLOT-PUBLISHER-ALLOCS -> hip-pool-allocs
  }];
  let dependentDialects = [
    "mlir::hip::HipDialect",
    "mlir::arith::ArithDialect",
    "mlir::memref::MemRefDialect"
  ];
}

def AnnotateInputDimSlotsPass
    : Pass<"hip-annotate-input-dim-slots", "mlir::ModuleOp"> {
  let summary = "Tag every consumer operand with the (dim_idx, slot_id) "
                "pairs of any RuntimeSlot leaves in the producer's "
                "output_dim_specs (transitively)";
  let description = [{
    Walks every op in the module. For each operand whose producer's
    `output_dim_specs` (resolved via `shape_interface::getResultDimSpec`)
    contains one or more `RuntimeSlot` leaves, attaches a per-operand
    `hipdnn.input_dim_slots` attribute. The attribute is an `ArrayAttr`
    with one entry per operand of the op; each entry is itself an
    `ArrayAttr` of `DenseI32ArrayAttr` two-element arrays `[dim_idx,
    slot_id]`, listing every dynamic dim of that operand that resolves
    to a slot.

    Rationale: bufferized memrefs encode the *upper-bound* shape (the
    DPS init buffer that pool-allocs allocated). The HipToLLVM lowering
    of consumer ops would otherwise read `descriptor.sizes[d]` of the
    upper-bound buffer for any dynamic dim — that value is the
    pre-allocated maximum, not the runtime-published exact size. By
    propagating the slot-id at compile time, the per-op lowering can
    emit a `hipdnn_ep_state_read_dim(state, slot_id)` call for the
    affected dim and feed the kernel the right element count.

    Producers handled today:
      * `hip.nonzero` (slot dim is the N output)
      * Any op whose `output_dim_specs` contains a `RuntimeSlot` leaf
        is treated as a producer, so future Category-C ops opt-in
        automatically.

    Consumer ops handled (lowering-side honour of the attribute):
      * `hip.transpose` (input dim size determines output element count)
      * `hip.scatter_nd` (indices dim 0 = number of scattered slices)
      * `hip.shape` (dim values are the result tensor)
      * `hip.slice` (data dim used in clamp; output dim sets write count)

    Other consumers (gather, elementwise, etc.) work unchanged because
    they either:
      * have a static result shape (no dynamic dim to override), or
      * route through an op whose lowering already calls back into
        `hipdnn_ep_state_read_dim` directly (e.g. the dyn-pool path).

    The pass is a pre-`convert-hip-to-llvm` annotator. It does not
    rewrite IR — only adds attributes — so it is a *safe* analysis: a
    subsequent lowering that does not look at the attribute remains
    correct for the static-shape case and exhibits the pre-pass
    behaviour for the dynamic case.

    Pipeline position: between `hip-compose-dim-specs` and
    `convert-hip-to-llvm`. Runs after ComposeDimSpecs only because
    `RuntimeSlot` leaves are present on every op that publishes a slot
    after `OnnxToHip`; ComposeDimSpecs does not modify per-op specs and
    runs first only to keep the global `hipdnn.output_dim_specs`
    annotation phase together.
  }];
  let dependentDialects = ["mlir::hip::HipDialect"];
}

def DumpDimSpecsPass : Pass<"hip-dump-dim-specs", "mlir::ModuleOp"> {
  let summary = "Dump every per-op output_dim_specs and the module-level "
                "composed hipdnn.output_dim_specs to stderr";
  let description = [{
    Analysis-only pass for the data-dependent dynamic output shape feature.
    Walks the module and prints every DimSpec attached to a HIP op
    (via the `output_dim_specs` / `element_dim_specs` attribute) as well as
    the model-level composed attribute `hipdnn.output_dim_specs` produced
    by `hip-compose-dim-specs`.

    Output format (one DimSpec per line):
        Output[i] dim[d] = mul(arg[0].shape[0], slot[3])
                                                 ^^^^^^^ root of the tree
    Same `toString()` form the EP-side resolver tracer uses, so DimSpec
    diagnostics are textually consistent across compiler, EP, and runtime.

    Useful when working on `OnnxToHip` conversions for ops that produce
    dynamic-shape results, or when debugging a `ComposeDimSpecs`
    composition that resolves to a surprising tree shape. Run via:
        hip-mlir-opt --hip-dump-dim-specs <model.mlir>

    Does not modify the IR.
  }];
  let dependentDialects = ["mlir::hip::HipDialect"];
}

def VerifyDimSpecsPass : Pass<"hip-verify-dim-specs", "mlir::ModuleOp"> {
  let summary = "Verify that DimSpec metadata is structurally well-formed "
                "and covers every dynamic result dim of @main_graph";
  let description = [{
    Analysis-only pass. Returns failure when any of the following hold:
      * `hipdnn.output_dim_specs` exists but its outer rank does not
        match the @main_graph result count, or its per-output inner
        rank does not match each result's rank;
      * any DimSpec tree fails `DimSpec::verify()` (cycle, invalid child
        index, leaf with children, binary node with wrong arity, ...);
      * any dynamic (`?`) dim of an @main_graph result has no
        corresponding non-empty DimSpec entry. This catches a class of
        latent bugs where `ComposeDimSpecs` silently failed to resolve
        a dim and left an empty entry — the runtime would later
        fall back to the legacy "-1 in shape" path and the EP would
        treat the dim as "unknown", producing confusing downstream
        crashes or wrong-shape allocations.
      * a RuntimeSlot leaf references a slot_id >= `hipdnn.dyn_dim_slots_count`.

    Phase 3 (post-SlotLifetimeCoalesce) invariants additionally checked:
      * Every `hipdnn.output_slot_ids` non-`-1` entry references a
        DYNAMIC dim of the op's result (a non-`-1` on a static dim
        leaks a slot id).
      * Output-bound slots (i.e. slot ids referenced inside
        `hipdnn.output_dim_specs` -- they live past `inference_compute`
        return) are NOT shared between two distinct publisher ops:
        coalescing such a slot would let the second publisher overwrite
        the value the EP-side resolver reads after stream sync. (Intra-
        graph intermediates sharing a slot via Phase 3 are allowed.)

    Typical use: as a final pre-emit gate inside a custom pipeline, or
    as a debugging check after editing `ComposeDimSpecs`:
        hip-mlir-opt --hip-compose-dim-specs --hip-verify-dim-specs <m.mlir>
  }];
  let dependentDialects = [
    "mlir::hip::HipDialect",
    "mlir::func::FuncDialect"
  ];
}

def PrintPoolStatsPass
    : Pass<"hip-print-pool-stats", "mlir::ModuleOp"> {
  let summary = "Emit slot-buffer-coalesce footprint stats (static pool, "
                "dyn slot count, output dim spec count) to stderr";
  let description = [{
    Analysis-only pass that prints the current state of the
    slot-buffer-coalesce metadata on a module to stderr. Used as the
    "before vs. after" reference when measuring the effect of an
    individual phase pass (Phase 1 ElideSlotPublisherAllocs, Phase 3
    SlotLifetimeCoalesce, Phase 4 IdentityPropagatorRebind).

    Output format (one block per @main_graph):

      [pool-stats] func=main_graph
        static_pool_bytes = NNNN
        dyn_dim_slots_count = N
        next_dyn_slot_id = N
        output_dim_specs.count = N
        identity_propagator_predicate_hits = N
        slot_publisher_count = N

    `static_pool_bytes` is read from the `hipdnn.pool_size` module
    attribute (set by `hip-pool-allocs`). `dyn_dim_slots_count` is the
    post-Phase-3 surviving slot count. `output_dim_specs.count` is the
    number of dynamic dims that resolved to a DimSpec (one entry per
    output * rank). `identity_propagator_predicate_hits` counts how
    many ops in the module would be elided by `IdentityPropagatorRebind`
    -- a quick way to detect whether Phase 4 should have removed
    something but didn't. `slot_publisher_count` is the number of ops
    publishing one or more slots (Cat-C ops + Phase 2 propagators).

    Run as part of a pipeline:
      hip-mlir-opt model.mlir --hip-print-pool-stats --hipdnn-pipeline --hip-print-pool-stats

    The pass does not modify IR. Output goes to stderr so it does not
    interfere with the textual MLIR output on stdout.
  }];
  let dependentDialects = [
    "mlir::hip::HipDialect",
    "mlir::func::FuncDialect"
  ];
}

def ResolveExternConstantsPass
    : Pass<"hip-resolve-extern-constants", "mlir::ModuleOp"> {
  let summary = "Replace extern memref.global constants with "
                "hip.get_constant ops";
  let description = [{
    Resolves externalized constants (memref.global ops with hip.external_data
    attributes produced by --convert-onnx-to-hip) by replacing each
    memref.get_global with a hip.get_constant(%ctx, %index) operation.

    The index is read from the hip.external_data attribute on each extern
    memref.global. The !hip.context is taken from function argument 0
    (inserted by --hip-add-context-arg).

    The runtime is responsible for loading the sidecar constants.bin file
    via morphizen::FileSystem, uploading constants to GPU memory, and
    returning GPU pointers via hipdnn_ep_constant_get(state, index).

    This pass should run after all buffer optimizations (hip-pool-allocs,
    hip-lower-allocs) and before --convert-hip-to-llvm.
  }];
  let dependentDialects = [
    "mlir::hip::HipDialect",
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect",
    "mlir::memref::MemRefDialect"
  ];
}

#endif // HIP_PASSES
