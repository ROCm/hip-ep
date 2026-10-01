#!r6rs
;;===----------------------------------------------------------------------===;;
;;
;; Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
;; Licensed under the MIT License.
;;
;;===----------------------------------------------------------------------===;;
;;
;; (mlir core builder) — MLIR builder API and dynamic builder context.
;;
;; Mirrors mlir/IR/Builders.h. Provides:
;;   - Dynamic context parameters (current-rewriter, current-block-builder,
;;     current-loc) and their RAII macros.
;;   - mlir-build-operation: context-dispatching op constructor.
;;   - Low-level builder FFI (block/region management).
;;   - Generic with-raii.
;;
;;===----------------------------------------------------------------------===;;

(library (mlir core builder)
  (export
    ;; Dynamic context
    current-rewriter
    current-block-builder
    current-loc
    ;; Context-dispatching constructor
    mlir-build-operation
    ;; RAII macros
    with-raii
    with-rewrite-builder
    with-current-block-builder
    with-block-builder
    with-op-location
    ;; Low-level rewriter ops
    mlir-build-op
    mlir-replace-op
    mlir-erase-op
    mlir-op-erase
    mlir-set-insertion-point-before
    mlir-set-insertion-point-to-block-end
    ;; Block / region primitives
    mlir-op-get-region
    mlir-region-create-block
    mlir-block-get-argument
    mlir-new-block
    mlir-builder-at-block-end
    mlir-destroy-builder
    mlir-create-op
    ;; Type context
    mlir-type-get-context
    ;; Type constructors needed by builder callers
    mlir-get-index-type
    mlir-get-i64-type
    mlir-get-i1-type
    ;; Type queries
    mlir-type-is-ranked-tensor
    mlir-type-get-element-type
    mlir-type-get-shape
    mlir-type-get-rank
    mlir-type-get-encoding
    mlir-type-set-memory-space
    ;; Pattern application
    mlir-apply-patterns-greedy
    ;; Generic op rebuild
    mlir-op-clone-with-types
    ;; hip fusion C++ helpers
    hip-extract-splat-scale
    hip-build-init
    hip-create-requantized-layout-op)

  (import (rnrs)
          (only (chezscheme) foreign-procedure parameterize make-parameter void))

  ;;===--------------------------------------------------------------------===;;
  ;; Low-level rewriter FFI
  ;;===--------------------------------------------------------------------===;;

  ;; Replace old-op's results with new-val via the rewriter.
  ;; rewriter: RewriterBase* uptr, old-op: Operation* uptr, new-val: Value* uptr
  ;; Returns 1 on success, 0 on null input.
  (define mlir-replace-op
    (foreign-procedure "mlir_replace_op" (uptr uptr uptr) int))

  ;; Erase old-op via the rewriter (notifies listeners).
  ;; rewriter: RewriterBase* uptr, op: Operation* uptr
  ;; Returns 1 on success, 0 on null input.
  (define mlir-erase-op
    (foreign-procedure "mlir_erase_op" (uptr uptr) int))

  ;; Erase an op directly without a rewriter (for post-pass cleanup).
  ;; op: Operation* uptr — must have no uses
  (define mlir-op-erase
    (foreign-procedure "mlir_op_erase" (uptr) void))

  ;; Set the rewriter's insertion point to immediately before op.
  ;; rewriter: RewriterBase* uptr, op: Operation* uptr
  (define mlir-set-insertion-point-before
    (foreign-procedure "mlir_set_insertion_point_before" (uptr uptr) void))

  ;; Set the rewriter's insertion point to the end of a block.
  ;; rewriter: RewriterBase* uptr, block: Block* uptr
  (define mlir-set-insertion-point-to-block-end
    (foreign-procedure "mlir_set_insertion_point_to_block_end" (uptr uptr) void))

  ;;===--------------------------------------------------------------------===;;
  ;; Block / region / builder FFI
  ;;===--------------------------------------------------------------------===;;

  ;; Build an op at the rewriter's current insertion point (set to before loc-op).
  ;; rewriter:     RewriterBase* uptr
  ;; loc-op:       Operation* uptr — source of location and insertion point
  ;; op-name:      string, e.g. "hipsr.cast"
  ;; operands:     Scheme list of Value* uptrs
  ;; result-types: Scheme list of Type* uptrs
  ;; Returns: Operation* uptr of the newly created op
  (define %build-op
    (foreign-procedure "mlir_build_op"
                       (uptr uptr string scheme-object scheme-object) uptr))
  ;; Public alias — mirrors the C symbol name mlir_build_op.
  (define mlir-build-op %build-op)

  ;; Like %build-op but pre-allocates num-regions empty regions.
  ;; Required for ops that verify they have exactly N regions at creation time.
  (define %build-op-regions
    (foreign-procedure "mlir_build_op_with_regions"
                       (uptr uptr string scheme-object scheme-object int) uptr))

  ;; Like %build-op but uses a plain OpBuilder* (from mlir-builder-at-block-end).
  ;; builder: OpBuilder* uptr (not a RewriterBase)
  (define %build-op-in-block
    (foreign-procedure "mlir_build_op_in_block"
                       (uptr uptr string scheme-object scheme-object) uptr))

  ;; Like %build-op-in-block but pre-allocates num-regions empty regions.
  (define %build-op-in-block-regions
    (foreign-procedure "mlir_build_op_in_block_with_regions"
                       (uptr uptr string scheme-object scheme-object int) uptr))

  ;; Get the i-th region of an operation.
  ;; op: Operation* uptr, i: 0-based region index
  ;; Returns: Region* uptr
  (define mlir-op-get-region
    (foreign-procedure "mlir_op_get_region" (uptr int) uptr))

  ;; Create a block inside a region with given argument types; sets IP to its end.
  ;; rewriter:   RewriterBase* uptr
  ;; region:     Region* uptr
  ;; arg-types:  Scheme list of Type* uptrs for the new block's arguments
  ;; Returns: Block* uptr
  (define mlir-region-create-block
    (foreign-procedure "mlir_region_create_block" (uptr uptr scheme-object) uptr))

  ;; Get the i-th block argument as a Value* uptr.
  ;; block: Block* uptr, i: 0-based argument index
  (define mlir-block-get-argument
    (foreign-procedure "mlir_block_get_argument" (uptr int) uptr))

  ;; Create a new Block in a region with the given argument types.
  ;; region:    Region* uptr
  ;; arg-types: Scheme list of Type* uptrs
  ;; Returns: Block* uptr
  (define mlir-new-block
    (foreign-procedure "mlir_new_block" (uptr scheme-object) uptr))

  ;; Create a heap-allocated OpBuilder positioned at the end of a block.
  ;; block: Block* uptr
  ;; Returns: OpBuilder* uptr — must be destroyed with mlir-destroy-builder
  (define mlir-builder-at-block-end
    (foreign-procedure "mlir_builder_at_block_end" (uptr) uptr))

  ;; Destroy an OpBuilder created by mlir-builder-at-block-end.
  ;; builder: OpBuilder* uptr
  (define mlir-destroy-builder
    (foreign-procedure "mlir_destroy_builder" (uptr) void))

  ;; Low-level op creation via an explicit OpBuilder* (not a rewriter).
  ;; builder: OpBuilder* uptr, loc: Operation* uptr (source of location)
  ;; name: string, ops: Scheme list of Value* uptrs
  ;; types: Scheme list of Type* uptrs, num-regions: int (default 0)
  ;; Returns: Operation* uptr
  (define %mlir-create-op
    (foreign-procedure "mlir_create_op"
                       (uptr uptr string scheme-object scheme-object int) uptr))
  (define (mlir-create-op builder loc name ops types . rest)
    (%mlir-create-op builder loc name ops types (if (pair? rest) (car rest) 0)))

  ;;===--------------------------------------------------------------------===;;
  ;; Type constructors / queries
  ;;===--------------------------------------------------------------------===;;

  ;; Get the MLIRContext* from any Type* (types carry their context).
  ;; type: Type opaque uptr (getAsOpaquePointer)
  ;; Returns: MLIRContext* uptr, or 0 on null input
  (define mlir-type-get-context
    (foreign-procedure "mlir_type_get_context" (uptr) uptr))

  ;; Construct the built-in IndexType for the given context.
  ;; ctx: MLIRContext* uptr  Returns: Type opaque uptr
  (define mlir-get-index-type
    (foreign-procedure "mlir_get_index_type" (uptr) uptr))

  ;; Construct IntegerType<64> for the given context.
  ;; ctx: MLIRContext* uptr  Returns: Type opaque uptr
  (define mlir-get-i64-type
    (foreign-procedure "mlir_get_i64_type" (uptr) uptr))

  ;; Construct IntegerType<1> (i1 / bool) for the given context.
  ;; ctx: MLIRContext* uptr  Returns: Type opaque uptr
  (define mlir-get-i1-type
    (foreign-procedure "mlir_get_i1_type" (uptr) uptr))

  ;; Returns 1 if the type is a RankedTensorType, 0 otherwise.
  ;; type: Type opaque uptr
  (define mlir-type-is-ranked-tensor
    (foreign-procedure "mlir_type_is_ranked_tensor" (uptr) int))

  ;; Get the element type of a shaped type (tensor, memref, vector).
  ;; type: ShapedType opaque uptr  Returns: element Type opaque uptr
  (define mlir-type-get-element-type
    (foreign-procedure "mlir_type_get_element_type" (uptr) uptr))

  ;; Get the shape of a ranked tensor as a Scheme list of exact integers.
  ;; Negative values indicate dynamic dimensions (mlir::ShapedType::kDynamic).
  ;; type: RankedTensorType opaque uptr  Returns: Scheme list of integers
  (define mlir-type-get-shape
    (foreign-procedure "mlir_type_get_shape" (uptr) scheme-object))

  ;; Get the rank (number of dimensions) of a ranked tensor type.
  ;; type: RankedTensorType opaque uptr  Returns: non-negative int
  (define mlir-type-get-rank
    (foreign-procedure "mlir_type_get_rank" (uptr) int))

  ;; Get the encoding attribute of a tensor type, or 0 if absent.
  ;; type: RankedTensorType opaque uptr
  ;; Returns: Attribute opaque uptr, or 0 if no encoding is set
  (define mlir-type-get-encoding
    (foreign-procedure "mlir_type_get_encoding" (uptr) uptr))

  ;; Clone a RankedTensorType with a new HipSR memory space (integer enum).
  ;; type:  RankedTensorType opaque uptr
  ;; space: int — MemorySpace enum value (1 = Device, 2 = Host, ...)
  ;; Returns: new RankedTensorType opaque uptr with the given encoding
  (define mlir-type-set-memory-space
    (foreign-procedure "mlir_type_set_memory_space" (uptr int) uptr))

  ;;===--------------------------------------------------------------------===;;
  ;; Generic RAII
  ;;===--------------------------------------------------------------------===;;

  ;; Single-resource RAII: bind var to (ctor), run body, call (dtor var) on exit.
  ;; Cleanup fires whether body returns normally, raises, or escapes.
  (define-syntax with-raii
    (syntax-rules ()
      [(_ (var ctor dtor) body ...)
       (let ([var ctor])
         (dynamic-wind void
           (lambda () body ...)
           (lambda () (dtor var))))]))

  ;;===--------------------------------------------------------------------===;;
  ;; Dynamic builder context
  ;;===--------------------------------------------------------------------===;;

  ;; Current ConversionPatternRewriter* (or #f when not in a pattern callback).
  (define current-rewriter      (make-parameter #f))

  ;; Current OpBuilder* for region/block filling (or #f when using a rewriter).
  (define current-block-builder (make-parameter #f))

  ;; Current location source Operation* uptr (or #f when unset).
  ;; Used by mlir-build-operation as the loc argument to the build functions.
  (define current-loc           (make-parameter #f))

  ;; Build an op using whichever builder context is currently active.
  ;; Dispatches to the rewriter path if current-rewriter is set, otherwise
  ;; to the block-builder path.  Raises if neither is installed.
  ;; name:      string op name, e.g. "hipsr.cast"
  ;; operands:  Scheme list of Value* uptrs
  ;; types:     Scheme list of result Type* uptrs
  ;; nregions:  optional int — number of empty regions to pre-allocate (default 0)
  ;; Returns: Operation* uptr of the created op
  (define (mlir-build-operation name operands types . rest)
    (let ([nregions (if (pair? rest) (car rest) 0)]
          [loc      (current-loc)])
      (cond
        [(current-rewriter) =>
         (lambda (rw)
           (if (zero? nregions)
               (%build-op rw loc name operands types)
               (%build-op-regions rw loc name operands types nregions)))]
        [(current-block-builder) =>
         (lambda (b)
           (if (zero? nregions)
               (%build-op-in-block b loc name operands types)
               (%build-op-in-block-regions b loc name operands types nregions)))]
        [else (error 'mlir-build-operation "no current builder installed")])))

  ;; Install rw as current-rewriter and loc as current-loc for the duration of body.
  ;; Sets current-block-builder to #f (rewriter and block-builder are mutually exclusive).
  ;; rw:  RewriterBase* uptr (passed by the pattern callback)
  ;; loc: Operation* uptr used as both the insertion-point anchor and location source
  (define-syntax with-rewrite-builder
    (syntax-rules ()
      [(_ (rw loc) body ...)
       (parameterize ([current-rewriter rw] [current-block-builder #f] [current-loc loc])
         body ...)]))

  ;; Install an explicit OpBuilder* as current-block-builder for the duration of body.
  ;; builder: OpBuilder* uptr, loc: Operation* uptr (location source)
  (define-syntax with-current-block-builder
    (syntax-rules ()
      [(_ (builder loc) body ...)
       (parameterize ([current-block-builder builder] [current-rewriter #f] [current-loc loc])
         body ...)]))

  ;; Create an OpBuilder at the end of block, install it as current-block-builder,
  ;; run body, then destroy the builder.  Inherits current-loc from the enclosing scope.
  ;; block: Block* uptr
  (define-syntax with-block-builder
    (syntax-rules ()
      [(_ block body ...)
       (let ([%builder (mlir-builder-at-block-end block)])
         (dynamic-wind
           (lambda () #f)
           (lambda ()
             (parameterize ([current-block-builder %builder]
                            [current-rewriter #f])
               body ...))
           (lambda () (mlir-destroy-builder %builder))))]))

  ;; Temporarily override current-loc with loc for the duration of body.
  ;; loc: Operation* uptr used as the location/insertion-point source
  (define-syntax with-op-location
    (syntax-rules ()
      [(_ loc body ...)
       (parameterize ([current-loc loc]) body ...)]))

  ;; === Pattern application ===

  (define mlir-apply-patterns-greedy
    (foreign-procedure "mlir_apply_patterns_greedy" (uptr uptr) boolean))

  ;; === Generic op rebuild ===

  (define mlir-op-clone-with-types
    (foreign-procedure "mlir_op_clone_with_types" (uptr uptr scheme-object scheme-object) uptr))

  ;; === hip fusion C++ helpers ===

  (define hip-extract-splat-scale
    (foreign-procedure "hip_extract_splat_scale" (uptr) double))

  (define hip-build-init
    (foreign-procedure "hip_build_init" (uptr uptr uptr) uptr))

  (define hip-create-requantized-layout-op
    (foreign-procedure "hip_create_requantized_layout_op" (uptr uptr uptr uptr) uptr))

) ;; end library (mlir core builder)
