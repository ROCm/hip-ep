#!r6rs
;;===----------------------------------------------------------------------===;;
;;
;; Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
;; Licensed under the MIT License.
;;
;;===----------------------------------------------------------------------===;;
;;
;; (mlir core ir) — Core MLIR IR primitives
;;
;; Mirrors mlir/IR/ in the MLIR project: operations, values, types, blocks,
;; regions, the builder API, logging, and raw attribute access.
;;
;; All MLIR pointers are represented as exact integers via the `uptr` FFI type.
;; See (mlir core ir) for the full pointer-convention documentation.
;;
;;===----------------------------------------------------------------------===;;

(library (mlir core ir)

  (export
    ;; Logging (re-exported from (mlir core logging))
    mlir-log-trace mlir-log-debug mlir-log-info
    mlir-log-warning mlir-log-error mlir-log-fatal

    ;; ValueArrayRef accessors
    value-array-ref-size
    value-array-ref-at

    ;; Operation inspection
    mlir-operation-name
    mlir-operation-get-context
    mlir-operation-num-operands
    mlir-operation-num-results
    mlir-operation-get-operand
    mlir-operation-get-result
    mlir-operation-get-parent
    mlir-operation-get-operand-value
    mlir-operation-get-result-value
    mlir-operation-get-loc
    mlir-operation-get-block-argument
    mlir-operation-walk

    ;; Value operations
    mlir-value-get-defining-op
    mlir-value-get-type
    mlir-value-is-block-argument?
    mlir-value-get-result-number

    ;; Attribute access and mutation
    mlir-operation-set-attr!
    mlir-operation-get-attr
    mlir-operation-copy-attr
    mlir-operation-has-attr?
    mlir-operation-set-operand
    mlir-operation-use-empty?
    mlir-operation-num-dps-inits
    mlir-operation-get-dps-init-value

    ;; Type queries
    mlir-type-is-ranked-tensor
    mlir-type-get-element-type
    mlir-type-get-shape
    mlir-type-get-rank
    mlir-type-get-encoding

    ;; Logging

    ;; Builder — explicit rewriter-based op construction
    mlir-build-operation-op
    mlir-set-insertion-point-before
    mlir-set-insertion-point-to-block-end
    mlir-op-get-region
    mlir-region-create-block
    mlir-block-get-argument

    ;; Shape dialect types (TODO: move to (mlir dialects shape) when it exists)

    ;; Pattern rewriting
    mlir-replace-op
    mlir-erase-op
    mlir-op-erase
    mlir-emit-error!
    mlir-emit-warning!
    mlir-emit-remark!

    ;; Generic RAII (type-specific RAII macros are in (mlir core conversion))
    with-raii

    mlir-type-set-memory-space

    ;; More builder ops
    mlir-build-operation-op-with-regions
    mlir-build-operation-op-in-block
    mlir-build-operation-op-in-block-with-regions
    mlir-builder-at-block-end
    mlir-destroy-builder
    mlir-new-block
    mlir-create-op

    ;; More type getters
    mlir-get-index-type
    mlir-get-i64-type
    mlir-get-i1-type

    ;; More attr helpers

    ;; Dynamic builder context
    current-rewriter
    current-block-builder
    current-loc
    mlir-build-operation
    with-rewrite-builder
    with-current-block-builder
    with-block-builder
    with-op-location

    )

  (import (chezscheme)
          (mlir core logging)
          (mlir dialects shape))

  ;;===--------------------------------------------------------------------===;;
  ;; Foreign Type Definitions
  ;;===--------------------------------------------------------------------===;;

  (define-ftype ValueArrayRef
    (struct
      [data uptr]
      [size uptr]))

  ;;===--------------------------------------------------------------------===;;
  ;; Operation Inspection
  ;;===--------------------------------------------------------------------===;;

  (define mlir-operation-name
    (foreign-procedure "mlir_operation_get_name" (uptr) string))

  (define mlir-operation-get-context
    (foreign-procedure "mlir_operation_get_context" (uptr) uptr))

  (define mlir-operation-num-operands
    (foreign-procedure "mlir_operation_num_operands" (uptr) iptr))

  (define mlir-operation-num-results
    (foreign-procedure "mlir_operation_num_results" (uptr) iptr))

  (define mlir-operation-get-operand
    (foreign-procedure "mlir_operation_get_operand" (uptr iptr) uptr))

  (define mlir-operation-get-result
    (foreign-procedure "mlir_operation_get_result" (uptr iptr) uptr))

  (define mlir-value-get-defining-op
    (foreign-procedure "mlir_value_get_defining_op" (uptr) uptr))

  (define mlir-value-is-block-argument
    (foreign-procedure "mlir_value_is_block_argument" (uptr) int))

  (define (mlir-value-is-block-argument? v)
    (= 1 (mlir-value-is-block-argument v)))

  (define mlir-value-get-result-number
    (foreign-procedure "mlir_value_get_result_number" (uptr) int))

  (define mlir-operation-get-parent
    (foreign-procedure "mlir_operation_get_parent" (uptr) uptr))

  (define mlir-operation-get-operand-value
    (foreign-procedure "mlir_operation_get_operand_value" (uptr int) uptr))

  (define mlir-operation-get-result-value
    (foreign-procedure "mlir_operation_get_result_value" (uptr int) uptr))

  (define mlir-operation-get-loc
    (foreign-procedure "mlir_operation_get_loc" (uptr) uptr))

  (define mlir-operation-get-block-argument
    (foreign-procedure "mlir_operation_get_block_argument" (uptr int) uptr))

  (define mlir-operation-walk
    (foreign-procedure "mlir_operation_walk" (uptr scheme-object) void))

  ;;===--------------------------------------------------------------------===;;
  ;; Value Operations
  ;;===--------------------------------------------------------------------===;;

  (define mlir-value-get-type
    (foreign-procedure "mlir_value_get_type" (uptr) uptr))

  ;;===--------------------------------------------------------------------===;;
  ;; Attribute Access and Mutation
  ;;===--------------------------------------------------------------------===;;

  (define %set-i64-attr
    (foreign-procedure "mlir_operation_set_attr" (uptr string iptr) void))

  (define (mlir-operation-set-attr! op name value type)
    (case type
      [(:i64)       (%set-i64-attr op name value)]
      [(:index)     (%set-index-attr op name value)]
      [(:i32-array) (%set-i32-array-attr op name value)]
      [(:i64-array) (%set-i64-array-attr op name value)]
      [else (error 'mlir-operation-set-attr! "unknown attr type" type)]))

  (define %get-string-attr
    (foreign-procedure "mlir_operation_get_string_attr" (uptr string) string))
  (define %get-i64-attr
    (foreign-procedure "mlir_operation_get_integer_attr" (uptr string integer-64) integer-64))
  (define %get-i64-array-attr
    (foreign-procedure "mlir_operation_get_integer_array_attr" (uptr string) scheme-object))
  (define %set-i64-array-attr
    (foreign-procedure "mlir_operation_set_dense_i64_array" (uptr string scheme-object) void))

  (define (mlir-operation-get-attr op name type . rest)
    (case type
      [(:string)    (%get-string-attr op name)]
      [(:i64)       (%get-i64-attr op name (if (null? rest) 0 (car rest)))]
      [(:i64-array) (%get-i64-array-attr op name)]
      [else (error 'mlir-operation-get-attr "unknown attr type" type)]))

  (define mlir-operation-copy-attr
    (foreign-procedure "mlir_operation_copy_attr" (uptr string uptr string) void))

  (define (mlir-operation-has-attr? op name)
    (= 1 ((foreign-procedure "mlir_operation_has_attr" (uptr string) int) op name)))

  (define mlir-operation-set-operand
    (foreign-procedure "mlir_operation_set_operand" (uptr int uptr) void))

  ;; Returns #t when all results of the operation have no uses (op is dead).
  (define (mlir-operation-use-empty? op)
    (= 1 ((foreign-procedure "mlir_operation_use_empty" (uptr) int) op)))

  (define mlir-operation-num-dps-inits
    (foreign-procedure "mlir_operation_num_dps_inits" (uptr) int))

  (define mlir-operation-get-dps-init-value
    (foreign-procedure "mlir_operation_get_dps_init_value" (uptr int) uptr))

  ;;===--------------------------------------------------------------------===;;
  ;; Type Queries
  ;;===--------------------------------------------------------------------===;;

  (define mlir-type-is-ranked-tensor
    (foreign-procedure "mlir_type_is_ranked_tensor" (uptr) int))

  (define mlir-type-get-element-type
    (foreign-procedure "mlir_type_get_element_type" (uptr) uptr))

  (define mlir-type-get-shape
    (foreign-procedure "mlir_type_get_shape" (uptr) scheme-object))

  (define mlir-type-get-rank
    (foreign-procedure "mlir_type_get_rank" (uptr) int))

  (define mlir-type-get-encoding
    (foreign-procedure "mlir_type_get_encoding" (uptr) uptr))

  ;;===--------------------------------------------------------------------===;;
  ;; Logging
  ;;===--------------------------------------------------------------------===;;


  ;;===--------------------------------------------------------------------===;;
  ;; Builder — explicit rewriter-based op construction
  ;;===--------------------------------------------------------------------===;;

  ;;; Create an op at the current rewriter insertion point.
  ;;; operands and result-types are Scheme lists of uptr values.
  (define mlir-build-operation-op
    (foreign-procedure "mlir_build_op"
                       (uptr uptr string scheme-object scheme-object) uptr))

  (define mlir-set-insertion-point-before
    (foreign-procedure "mlir_set_insertion_point_before" (uptr uptr) void))

  (define mlir-set-insertion-point-to-block-end
    (foreign-procedure "mlir_set_insertion_point_to_block_end" (uptr uptr) void))

  ;;; Get the i-th region of an operation.
  (define mlir-op-get-region
    (foreign-procedure "mlir_op_get_region" (uptr int) uptr))

  ;;; Create a block in a region with given arg types; sets IP to its end.
  ;;; arg-types is a Scheme list of Type* uptrs.
  (define mlir-region-create-block
    (foreign-procedure "mlir_region_create_block" (uptr uptr scheme-object) uptr))

  ;;; Get the i-th argument of a block as a Value* uptr.
  (define mlir-block-get-argument
    (foreign-procedure "mlir_block_get_argument" (uptr int) uptr))

  ;;===--------------------------------------------------------------------===;;
  ;; Pattern Rewriting
  ;;===--------------------------------------------------------------------===;;

  ;;; Replace old-op with new-val via rewriter.
  (define mlir-replace-op
    (foreign-procedure "mlir_replace_op" (uptr uptr uptr) int))

  ;;; Erase op via rewriter.
  (define mlir-erase-op
    (foreign-procedure "mlir_erase_op" (uptr uptr) int))

  ;;; Erase op directly without a rewriter (for post-pass cleanup).
  (define mlir-op-erase
    (foreign-procedure "mlir_op_erase" (uptr) void))

  ;;===--------------------------------------------------------------------===;;
  ;; MLIR Diagnostic Emission
  ;;
  ;; Each function takes ctx and an optional loc.
  ;; loc defaults to (current-loc); when #f, the C++ side uses UnknownLoc.
  ;;===--------------------------------------------------------------------===;;

  ;; Emit an MLIR diagnostic attached to op, routing through MLIR's diagnostic
  ;; engine (visible to ORT's handler, mlir-opt output, etc.).
  ;; When op is #f/0 (no op in scope), falls back to mlir_log_*.
  ;; These do not raise Scheme exceptions; callers propagate failure explicitly.
  (define mlir-emit-error!
    (foreign-procedure "mlir_emit_error"   (uptr string) void))
  (define mlir-emit-warning!
    (foreign-procedure "mlir_emit_warning" (uptr string) void))
  (define mlir-emit-remark!
    (foreign-procedure "mlir_emit_remark"  (uptr string) void))

  ;;===--------------------------------------------------------------------===;;
  ;; ValueArrayRef Accessors
  ;;===--------------------------------------------------------------------===;;

  (define (value-array-ref-size ref-ptr)
    (ftype-ref ValueArrayRef (size) (make-ftype-pointer ValueArrayRef ref-ptr)))

  (define (value-array-ref-at ref-ptr index)
    (let* ([ptr (make-ftype-pointer ValueArrayRef ref-ptr)]
           [data-ptr (ftype-ref ValueArrayRef (data) ptr)])
      (foreign-ref 'uptr data-ptr (* index 8))))

  ;;===--------------------------------------------------------------------===;;
  ;; Generic RAII
  ;;===--------------------------------------------------------------------===;;

  ;;; (with-raii ((var ctor dtor) ...) body ...)
  ;;; Each resource is created by ctor, bound to var, and destroyed by (dtor var)
  ;;; on exit — whether normal, exception, or continuation escape.
  ;; Single-resource RAII: (with-raii (var ctor dtor) body ...)
  (define-syntax with-raii
    (syntax-rules ()
      [(_ (var ctor dtor) body ...)
       (let ([var ctor])
         (dynamic-wind void
           (lambda () body ...)
           (lambda () (dtor var))))]))


  ;; ── Additional type helpers ──────────────────────────────────────────────

  (define mlir-type-set-memory-space
    (foreign-procedure "mlir_type_set_memory_space" (uptr int) uptr))



  (define mlir-build-operation-op-with-regions
    (foreign-procedure "mlir_build_op_with_regions"
                       (uptr uptr string scheme-object scheme-object int) uptr))

  (define mlir-build-operation-op-in-block
    (foreign-procedure "mlir_build_op_in_block"
                       (uptr uptr string scheme-object scheme-object) uptr))

  (define mlir-build-operation-op-in-block-with-regions
    (foreign-procedure "mlir_build_op_in_block_with_regions"
                       (uptr uptr string scheme-object scheme-object int) uptr))

  (define mlir-builder-at-block-end
    (foreign-procedure "mlir_builder_at_block_end" (uptr) uptr))

  (define mlir-destroy-builder
    (foreign-procedure "mlir_destroy_builder" (uptr) void))

  (define mlir-new-block
    (foreign-procedure "mlir_new_block" (uptr scheme-object) uptr))

  (define mlir-create-op%
    (foreign-procedure "mlir_create_op" (uptr uptr string scheme-object scheme-object int) uptr))
  (define (mlir-create-op builder loc name ops types . rest)
    (mlir-create-op% builder loc name ops types (if (pair? rest) (car rest) 0)))


  (define mlir-get-index-type
    (foreign-procedure "mlir_get_index_type" (uptr) uptr))

  (define mlir-get-i64-type
    (foreign-procedure "mlir_get_i64_type" (uptr) uptr))

  (define mlir-get-i1-type
    (foreign-procedure "mlir_get_i1_type" (uptr) uptr))



  (define %set-index-attr
    (foreign-procedure "mlir_operation_set_index_attr" (uptr string iptr) void))

  (define %set-i32-array-attr
    (foreign-procedure "mlir_operation_set_dense_i32_array" (uptr string scheme-object) void))



  ;; ── Dynamic builder context ──────────────────────────────────────────────

  (define current-rewriter      (make-parameter #f))
  (define current-block-builder (make-parameter #f))
  (define current-loc           (make-parameter #f))

  ;; Dispatch to the active builder context.
  ;; Requires either current-rewriter or current-block-builder to be non-#f;
  ;; raises an error if neither is set (i.e., called outside a builder scope).
  (define (mlir-build-operation name operands types . rest)
    (let ([nregions (if (pair? rest) (car rest) 0)]
          [loc      (current-loc)])
      (cond
        [(current-rewriter) =>
         (lambda (rw)
           (if (zero? nregions)
               (mlir-build-operation-op rw loc name operands types)
               (mlir-build-operation-op-with-regions rw loc name operands types nregions)))]
        [(current-block-builder) =>
         (lambda (b)
           (if (zero? nregions)
               (mlir-build-operation-op-in-block b loc name operands types)
               (mlir-build-operation-op-in-block-with-regions b loc name operands types nregions)))]
        [else (error 'mlir-build-operation "no current builder installed")])))

  (define-syntax with-rewrite-builder
    (syntax-rules ()
      [(_ (rw loc) body ...)
       (parameterize ([current-rewriter rw] [current-block-builder #f] [current-loc loc])
         body ...)]))

  (define-syntax with-current-block-builder
    (syntax-rules ()
      [(_ (builder loc) body ...)
       (parameterize ([current-block-builder builder] [current-rewriter #f] [current-loc loc])
         body ...)]))

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

  (define-syntax with-op-location
    (syntax-rules ()
      [(_ loc body ...)
       (parameterize ([current-loc loc]) body ...)]))

) ;; end library (mlir core ir)
