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
    mlir-type-set-memory-space)

  (import (rnrs)
          (only (chezscheme) foreign-procedure parameterize make-parameter
                dynamic-wind void))

  ;;===--------------------------------------------------------------------===;;
  ;; Low-level rewriter FFI
  ;;===--------------------------------------------------------------------===;;

  (define mlir-replace-op
    (foreign-procedure "mlir_replace_op" (uptr uptr uptr) int))
  (define mlir-erase-op
    (foreign-procedure "mlir_erase_op" (uptr uptr) int))
  (define mlir-op-erase
    (foreign-procedure "mlir_op_erase" (uptr) void))
  (define mlir-set-insertion-point-before
    (foreign-procedure "mlir_set_insertion_point_before" (uptr uptr) void))
  (define mlir-set-insertion-point-to-block-end
    (foreign-procedure "mlir_set_insertion_point_to_block_end" (uptr uptr) void))

  ;;===--------------------------------------------------------------------===;;
  ;; Block / region / builder FFI
  ;;===--------------------------------------------------------------------===;;

  (define %build-op
    (foreign-procedure "mlir_build_op"
                       (uptr uptr string scheme-object scheme-object) uptr))
  (define %build-op-regions
    (foreign-procedure "mlir_build_op_with_regions"
                       (uptr uptr string scheme-object scheme-object int) uptr))
  (define %build-op-in-block
    (foreign-procedure "mlir_build_op_in_block"
                       (uptr uptr string scheme-object scheme-object) uptr))
  (define %build-op-in-block-regions
    (foreign-procedure "mlir_build_op_in_block_with_regions"
                       (uptr uptr string scheme-object scheme-object int) uptr))

  (define mlir-op-get-region
    (foreign-procedure "mlir_op_get_region" (uptr int) uptr))
  (define mlir-region-create-block
    (foreign-procedure "mlir_region_create_block" (uptr uptr scheme-object) uptr))
  (define mlir-block-get-argument
    (foreign-procedure "mlir_block_get_argument" (uptr int) uptr))
  (define mlir-new-block
    (foreign-procedure "mlir_new_block" (uptr scheme-object) uptr))
  (define mlir-builder-at-block-end
    (foreign-procedure "mlir_builder_at_block_end" (uptr) uptr))
  (define mlir-destroy-builder
    (foreign-procedure "mlir_destroy_builder" (uptr) void))

  (define %mlir-create-op
    (foreign-procedure "mlir_create_op"
                       (uptr uptr string scheme-object scheme-object int) uptr))
  (define (mlir-create-op builder loc name ops types . rest)
    (%mlir-create-op builder loc name ops types (if (pair? rest) (car rest) 0)))

  ;;===--------------------------------------------------------------------===;;
  ;; Type constructors / queries
  ;;===--------------------------------------------------------------------===;;

  (define mlir-get-index-type
    (foreign-procedure "mlir_get_index_type" (uptr) uptr))
  (define mlir-get-i64-type
    (foreign-procedure "mlir_get_i64_type" (uptr) uptr))
  (define mlir-get-i1-type
    (foreign-procedure "mlir_get_i1_type" (uptr) uptr))

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
  (define mlir-type-set-memory-space
    (foreign-procedure "mlir_type_set_memory_space" (uptr int) uptr))

  ;;===--------------------------------------------------------------------===;;
  ;; Generic RAII
  ;;===--------------------------------------------------------------------===;;

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

  (define current-rewriter      (make-parameter #f))
  (define current-block-builder (make-parameter #f))
  (define current-loc           (make-parameter #f))

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

) ;; end library (mlir core builder)
