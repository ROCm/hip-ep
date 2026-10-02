#!r6rs
;;===----------------------------------------------------------------------===;;
;;
;; Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
;; Licensed under the MIT License.
;;
;;===----------------------------------------------------------------------===;;
;;
;; (mlir core ir) — Re-export hub for core MLIR primitives.
;;
;; Importers that want everything in one place can use this library.
;; Prefer importing the specific sub-library when only a subset is needed.
;;
;;   (mlir core attribute) — make-mlir-attribute
;;   (mlir core operation) — mlir-operation-*, mlir-emit-*!
;;   (mlir core value)     — mlir-value-*, value-array-ref-*
;;   (mlir core builder)   — mlir-build-operation, with-*-builder, with-raii
;;   (mlir core logging)   — mlir-log-*
;;   (mlir dialects shape) — mlir-shape.{shape,size,witness}-type
;;
;;===----------------------------------------------------------------------===;;

(library (mlir core ir)
  (export
    ;; (mlir core attribute)
    make-mlir-attribute
    ;; (mlir core operation)
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
    mlir-operation-get-attribute
    mlir-operation-get-attr
    mlir-operation-set-attribute!
    mlir-operation-has-attr?
    mlir-operation-set-operand
    mlir-operation-use-empty?
    mlir-operation-num-dps-inits
    mlir-operation-get-dps-init-value
    mlir-emit-error!
    mlir-emit-warning!
    mlir-emit-remark!
    mlir-operation-set-f32-attr!
    mlir-operation-set-i64-attr!
    mlir-operation-set-unit-attr!
    mlir-operation-get-integer-attr
    ;; (mlir core value)
    mlir-value-get-defining-op
    mlir-value-get-type
    mlir-value-is-block-argument?
    mlir-value-get-result-number
    mlir-value-num-uses
    value-array-ref-size
    value-array-ref-at
    ;; (mlir core builder)
    mlir-type-get-context
    current-rewriter
    current-block-builder
    current-loc
    mlir-build-operation
    with-raii
    with-rewrite-builder
    with-current-block-builder
    with-block-builder
    with-op-location
    mlir-build-op
    mlir-replace-op
    mlir-erase-op
    mlir-op-erase
    mlir-set-insertion-point-before
    mlir-set-insertion-point-to-block-end
    mlir-op-get-region
    mlir-region-create-block
    mlir-block-get-argument
    mlir-new-block
    mlir-builder-at-block-end
    mlir-destroy-builder
    mlir-create-op
    mlir-get-index-type
    mlir-get-i64-type
    mlir-get-i1-type
    mlir-type-is-ranked-tensor
    mlir-type-get-element-type
    mlir-type-get-shape
    mlir-type-get-rank
    mlir-type-get-encoding
    mlir-type-set-memory-space
    ;; Type inspection
    mlir-type-element-type        ; element type of shaped/vector type
    mlir-type-integer-width       ; bit width of IntegerType
    mlir-type-is-unsigned         ; 1 if unsigned IntegerType
    ;; Attribute inspection
    mlir-op-get-float-attr        ; FloatAttr by name (NaN if absent)
    mlir-attr-is-splat            ; 1 if DenseElementsAttr splat
    mlir-attr-splat-float-value   ; splat float as double (NaN if absent)
    mlir-op-get-operand-segment-sizes  ; "operandSegmentSizes" as Scheme list
    ;; Pattern application
    mlir-apply-patterns-greedy    ; applyPatternsAndFoldGreedily
    ;; Generic op rebuild
    mlir-op-clone-with-types      ; clone op with new operands and result types
    ;; hip fusion C++ helpers
    hip-extract-splat-scale       ; extract float from hip.constant splat scale
    hip-build-init                ; build tensor.empty init from shape source
    hip-create-requantized-layout-op  ; clone layout op with quantized types
    ;; (mlir core logging)
    mlir-log-trace mlir-log-debug mlir-log-info
    mlir-log-warning mlir-log-error mlir-log-fatal
    ;; (mlir dialects shape)
    mlir-shape.shape-type mlir-shape.size-type mlir-shape.witness-type)

  (import (mlir core attribute)
          (mlir core operation)
          (mlir core value)
          (mlir core builder)
          (mlir core logging)
          (mlir dialects shape))

  ;; mlir-type-element-type, mlir-type-integer-width, mlir-type-is-unsigned,
  ;; mlir-op-get-float-attr, mlir-attr-is-splat, mlir-attr-splat-float-value,
  ;; mlir-op-get-operand-segment-sizes are re-exported from (mlir core attribute).
  ;; mlir-apply-patterns-greedy, mlir-op-clone-with-types,
  ;; hip-extract-splat-scale, hip-build-init, hip-create-requantized-layout-op
  ;; are re-exported from (mlir core builder).

) ;; end library (mlir core ir)
