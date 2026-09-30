#!r6rs
;;===----------------------------------------------------------------------===;;
;;
;; Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
;; Licensed under the MIT License.
;;
;;===----------------------------------------------------------------------===;;
;;
;; (mlir core value) — MLIR Value and ValueArrayRef primitives.
;;
;; Mirrors mlir/IR/Value.h.
;;
;;===----------------------------------------------------------------------===;;

(library (mlir core value)
  (export
    mlir-value-get-defining-op
    mlir-value-get-type
    mlir-value-is-block-argument?
    mlir-value-get-result-number
    value-array-ref-size
    value-array-ref-at)

  (import (rnrs)
          (only (chezscheme) foreign-procedure define-ftype ftype-ref
                make-ftype-pointer foreign-ref))

  (define-ftype ValueArrayRef
    (struct [data uptr] [size uptr]))

  (define mlir-value-get-defining-op
    (foreign-procedure "mlir_value_get_defining_op" (uptr) uptr))
  (define mlir-value-get-type
    (foreign-procedure "mlir_value_get_type" (uptr) uptr))
  (define %mlir-value-is-block-argument
    (foreign-procedure "mlir_value_is_block_argument" (uptr) int))
  (define (mlir-value-is-block-argument? v)
    (= 1 (%mlir-value-is-block-argument v)))
  (define mlir-value-get-result-number
    (foreign-procedure "mlir_value_get_result_number" (uptr) int))

  (define (value-array-ref-size ref-ptr)
    (ftype-ref ValueArrayRef (size) (make-ftype-pointer ValueArrayRef ref-ptr)))
  (define (value-array-ref-at ref-ptr index)
    (let* ([ptr      (make-ftype-pointer ValueArrayRef ref-ptr)]
           [data-ptr (ftype-ref ValueArrayRef (data) ptr)])
      (foreign-ref 'uptr data-ptr (* index 8))))

) ;; end library (mlir core value)
