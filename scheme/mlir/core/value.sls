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

  ;; C struct layout for the ValueArrayRef passed by the C++ callback ABI.
  ;; data: pointer to the first Value* element
  ;; size: number of elements
  (define-ftype ValueArrayRef
    (struct [data uptr] [size uptr]))

  ;; Return the operation that defines this value, or 0 for block arguments.
  ;; value: Value* opaque ptr uptr
  ;; Returns: Operation* uptr, or 0 if the value is a block argument
  (define mlir-value-get-defining-op
    (foreign-procedure "mlir_value_get_defining_op" (uptr) uptr))

  ;; Return the MLIR type of the value as an opaque Type pointer.
  ;; value: Value* opaque ptr uptr
  ;; Returns: Type opaque ptr uptr
  (define mlir-value-get-type
    (foreign-procedure "mlir_value_get_type" (uptr) uptr))

  ;; Raw predicate: returns 1 if the value is a block argument, 0 otherwise.
  ;; value: Value* opaque ptr uptr
  (define %mlir-value-is-block-argument
    (foreign-procedure "mlir_value_is_block_argument" (uptr) int))

  ;; Return #t if the value is a block argument (not an op result).
  ;; value: Value* opaque ptr uptr
  (define (mlir-value-is-block-argument? v)
    (= 1 (%mlir-value-is-block-argument v)))

  ;; Return the 0-based result number of this op-result value.
  ;; value: Value* opaque ptr uptr (must be an op result, not a block argument)
  ;; Returns: result index as int
  (define mlir-value-get-result-number
    (foreign-procedure "mlir_value_get_result_number" (uptr) int))

  ;; Return the number of Value* elements in a ValueArrayRef.
  ;; ref-ptr: uptr pointing to a ValueArrayRef struct in C memory
  (define (value-array-ref-size ref-ptr)
    (ftype-ref ValueArrayRef (size) (make-ftype-pointer ValueArrayRef ref-ptr)))

  ;; Return the i-th Value* opaque ptr from a ValueArrayRef.
  ;; ref-ptr: uptr pointing to a ValueArrayRef struct in C memory
  ;; index:   0-based element index
  ;; Returns: Value* opaque ptr uptr
  (define (value-array-ref-at ref-ptr index)
    (let* ([ptr      (make-ftype-pointer ValueArrayRef ref-ptr)]
           [data-ptr (ftype-ref ValueArrayRef (data) ptr)])
      (foreign-ref 'uptr data-ptr (* index 8))))

) ;; end library (mlir core value)
