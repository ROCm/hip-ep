#!r6rs
;;===----------------------------------------------------------------------===;;
;;
;; Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
;; Licensed under the MIT License.
;;
;;===----------------------------------------------------------------------===;;
;;
;; (mlir core attribute) — MLIR attribute construction.
;;
;; Mirrors mlir/IR/Attribute.h. Attributes are first-class opaque uptr values
;; (Attribute::getAsOpaquePointer / getFromOpaquePointer).
;;
;;   (make-mlir-attribute ctx type value)
;;     ctx   : MLIRContext* uptr
;;     type  : :i64 | :index | :i32-array | :i64-array
;;     value : integer or Scheme list (for array types)
;;
;;===----------------------------------------------------------------------===;;

(library (mlir core attribute)
  (export make-mlir-attribute)

  (import (rnrs)
          (only (chezscheme) foreign-procedure))

  (define %make-i64
    (foreign-procedure "mlir_make_attr_i64"       (uptr integer-64)    uptr))
  (define %make-index
    (foreign-procedure "mlir_make_attr_index"     (uptr integer-64)    uptr))
  (define %make-i32-array
    (foreign-procedure "mlir_make_attr_i32_array" (uptr scheme-object) uptr))
  (define %make-i64-array
    (foreign-procedure "mlir_make_attr_i64_array" (uptr scheme-object) uptr))

  (define (make-mlir-attribute ctx type value)
    (case type
      [(:i64)       (%make-i64       ctx value)]
      [(:index)     (%make-index     ctx value)]
      [(:i32-array) (%make-i32-array ctx value)]
      [(:i64-array) (%make-i64-array ctx value)]
      [else (error 'make-mlir-attribute "unknown attr type" type)]))

) ;; end library (mlir core attribute)
