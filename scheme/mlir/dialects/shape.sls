#!r6rs
;;===----------------------------------------------------------------------===;;
;;
;; Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
;; Licensed under the MIT License.
;;
;;===----------------------------------------------------------------------===;;
;;
;; (mlir dialects shape) — Shape dialect type accessors.
;;
;; Mirrors lib/Scheme/Bindings/Dialects/Shape.cpp.
;;
;;===----------------------------------------------------------------------===;;

(library (mlir dialects shape)
  (export
    mlir-get-shape-shape-type
    mlir-get-shape-size-type
    mlir-get-shape-witness-type)
  (import (chezscheme))

  (define mlir-get-shape-shape-type
    (foreign-procedure "mlir_get_shape_shape_type" (uptr) uptr))
  (define mlir-get-shape-size-type
    (foreign-procedure "mlir_get_shape_size_type" (uptr) uptr))
  (define mlir-get-shape-witness-type
    (foreign-procedure "mlir_get_shape_witness_type" (uptr) uptr))

) ;; end library (mlir dialects shape)
