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
    mlir-shape.shape-type
    mlir-shape.size-type
    mlir-shape.witness-type)
  (import (chezscheme))

  (define mlir-shape.shape-type
    (foreign-procedure "mlir_get_shape_shape_type" (uptr) uptr))
  (define mlir-shape.size-type
    (foreign-procedure "mlir_get_shape_size_type" (uptr) uptr))
  (define mlir-shape.witness-type
    (foreign-procedure "mlir_get_shape_witness_type" (uptr) uptr))

) ;; end library (mlir dialects shape)
