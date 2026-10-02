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

  ;; Return the !shape.shape type for the given MLIRContext.
  ;; ctx: MLIRContext* uptr
  ;; Returns: shape::ShapeType uptr (opaque type pointer)
  (define mlir-shape.shape-type
    (foreign-procedure "mlir_get_shape_shape_type" (uptr) uptr))

  ;; Return the !shape.size type for the given MLIRContext.
  ;; ctx: MLIRContext* uptr
  ;; Returns: shape::SizeType uptr (opaque type pointer)
  (define mlir-shape.size-type
    (foreign-procedure "mlir_get_shape_size_type" (uptr) uptr))

  ;; Return the !shape.witness type for the given MLIRContext.
  ;; ctx: MLIRContext* uptr
  ;; Returns: shape::WitnessType uptr (opaque type pointer)
  (define mlir-shape.witness-type
    (foreign-procedure "mlir_get_shape_witness_type" (uptr) uptr))

) ;; end library (mlir dialects shape)
