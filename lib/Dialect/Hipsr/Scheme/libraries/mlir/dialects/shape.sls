#!r6rs
;;===----------------------------------------------------------------------===;;
;;
;; Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
;; Licensed under the MIT License.
;;
;;===----------------------------------------------------------------------===;;
;;
;; Shape dialect FFI — type getters for shape::ShapeType and shape::SizeType.
;;
;; Mirrors: mlir/Dialect/Shape/IR/Shape.h
;;
;;===----------------------------------------------------------------------===;;

(library (mlir dialects shape)
  (export
    mlir-get-shape-shape-type
    mlir-get-shape-size-type)

  (import (chezscheme))

  ;;; @brief Get the shape::ShapeType from an MLIRContext.
  ;;; @param ctx-ptr MLIRContext* as uptr
  ;;; @return Type* for !shape.shape as uptr
  (define mlir-get-shape-shape-type
    (foreign-procedure "mlir_get_shape_shape_type" (uptr) uptr))

  ;;; @brief Get the shape::SizeType from an MLIRContext.
  ;;; @param ctx-ptr MLIRContext* as uptr
  ;;; @return Type* for !shape.size as uptr
  (define mlir-get-shape-size-type
    (foreign-procedure "mlir_get_shape_size_type" (uptr) uptr))

) ;; end library (mlir dialects shape)
