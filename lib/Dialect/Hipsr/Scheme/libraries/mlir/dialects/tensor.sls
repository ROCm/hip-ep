#!r6rs
;;===----------------------------------------------------------------------===;;
;;
;; Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
;; Licensed under the MIT License.
;;
;;===----------------------------------------------------------------------===;;
;;
;; Tensor dialect FFI — dialect-agnostic tensor type queries.
;;
;; Mirrors: mlir/Dialect/Tensor/IR/Tensor.h (type system queries only)
;;
;;===----------------------------------------------------------------------===;;

(library (mlir dialects tensor)
  (export
    mlir-type-is-ranked-tensor
    mlir-type-get-element-type
    mlir-type-get-shape
    mlir-type-get-rank
    mlir-type-get-encoding
    mlir-value-get-type)

  (import (chezscheme))

  ;;; @brief Check if a type is a ranked tensor type
  ;;; @param type-ptr Type* as uptr
  ;;; @return 1 if ranked tensor, 0 otherwise
  (define mlir-type-is-ranked-tensor
    (foreign-procedure "mlir_type_is_ranked_tensor" (uptr) int))

  ;;; @brief Get the element type of a shaped type (tensor, memref, etc.)
  ;;; @param type-ptr Type* as uptr
  ;;; @return Element Type* as uptr
  (define mlir-type-get-element-type
    (foreign-procedure "mlir_type_get_element_type" (uptr) uptr))

  ;;; @brief Get the shape dimensions of a shaped type
  ;;; @param type-ptr Type* as uptr
  ;;; @return Scheme list of integers representing shape (e.g., '(1 3 224 224))
  ;;; @note Returns a Scheme object (list), NOT an uptr pointer
  (define mlir-type-get-shape
    (foreign-procedure "mlir_type_get_shape" (uptr) scheme-object))

  ;;; @brief Get the rank (number of dimensions) of a shaped type
  ;;; @param type-ptr Type* as uptr
  ;;; @return Rank as int
  (define mlir-type-get-rank
    (foreign-procedure "mlir_type_get_rank" (uptr) int))

  ;;; @brief Get the encoding attribute of a RankedTensorType (0 if none)
  ;;; @param type-ptr Type* as uptr
  ;;; @return Attribute* as uptr, or 0 if not a ranked tensor or has no encoding
  (define mlir-type-get-encoding
    (foreign-procedure "mlir_type_get_encoding" (uptr) uptr))

  ;;; @brief Get the type of a value
  ;;; @param value-ptr Value* as uptr
  ;;; @return Type* as uptr
  (define mlir-value-get-type
    (foreign-procedure "mlir_value_get_type" (uptr) uptr))

) ;; end library (mlir dialects tensor)
