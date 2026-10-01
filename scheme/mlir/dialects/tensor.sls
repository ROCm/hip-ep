#!r6rs
;;===----------------------------------------------------------------------===;;
;;
;; Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
;; Licensed under the MIT License.
;;
;;===----------------------------------------------------------------------===;;
;;
;; (mlir dialects tensor) — Generic tensor type operations.
;;
;; Mirrors mlir/Dialect/Tensor/IR/Tensor.h for type-level operations.
;; These are dialect-agnostic: they operate on RankedTensorType and
;; accept any MLIR attribute as an encoding.
;;
;;===----------------------------------------------------------------------===;;

(library (mlir dialects tensor)
  (export mlir-tensor-type-with-encoding)

  (import (rnrs)
          (only (chezscheme) foreign-procedure))

  ;; Attach any MLIR attribute as the encoding of a RankedTensorType.
  ;; Returns a new type with the encoding set; the original is unchanged.
  ;; Returns 0 if the input type is not a RankedTensorType.
  (define mlir-tensor-type-with-encoding
    (foreign-procedure "mlir_tensor_type_with_encoding" (uptr uptr) uptr))

) ;; end library (mlir dialects tensor)
