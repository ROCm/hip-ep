#!r6rs
;;===----------------------------------------------------------------------===;;
;;
;; Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
;; Licensed under the MIT License.
;;
;;===----------------------------------------------------------------------===;;
;;
;; CastConversion - Convert ONNX Cast operations to HipSR dialect
;;
;; This demonstrates pattern matching and rewriting MLIR operations using
;; the Pattern DSL. Port of lib/Conversion/OnnxToHipsr/CastConversion.cpp
;;
;; Pattern: onnx.Cast(input) -> hipsr.placeholder(ctx, input) + hipsr.cast(ctx, input, init)
;;
;;===----------------------------------------------------------------------===;;

(library (mlir conversion cast)
  (export
    onnx-cast-pattern
    )

  (import (rnrs (6))
          (mlir ffi)
          (mlir pattern-dsl)
          (for (rime loop) expand))  ; Import only at compile time (expand phase)

  ;;===--------------------------------------------------------------------===;;
  ;; Pattern Definition
  ;;===--------------------------------------------------------------------===;;

  ;; Define the onnx.Cast -> hipsr.cast conversion pattern
  (define onnx-cast-pattern
    (define-conversion-pattern "onnx.Cast"
      ;; Constraints: must have 1 input, 1 output, output must be ranked tensor
      (list (has-n-operands 1)
            (has-n-results 1)
            (result-0-is-ranked-tensor))
      ;; Rewrite action
      rewrite-with-placeholder-and-cast))

) ;; end library (mlir conversion cast)
