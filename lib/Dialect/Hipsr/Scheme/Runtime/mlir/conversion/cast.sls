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
    run-pass
    )

  (import (rnrs (6))
          (mlir ffi)
          (mlir pattern-dsl)
          (rime loop))

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

  ;;===--------------------------------------------------------------------===;;
  ;; Pass Entry Point
  ;;===--------------------------------------------------------------------===;;

  ;; Entry point: Apply pattern to all operations in module
  (define (run-pass module-op . args)
    (mlir-log-info "Starting CastConversion Pass (Scheme)")
    (mlir-log-debug (string-append "Module: " (mlir-operation-name module-op)))

    (let ((match-count 0))
      ;; Walk all operations and apply pattern with rewriting enabled
      (mlir-operation-walk-rewrite module-op
        (lambda (op)
          (when (apply-pattern onnx-cast-pattern op)
            (set! match-count (+ match-count 1)))
          #f)) ; Return #f because apply-pattern handles the rewrite

      (mlir-log-info (string-append "CastConversion: "
                                    (number->string match-count)
                                    " patterns matched"))
      (mlir-log-info "Completed CastConversion Pass (Scheme)")))

) ;; end library (mlir conversion cast)
