;;===----------------------------------------------------------------------===;;
;;
;; Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
;; Licensed under the MIT License.
;;
;;===----------------------------------------------------------------------===;;
;;
;; CastConversion - Convert ONNX Cast operations to HipSR dialect (Scheme)
;;
;; This demonstrates pattern matching and rewriting MLIR operations using
;; the Pattern DSL. Port of lib/Conversion/OnnxToHipsr/CastConversion.cpp
;;
;; Pattern: onnx.Cast(input) -> hipsr.placeholder(ctx, input) + hipsr.cast(ctx, input, init)
;;
;;===----------------------------------------------------------------------===;;

;;===----------------------------------------------------------------------===;;
;; Imports
;;===----------------------------------------------------------------------===;;

(import (rime loop))
;; PatternDSL is loaded during Scheme runtime initialization

;;===----------------------------------------------------------------------===;;
;; Pattern Definition
;;===----------------------------------------------------------------------===;;

;; Define the onnx.Cast -> hipsr.cast conversion pattern
(define onnx-cast-pattern
  (define-conversion-pattern "onnx.Cast"
    ;; Constraints: must have 1 input, 1 output, output must be ranked tensor
    (list (has-n-operands 1)
          (has-n-results 1)
          (result-0-is-ranked-tensor))
    ;; Rewrite action
    rewrite-with-placeholder-and-cast))

;;===----------------------------------------------------------------------===;;
;; Public API
;;===----------------------------------------------------------------------===;;

;; Entry point: Apply pattern to all operations in module
(define (run-pass module-op . args)
  (mlir-log-info "Starting CastConversion Pass (Scheme)")
  (mlir-log-debug (format "Module: ~a" (mlir-operation-name module-op)))

  (let ((match-count 0))
    ;; Walk all operations and apply pattern
    (mlir-operation-walk module-op
      (lambda (op)
        (when (apply-pattern onnx-cast-pattern op)
          (set! match-count (+ match-count 1)))))

    (mlir-log-info (format "CastConversion: ~a patterns matched" match-count))
    (mlir-log-info "Completed CastConversion Pass (Scheme)")))
