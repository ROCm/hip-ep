#!r6rs
;;===----------------------------------------------------------------------===;;
;;
;; Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
;; Licensed under the MIT License.
;;
;;===----------------------------------------------------------------------===;;
;;
;; ONNX to HipSR Conversion - Scheme Implementation
;;
;; Equivalent to lib/Conversion/OnnxToHipsr/OnnxToHipsr.cpp
;; This demonstrates writing a complete dialect conversion pass in Scheme.
;;
;; The pass:
;; 1. Walks all operations in the module
;; 2. Applies conversion patterns (currently just onnx.Cast)
;; 3. Rewrites matched operations to HipSR dialect
;;
;; NOTE: This is a simplified version using greedy rewriting.
;; The C++ version uses dialect conversion framework with TypeConverter.
;; To match C++ exactly, we'd need:
;;   - TypeConverter integration (add device memory space to function signatures)
;;   - ConversionTarget (mark ONNX illegal, HipSR legal)
;;   - applyFullConversion with pattern infrastructure
;;
;; This Scheme version demonstrates the pattern matching and rewriting logic.
;;===----------------------------------------------------------------------===;;

(library (onnx-to-hipsr)
  (export run-pass)
  (import (rnrs (6))
          (only (chezscheme) format)
          (mlir ffi)
          (mlir pattern-dsl)
          (mlir conversion cast))

  ;;===--------------------------------------------------------------------===;;
  ;; Pattern Registry
  ;;===--------------------------------------------------------------------===;;

  ;; List of all ONNX -> HipSR conversion patterns
  ;; Currently only Cast is implemented
  (define onnx-to-hipsr-patterns
    (list onnx-cast-pattern))

  ;;===--------------------------------------------------------------------===;;
  ;; Pass Entry Point
  ;;===--------------------------------------------------------------------===;;

  ;; Main conversion pass
  ;; Walks all operations and applies conversion patterns
  (define (run-pass module-op . args)
    (mlir-log-info "Starting ONNX to HipSR Conversion (Scheme)")

    ;; Step 1: Convert function signatures
    ;;(mlir-operation-walk module-op
    ;;  (lambda (op)
    ;;    (when (string=? (mlir-operation-name op) "func.func")
    ;;      (mlir-func-convert-signature op))))

    ;; Step 2: Apply conversion patterns
    (let ((total-ops 0)
          (converted-ops 0))

      (mlir-operation-walk-rewrite module-op
        (lambda (op)
          (set! total-ops (+ total-ops 1))

          (when (apply-patterns onnx-to-hipsr-patterns op)
            (set! converted-ops (+ converted-ops 1)))

          #f))

      (mlir-log-info (format "ONNX to HipSR Conversion (Scheme): ~a/~a operations converted"
                             converted-ops total-ops)))

    (mlir-log-info "Completed ONNX to HipSR Conversion (Scheme)"))

  ;;===--------------------------------------------------------------------===;;
  ;; Future Work
  ;;===--------------------------------------------------------------------===;;

  ;; To match C++ OnnxToHipsr.cpp exactly, we need to expose:
  ;;
  ;; 1. TypeConverter API:
  ;;    - mlir-type-converter-create
  ;;    - mlir-type-converter-add-conversion
  ;;    - mlir-type-converter-is-legal
  ;;
  ;; 2. ConversionTarget API:
  ;;    - mlir-conversion-target-create
  ;;    - mlir-conversion-target-add-illegal-dialect
  ;;    - mlir-conversion-target-add-legal-dialect
  ;;    - mlir-conversion-target-add-dynamically-legal-op
  ;;
  ;; 3. Pattern Application:
  ;;    - mlir-apply-full-conversion
  ;;    - mlir-apply-partial-conversion
  ;;
  ;; 4. Post-processing:
  ;;    - mlir-erase-dead-onnx-no-value
  ;;    - mlir-rewire-placeholder-inputs
  ;;
  ;; For now, this greedy rewriting approach works for demonstration.

) ;; end library (onnx-to-hipsr)
