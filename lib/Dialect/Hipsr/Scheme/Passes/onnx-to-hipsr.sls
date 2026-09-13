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
;;
;; Orchestrates the dialect conversion using MLIR framework primitives.
;; The C++ FFI layer provides primitives and reusable helpers, while this
;; Scheme code implements the high-level conversion logic.
;;===----------------------------------------------------------------------===;;

(library (onnx-to-hipsr)
  (export run-pass)
  (import (rnrs (6))
          (only (chezscheme) format)
          (mlir ffi))

  ;;===--------------------------------------------------------------------===;;
  ;; Pass Entry Point
  ;;===--------------------------------------------------------------------===;;

  ;; Main conversion pass
  ;; Orchestrates dialect conversion using MLIR primitives
  ;;
  ;; NOTE: Required dialects (HipsrDialect, OnnxDialect, FuncDialect) are
  ;; loaded automatically by MLIR's pass infrastructure via the dependentDialects
  ;; declaration in Passes.td. No need to check them here.
  (define (run-pass module-op . args)
    (mlir-log-info "Starting ONNX to HipSR Conversion (Scheme)")

    ;; Step 1: Apply dialect conversion
    ;; NOTE: TypeConverter, ConversionTarget, and applyFullConversion
    ;; are complex C++ framework objects that cannot be easily exposed via FFI.
    ;; They are handled by C++ helpers that follow the standard MLIR patterns.
    ;;
    ;; Future work: If we need Scheme to customize TypeConverter or
    ;; ConversionTarget, we would need to add FFI for those specific
    ;; customization points (e.g., callback registration).
    (mlir-log-debug "Applying dialect conversion...")

    ;; This helper creates TypeConverter, ConversionTarget, patterns,
    ;; and calls applyFullConversion - standard MLIR dialect conversion
    ;; The helper is in C++ because these framework objects are not
    ;; easily manipulated through FFI
    (let ((success (mlir-apply-dialect-conversion-onnx-to-hipsr module-op)))
      (if (= success 1)
          (begin
            (mlir-log-debug "Dialect conversion successful")

            ;; Step 2: Post-processing using primitives
            (mlir-log-debug "Erasing dead NoValue ops...")
            (mlir-erase-dead-novalue-ops module-op)

            (mlir-log-debug "Rewiring placeholder inputs...")
            (mlir-rewire-placeholder-inputs module-op)

            (mlir-log-info "ONNX to HipSR Conversion (Scheme): Success"))
          (begin
            (mlir-log-error "ONNX to HipSR Conversion (Scheme): FAILED")
            (error 'run-pass "Dialect conversion failed")))))

) ;; end library (onnx-to-hipsr)
