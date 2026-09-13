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
;; Uses the dialect conversion framework with TypeConverter, matching the
;; C++ implementation exactly.
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
  ;; Delegates to C++ implementation that uses applyFullConversion with TypeConverter
  (define (run-pass module-op . args)
    (mlir-log-info "Starting ONNX to HipSR Conversion (Scheme)")

    ;; Apply dialect conversion using C++ helper
    ;; This does:
    ;; 1. Ensures HipsrDialect is loaded
    ;; 2. Creates TypeConverter (adds device memory space to tensors)
    ;; 3. Creates ConversionTarget (marks ONNX illegal, HipSR legal)
    ;; 4. Populates conversion patterns (Cast pattern)
    ;; 5. Calls applyFullConversion
    ;; 6. Post-processes (erases dead NoValue, rewires placeholders)
    (let ((success (mlir-apply-onnx-to-hipsr-conversion module-op)))
      (if (= success 1)
          (mlir-log-info "ONNX to HipSR Conversion (Scheme): Success")
          (begin
            (mlir-log-error "ONNX to HipSR Conversion (Scheme): FAILED")
            (error 'run-pass "Conversion failed")))))

) ;; end library (onnx-to-hipsr)
