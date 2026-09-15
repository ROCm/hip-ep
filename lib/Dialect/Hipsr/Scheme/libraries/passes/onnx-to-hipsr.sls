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

(library (passes onnx-to-hipsr)
  (export run-pass)
  (import (rnrs (6))
          (mlir ffi)
          (patterns cast))

  ;; Helper: Apply conversion and post-process
  (define (do-conversion module-op ctx converter target patterns)
    (mlir-log-debug "Applying full conversion...")
    (let ((success (mlir-apply-full-conversion module-op target patterns)))
      (mlir-destroy-rewrite-pattern-set patterns)
      (mlir-destroy-conversion-target target)
      (mlir-destroy-type-converter converter)
      (if (= success 1)
          (begin
            (mlir-log-debug "Dialect conversion successful")
            (mlir-log-debug "Erasing dead NoValue ops...")
            (mlir-erase-dead-novalue-ops module-op)
            (mlir-log-debug "Rewiring placeholder inputs...")
            (mlir-rewire-placeholder-inputs module-op)
            (mlir-log-info "ONNX to HipSR Conversion (Scheme): Success"))
          (begin
            (mlir-log-error "ONNX to HipSR Conversion (Scheme): FAILED")
            (error (quote run-pass) "Dialect conversion failed")))))

  (define (run-pass module-op . args)
    (mlir-log-info "Starting ONNX to HipSR Conversion (Scheme)")
    (let ((ctx (mlir-operation-get-context module-op)))
      (mlir-log-debug "Creating TypeConverter...")
      (let ((converter (mlir-create-type-converter)))
        (mlir-type-converter-add-device-memory-conversions converter)
        (mlir-log-debug "Creating ConversionTarget...")
        (let ((target (mlir-create-conversion-target ctx)))
          (mlir-conversion-target-add-illegal-onnx target)
          (mlir-conversion-target-add-legal-hipsr target)
          (mlir-conversion-target-add-legal-common-ops target)
          (mlir-conversion-target-add-dynamically-legal-func target converter)
          (mlir-conversion-target-mark-unknown-ops-nested-legal target)
          (mlir-log-debug "Populating conversion patterns...")
          (let ((patterns (mlir-create-rewrite-pattern-set ctx)))
            (populate-cast-patterns converter patterns ctx)
            (mlir-populate-return-conversion-patterns converter patterns ctx)
            (mlir-populate-func-type-conversion-pattern patterns converter)
            (do-conversion module-op ctx converter target patterns))))))

) ;; end library (passes onnx-to-hipsr)
