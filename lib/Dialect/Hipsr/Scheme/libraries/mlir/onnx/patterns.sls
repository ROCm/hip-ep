#!r6rs
;;===----------------------------------------------------------------------===;;
;;
;; Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
;; Licensed under the MIT License.
;;
;;===----------------------------------------------------------------------===;;
;;
;; ONNX-to-HipSR conversion pattern population helpers.
;;
;; Mirrors: hip/Conversion/OnnxToHipsr/
;;
;; Each function populates the C++-defined conversion patterns for one ONNX op.
;; The Scheme DSL patterns (cast.sls, equal.sls, ...) are registered separately
;; in passes/onnx-to-hipsr.sls and do NOT appear here.
;;
;; All populate helpers share the same signature:
;;   (converter-ptr patterns-ptr context-ptr) → void
;;   where all three are uptr (TypeConverter*, RewritePatternSet*, MLIRContext*)
;;
;;===----------------------------------------------------------------------===;;

(library (mlir onnx patterns)
  (export
    ;; Conversion target helpers
    mlir-conversion-target-add-illegal-onnx

    ;; Per-op C++ populate helpers
    mlir-populate-matmul-conversion-patterns
    mlir-populate-expand-conversion-patterns
    mlir-populate-min-conversion-patterns
    mlir-populate-shape-conversion-patterns
    mlir-populate-reshape-conversion-patterns
    mlir-populate-unsqueeze-conversion-patterns
    mlir-populate-equal-conversion-patterns
    mlir-populate-transpose-conversion-patterns
    mlir-populate-gather-conversion-patterns
    mlir-populate-slice-conversion-patterns
    mlir-populate-scatter-nd-conversion-patterns
    mlir-populate-nonzero-conversion-patterns
    mlir-populate-constant-conversion-patterns)

  (import (chezscheme))

  ;;===--------------------------------------------------------------------===;;
  ;; Conversion target helpers
  ;;===--------------------------------------------------------------------===;;

  ;;; @brief Mark the OnnxDialect as illegal in a ConversionTarget, except
  ;;;        onnx.NoValue which is left legal (erased post-conversion).
  ;;; @param target-ptr ConversionTarget* as uptr
  (define mlir-conversion-target-add-illegal-onnx
    (foreign-procedure "mlir_conversion_target_add_illegal_onnx" (uptr) void))

  ;;===--------------------------------------------------------------------===;;
  ;; Per-op C++ populate helpers
  ;;
  ;; Each function registers the C++-defined ConversionPattern for its ONNX op
  ;; into an existing RewritePatternSet. The shape region for the resulting
  ;; hipsr.placeholder is filled by PopulateShapeRegionPass (not here).
  ;;===--------------------------------------------------------------------===;;

  (define mlir-populate-matmul-conversion-patterns
    (foreign-procedure "mlir_populate_matmul_conversion_patterns"     (uptr uptr uptr) void))

  (define mlir-populate-expand-conversion-patterns
    (foreign-procedure "mlir_populate_expand_conversion_patterns"     (uptr uptr uptr) void))

  (define mlir-populate-min-conversion-patterns
    (foreign-procedure "mlir_populate_min_conversion_patterns"        (uptr uptr uptr) void))

  (define mlir-populate-shape-conversion-patterns
    (foreign-procedure "mlir_populate_shape_conversion_patterns"      (uptr uptr uptr) void))

  (define mlir-populate-reshape-conversion-patterns
    (foreign-procedure "mlir_populate_reshape_conversion_patterns"    (uptr uptr uptr) void))

  (define mlir-populate-unsqueeze-conversion-patterns
    (foreign-procedure "mlir_populate_unsqueeze_conversion_patterns"  (uptr uptr uptr) void))

  (define mlir-populate-equal-conversion-patterns
    (foreign-procedure "mlir_populate_equal_conversion_patterns"      (uptr uptr uptr) void))

  (define mlir-populate-transpose-conversion-patterns
    (foreign-procedure "mlir_populate_transpose_conversion_patterns"  (uptr uptr uptr) void))

  (define mlir-populate-gather-conversion-patterns
    (foreign-procedure "mlir_populate_gather_conversion_patterns"     (uptr uptr uptr) void))

  (define mlir-populate-slice-conversion-patterns
    (foreign-procedure "mlir_populate_slice_conversion_patterns"      (uptr uptr uptr) void))

  (define mlir-populate-scatter-nd-conversion-patterns
    (foreign-procedure "mlir_populate_scatter_nd_conversion_patterns" (uptr uptr uptr) void))

  (define mlir-populate-nonzero-conversion-patterns
    (foreign-procedure "mlir_populate_nonzero_conversion_patterns"    (uptr uptr uptr) void))

  (define mlir-populate-constant-conversion-patterns
    (foreign-procedure "mlir_populate_constant_conversion_patterns"   (uptr uptr uptr) void))

) ;; end library (mlir onnx patterns)
