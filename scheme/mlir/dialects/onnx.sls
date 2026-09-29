#!r6rs
;;===----------------------------------------------------------------------===;;
;; Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
;; Licensed under the MIT License.
;;===----------------------------------------------------------------------===;;
;;
;; (mlir dialects onnx) — ONNX-op populate helpers for the onnx-to-hipsr pass.
;;
;;===----------------------------------------------------------------------===;;
(library (mlir dialects onnx)
  (export
    mlir-populate-cast-conversion-patterns
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

  (define mlir-populate-cast-conversion-patterns
    (foreign-procedure "mlir_populate_cast_conversion_patterns"    (uptr uptr uptr) void))
  (define mlir-populate-matmul-conversion-patterns
    (foreign-procedure "mlir_populate_matmul_conversion_patterns"  (uptr uptr uptr) void))
  (define mlir-populate-expand-conversion-patterns
    (foreign-procedure "mlir_populate_expand_conversion_patterns"  (uptr uptr uptr) void))
  (define mlir-populate-min-conversion-patterns
    (foreign-procedure "mlir_populate_min_conversion_patterns"     (uptr uptr uptr) void))
  (define mlir-populate-shape-conversion-patterns
    (foreign-procedure "mlir_populate_shape_conversion_patterns"   (uptr uptr uptr) void))
  (define mlir-populate-reshape-conversion-patterns
    (foreign-procedure "mlir_populate_reshape_conversion_patterns" (uptr uptr uptr) void))
  (define mlir-populate-unsqueeze-conversion-patterns
    (foreign-procedure "mlir_populate_unsqueeze_conversion_patterns" (uptr uptr uptr) void))
  (define mlir-populate-equal-conversion-patterns
    (foreign-procedure "mlir_populate_equal_conversion_patterns"   (uptr uptr uptr) void))
  (define mlir-populate-transpose-conversion-patterns
    (foreign-procedure "mlir_populate_transpose_conversion_patterns" (uptr uptr uptr) void))
  (define mlir-populate-gather-conversion-patterns
    (foreign-procedure "mlir_populate_gather_conversion_patterns"  (uptr uptr uptr) void))
  (define mlir-populate-slice-conversion-patterns
    (foreign-procedure "mlir_populate_slice_conversion_patterns"   (uptr uptr uptr) void))
  (define mlir-populate-scatter-nd-conversion-patterns
    (foreign-procedure "mlir_populate_scatter_nd_conversion_patterns" (uptr uptr uptr) void))
  (define mlir-populate-nonzero-conversion-patterns
    (foreign-procedure "mlir_populate_nonzero_conversion_patterns" (uptr uptr uptr) void))
  (define mlir-populate-constant-conversion-patterns
    (foreign-procedure "mlir_populate_constant_conversion_patterns" (uptr uptr uptr) void))

) ;; end library (mlir dialects onnx)
