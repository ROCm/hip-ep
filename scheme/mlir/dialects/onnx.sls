#!r6rs
;;===----------------------------------------------------------------------===;;
;; Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
;; Licensed under the MIT License.
;;===----------------------------------------------------------------------===;;
;;
;; (mlir dialects onnx) — ONNX-op populate helpers for the onnx-to-hipsr pass.
;;
;; Each function registers C++ conversion patterns for one ONNX op into
;; a RewritePatternSet.  All share the same signature:
;;   type-converter: TypeConverter* uptr
;;   patterns:       RewritePatternSet* uptr — patterns added in-place
;;   ctx:            MLIRContext* uptr
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

  ;; Register C++ patterns for onnx.Cast → hipsr.cast.
  (define mlir-populate-cast-conversion-patterns
    (foreign-procedure "mlir_populate_cast_conversion_patterns"    (uptr uptr uptr) void))

  ;; Register C++ patterns for onnx.MatMul → hipsr.matmul.
  (define mlir-populate-matmul-conversion-patterns
    (foreign-procedure "mlir_populate_matmul_conversion_patterns"  (uptr uptr uptr) void))

  ;; Register C++ patterns for onnx.Expand → hipsr.expand (host-shape path).
  (define mlir-populate-expand-conversion-patterns
    (foreign-procedure "mlir_populate_expand_conversion_patterns"  (uptr uptr uptr) void))

  ;; Register C++ patterns for onnx.Min → hipsr.min.
  (define mlir-populate-min-conversion-patterns
    (foreign-procedure "mlir_populate_min_conversion_patterns"     (uptr uptr uptr) void))

  ;; Register C++ patterns for onnx.Shape → hipsr.compute (host output).
  (define mlir-populate-shape-conversion-patterns
    (foreign-procedure "mlir_populate_shape_conversion_patterns"   (uptr uptr uptr) void))

  ;; Register C++ patterns for onnx.Reshape → hipsr.compute (reshape body).
  (define mlir-populate-reshape-conversion-patterns
    (foreign-procedure "mlir_populate_reshape_conversion_patterns" (uptr uptr uptr) void))

  ;; Register C++ patterns for onnx.Unsqueeze → hipsr.compute.
  (define mlir-populate-unsqueeze-conversion-patterns
    (foreign-procedure "mlir_populate_unsqueeze_conversion_patterns" (uptr uptr uptr) void))

  ;; Register C++ patterns for onnx.Equal → hipsr.equal.
  (define mlir-populate-equal-conversion-patterns
    (foreign-procedure "mlir_populate_equal_conversion_patterns"   (uptr uptr uptr) void))

  ;; Register C++ patterns for onnx.Transpose → hipsr.placeholder + hipsr.transpose.
  (define mlir-populate-transpose-conversion-patterns
    (foreign-procedure "mlir_populate_transpose_conversion_patterns" (uptr uptr uptr) void))

  ;; Register C++ patterns for onnx.Gather → hipsr.gather (host-data fallback).
  ;; The Scheme DDR pattern handles device data; this C++ pattern handles host data.
  (define mlir-populate-gather-conversion-patterns
    (foreign-procedure "mlir_populate_gather_conversion_patterns"  (uptr uptr uptr) void))

  ;; Register C++ patterns for onnx.Slice → hipsr.compute (window-operand path).
  (define mlir-populate-slice-conversion-patterns
    (foreign-procedure "mlir_populate_slice_conversion_patterns"   (uptr uptr uptr) void))

  ;; Register C++ patterns for onnx.ScatterND → hipsr.scatter_nd.
  (define mlir-populate-scatter-nd-conversion-patterns
    (foreign-procedure "mlir_populate_scatter_nd_conversion_patterns" (uptr uptr uptr) void))

  ;; Register C++ patterns for onnx.NonZero → hipsr.compute.
  (define mlir-populate-nonzero-conversion-patterns
    (foreign-procedure "mlir_populate_nonzero_conversion_patterns" (uptr uptr uptr) void))

  ;; Register C++ fallback patterns for onnx.Constant → hipsr.constant.
  ;; Handles external data (location/offset/size) that the Scheme DDR pattern defers.
  (define mlir-populate-constant-conversion-patterns
    (foreign-procedure "mlir_populate_constant_conversion_patterns" (uptr uptr uptr) void))

) ;; end library (mlir dialects onnx)
