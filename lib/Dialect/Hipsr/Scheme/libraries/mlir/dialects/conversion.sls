#!r6rs
;;===----------------------------------------------------------------------===;;
;;
;; Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
;; Licensed under the MIT License.
;;
;;===----------------------------------------------------------------------===;;
;;
;; (mlir dialects conversion) — MLIR Dialect Conversion Framework
;;
;; Mirrors mlir/Transforms/DialectConversion.h: TypeConverter, ConversionTarget,
;; RewritePatternSet, applyFullConversion, and Scheme-pattern registration.
;;
;; Also includes populate helpers for func/return (dialect-agnostic utilities)
;; and per-ONNX-op populate helpers (TODO: move to (mlir onnx patterns) when
;; that module is created).
;;
;;===----------------------------------------------------------------------===;;

(library (mlir dialects conversion)

  (export
    ;; TypeConverter lifecycle
    mlir-create-type-converter
    mlir-destroy-type-converter

    ;; TypeConverter configuration
    mlir-type-converter-add-conversion
    mlir-type-converter-is-legal-type
    mlir-type-converter-is-legal
    mlir-type-converter-is-signature-legal
    mlir-type-converter-add-device-memory-conversions

    ;; ConversionTarget lifecycle
    mlir-create-conversion-target
    mlir-destroy-conversion-target

    ;; ConversionTarget configuration — generic
    mlir-conversion-target-add-illegal-dialect
    mlir-conversion-target-add-legal-dialect
    mlir-conversion-target-add-legal-op
    mlir-conversion-target-add-dynamically-legal-op
    mlir-conversion-target-mark-unknown-ops-dynamically-legal
    mlir-conversion-target-add-legal-common-ops
    mlir-conversion-target-add-dynamically-legal-func

    ;; ConversionTarget configuration — dialect-specific
    ;; TODO: move add-illegal-onnx to (mlir onnx patterns)
    ;; TODO: move add-legal-hipsr, mark-unknown-ops-nested-legal to (mlir hipsr ir)
    mlir-conversion-target-add-illegal-onnx
    mlir-conversion-target-add-legal-hipsr
    mlir-conversion-target-mark-unknown-ops-nested-legal

    ;; RewritePatternSet lifecycle
    mlir-create-rewrite-pattern-set
    mlir-destroy-rewrite-pattern-set

    ;; Applying conversion
    mlir-apply-full-conversion

    ;; Scheme-defined pattern registration
    mlir-register-conversion-pattern

    ;; Populate helpers — generic MLIR utilities
    mlir-populate-func-type-conversion-pattern
    mlir-populate-return-conversion-patterns

    ;; Populate helpers — per ONNX op
    ;; TODO: move to (mlir onnx patterns) when that module is created
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
    mlir-populate-constant-conversion-patterns

    ;; RAII macros (require conversion lifecycle functions above)
    with-type-converter
    with-conversion-target
    with-rewrite-pattern-set)

  (import (chezscheme)
          (mlir ir))  ; for with-raii

  ;;===--------------------------------------------------------------------===;;
  ;; TypeConverter
  ;;===--------------------------------------------------------------------===;;

  (define mlir-create-type-converter
    (foreign-procedure "mlir_create_type_converter" () uptr))

  (define mlir-destroy-type-converter
    (foreign-procedure "mlir_destroy_type_converter" (uptr) void))

  ;;; Add a Scheme type-conversion callback: (lambda (type-uptr) -> type-uptr or #f)
  (define mlir-type-converter-add-conversion
    (foreign-procedure "mlir_type_converter_add_conversion" (uptr scheme-object) void))

  (define mlir-type-converter-is-legal-type
    (foreign-procedure "mlir_type_converter_is_legal_type" (uptr uptr) int))

  (define mlir-type-converter-is-legal
    (foreign-procedure "mlir_type_converter_is_legal" (uptr uptr) int))

  (define mlir-type-converter-is-signature-legal
    (foreign-procedure "mlir_type_converter_is_signature_legal" (uptr uptr) int))

  ;;; Registers identity + device-memory-space tensor conversions.
  (define mlir-type-converter-add-device-memory-conversions
    (foreign-procedure "mlir_type_converter_add_device_memory_conversions" (uptr) void))

  ;;===--------------------------------------------------------------------===;;
  ;; ConversionTarget
  ;;===--------------------------------------------------------------------===;;

  (define mlir-create-conversion-target
    (foreign-procedure "mlir_create_conversion_target" (uptr) uptr))

  (define mlir-destroy-conversion-target
    (foreign-procedure "mlir_destroy_conversion_target" (uptr) void))

  (define mlir-conversion-target-add-illegal-dialect
    (foreign-procedure "mlir_conversion_target_add_illegal_dialect" (uptr string) void))

  (define mlir-conversion-target-add-legal-dialect
    (foreign-procedure "mlir_conversion_target_add_legal_dialect" (uptr string) void))

  (define mlir-conversion-target-add-legal-op
    (foreign-procedure "mlir_conversion_target_add_legal_op" (uptr uptr string) void))

  (define mlir-conversion-target-add-dynamically-legal-op
    (foreign-procedure "mlir_conversion_target_add_dynamically_legal_op"
                       (uptr uptr string scheme-object) void))

  (define mlir-conversion-target-mark-unknown-ops-dynamically-legal
    (foreign-procedure "mlir_conversion_target_mark_unknown_ops_dynamically_legal"
                       (uptr scheme-object) void))

  (define mlir-conversion-target-add-legal-common-ops
    (foreign-procedure "mlir_conversion_target_add_legal_common_ops" (uptr) void))

  (define mlir-conversion-target-add-dynamically-legal-func
    (foreign-procedure "mlir_conversion_target_add_dynamically_legal_func"
                       (uptr uptr) void))

  ;; Dialect-specific convenience wrappers (to be relocated)
  (define mlir-conversion-target-add-illegal-onnx
    (foreign-procedure "mlir_conversion_target_add_illegal_onnx" (uptr) void))

  (define mlir-conversion-target-add-legal-hipsr
    (foreign-procedure "mlir_conversion_target_add_legal_hipsr" (uptr) void))

  (define mlir-conversion-target-mark-unknown-ops-nested-legal
    (foreign-procedure "mlir_conversion_target_mark_unknown_ops_nested_legal" (uptr) void))

  ;;===--------------------------------------------------------------------===;;
  ;; RewritePatternSet
  ;;===--------------------------------------------------------------------===;;

  (define mlir-create-rewrite-pattern-set
    (foreign-procedure "mlir_create_rewrite_pattern_set" (uptr) uptr))

  (define mlir-destroy-rewrite-pattern-set
    (foreign-procedure "mlir_destroy_rewrite_pattern_set" (uptr) void))

  (define mlir-apply-full-conversion
    (foreign-procedure "mlir_apply_full_conversion" (uptr uptr uptr) int))

  ;;===--------------------------------------------------------------------===;;
  ;; Pattern Registration
  ;;===--------------------------------------------------------------------===;;

  ;;; Register a Scheme-defined ConversionPattern.
  ;;; callback signature: (lambda (op operands-ref rewriter type-converter) #t/#f)
  (define mlir-register-conversion-pattern
    (foreign-procedure "mlir_register_conversion_pattern"
                       (uptr string scheme-object uptr) void))

  ;;===--------------------------------------------------------------------===;;
  ;; Populate Helpers — generic
  ;;===--------------------------------------------------------------------===;;

  (define mlir-populate-func-type-conversion-pattern
    (foreign-procedure "mlir_populate_func_type_conversion_pattern" (uptr uptr) void))

  (define mlir-populate-return-conversion-patterns
    (foreign-procedure "mlir_populate_return_conversion_patterns" (uptr uptr uptr) void))

  ;;===--------------------------------------------------------------------===;;
  ;; Populate Helpers — per ONNX op (TODO: move to (mlir onnx patterns))
  ;;===--------------------------------------------------------------------===;;

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

  ;;===--------------------------------------------------------------------===;;
  ;; RAII Macros
  ;;===--------------------------------------------------------------------===;;

  (define-syntax with-type-converter
    (syntax-rules ()
      [(_ (var) body ...)
       (with-raii ((var (mlir-create-type-converter) mlir-destroy-type-converter))
         body ...)]))

  (define-syntax with-conversion-target
    (syntax-rules ()
      [(_ (var ctx) body ...)
       (with-raii ((var (mlir-create-conversion-target ctx) mlir-destroy-conversion-target))
         body ...)]))

  (define-syntax with-rewrite-pattern-set
    (syntax-rules ()
      [(_ (var ctx) body ...)
       (with-raii ((var (mlir-create-rewrite-pattern-set ctx) mlir-destroy-rewrite-pattern-set))
         body ...)]))

) ;; end library (mlir dialects conversion)
