#!r6rs
;;===----------------------------------------------------------------------===;;
;;
;; Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
;; Licensed under the MIT License.
;;
;;===----------------------------------------------------------------------===;;
;;
;; (mlir core conversion) — MLIR Dialect Conversion Framework
;;
;; Mirrors mlir/Transforms/DialectConversion.h: TypeConverter, ConversionTarget,
;; RewritePatternSet, applyFullConversion, and Scheme-pattern registration.
;;
;; Dialect-specific populate helpers live in:
;;   (mlir dialects onnx)  — per-ONNX-op populate helpers
;;   (mlir dialects func)  — func/return populate helpers
;;
;;===----------------------------------------------------------------------===;;

(library (mlir core conversion)

  (export
    ;; TypeConverter lifecycle
    mlir-create-type-converter
    mlir-destroy-type-converter

    ;; TypeConverter configuration
    mlir-type-converter-add-conversion
    mlir-type-converter-add-tensor-widening-materialization
    mlir-type-converter-is-legal-type
    mlir-type-converter-is-legal
    mlir-type-converter-is-signature-legal

    ;; ConversionTarget lifecycle
    mlir-create-conversion-target
    mlir-destroy-conversion-target

    ;; ConversionTarget configuration — generic
    mlir-conversion-target-add-illegal-dialect
    mlir-conversion-target-add-legal-dialect
    mlir-conversion-target-add-legal-op
    mlir-conversion-target-add-dynamically-legal-op
    mlir-conversion-target-mark-unknown-ops-dynamically-legal

    ;; RewritePatternSet lifecycle
    mlir-create-rewrite-pattern-set
    mlir-destroy-rewrite-pattern-set

    ;; Applying conversion
    mlir-apply-full-conversion

    ;; Scheme-defined pattern registration
    mlir-register-conversion-pattern

    ;; RAII macros (require conversion lifecycle functions above)
    with-type-converter
    with-conversion-target
    with-rewrite-pattern-set)

  (import (chezscheme)
          (mlir core ir))  ; for with-raii

  ;;===--------------------------------------------------------------------===;;
  ;; TypeConverter
  ;;===--------------------------------------------------------------------===;;

  (define mlir-create-type-converter
    (foreign-procedure "mlir_create_type_converter" () uptr))

  (define mlir-destroy-type-converter
    (foreign-procedure "mlir_destroy_type_converter" (uptr) void))

  (define mlir-type-converter-add-tensor-widening-materialization
    (foreign-procedure "mlir_type_converter_add_tensor_widening_materialization" (uptr) void))

  ;;; Add a Scheme type-conversion callback: (lambda (type-uptr) -> type-uptr or #f)
  (define mlir-type-converter-add-conversion
    (foreign-procedure "mlir_type_converter_add_conversion" (uptr scheme-object) void))

  (define mlir-type-converter-is-legal-type
    (foreign-procedure "mlir_type_converter_is_legal_type" (uptr uptr) int))

  (define mlir-type-converter-is-legal
    (foreign-procedure "mlir_type_converter_is_legal" (uptr uptr) int))

  (define mlir-type-converter-is-signature-legal
    (foreign-procedure "mlir_type_converter_is_signature_legal" (uptr uptr) int))

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
  ;; RAII Macros
  ;;===--------------------------------------------------------------------===;;

  (define-syntax with-type-converter
    (syntax-rules ()
      [(_ (var) body ...)
       (with-raii (var (mlir-create-type-converter) mlir-destroy-type-converter)
         body ...)]))

  (define-syntax with-conversion-target
    (syntax-rules ()
      [(_ (var ctx) body ...)
       (with-raii (var (mlir-create-conversion-target ctx) mlir-destroy-conversion-target)
         body ...)]))

  (define-syntax with-rewrite-pattern-set
    (syntax-rules ()
      [(_ (var ctx) body ...)
       (with-raii (var (mlir-create-rewrite-pattern-set ctx) mlir-destroy-rewrite-pattern-set)
         body ...)]))

) ;; end library (mlir core conversion)
