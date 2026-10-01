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

  ;; Allocate a new TypeConverter on the heap.
  ;; Returns: TypeConverter* uptr (must be freed with mlir-destroy-type-converter).
  (define mlir-create-type-converter
    (foreign-procedure "mlir_create_type_converter" () uptr))

  ;; Free a TypeConverter created by mlir-create-type-converter.
  ;; converter: TypeConverter* uptr
  (define mlir-destroy-type-converter
    (foreign-procedure "mlir_destroy_type_converter" (uptr) void))

  ;; Add a source-materialization that widens tensor<?x32> to tensor<?x?> when needed.
  ;; Inserted for shaped-type mismatches that arise during rank inference.
  ;; converter: TypeConverter* uptr — mutated in place
  (define mlir-type-converter-add-tensor-widening-materialization
    (foreign-procedure "mlir_type_converter_add_tensor_widening_materialization" (uptr) void))

  ;; Add a Scheme type-conversion callback.
  ;; converter: TypeConverter* uptr — mutated in place
  ;; callback:  Scheme procedure (lambda (type-uptr) -> type-uptr or #f/#0)
  ;;            Returns the converted type, or #f/#0 if this converter does not
  ;;            handle the given type.  The callback is GC-locked for its lifetime.
  (define mlir-type-converter-add-conversion
    (foreign-procedure "mlir_type_converter_add_conversion" (uptr scheme-object) void))

  ;; Returns 1 if the given type is legal according to the converter, 0 otherwise.
  ;; converter: TypeConverter* uptr
  ;; type:      Type* uptr (opaque)
  (define mlir-type-converter-is-legal-type
    (foreign-procedure "mlir_type_converter_is_legal_type" (uptr uptr) int))

  ;; Returns 1 if all result types of the given operation are legal, 0 otherwise.
  ;; converter: TypeConverter* uptr
  ;; op:        Operation* uptr
  (define mlir-type-converter-is-legal
    (foreign-procedure "mlir_type_converter_is_legal" (uptr uptr) int))

  ;; Returns 1 if the function signature (arg + result types) is legal, 0 otherwise.
  ;; converter: TypeConverter* uptr
  ;; op:        FunctionOpInterface Operation* uptr
  (define mlir-type-converter-is-signature-legal
    (foreign-procedure "mlir_type_converter_is_signature_legal" (uptr uptr) int))

  ;;===--------------------------------------------------------------------===;;
  ;; ConversionTarget
  ;;===--------------------------------------------------------------------===;;

  ;; Allocate a ConversionTarget for the given MLIRContext.
  ;; ctx:     MLIRContext* uptr
  ;; Returns: ConversionTarget* uptr (must be freed with mlir-destroy-conversion-target).
  (define mlir-create-conversion-target
    (foreign-procedure "mlir_create_conversion_target" (uptr) uptr))

  ;; Free a ConversionTarget created by mlir-create-conversion-target.
  ;; target: ConversionTarget* uptr
  (define mlir-destroy-conversion-target
    (foreign-procedure "mlir_destroy_conversion_target" (uptr) void))

  ;; Mark all ops in a dialect as illegal (must be converted).
  ;; target:       ConversionTarget* uptr — mutated in place
  ;; dialect-name: string, e.g. "onnx"
  (define mlir-conversion-target-add-illegal-dialect
    (foreign-procedure "mlir_conversion_target_add_illegal_dialect" (uptr string) void))

  ;; Mark all ops in a dialect as legal (may pass through unchanged).
  ;; target:       ConversionTarget* uptr — mutated in place
  ;; dialect-name: string, e.g. "func"
  (define mlir-conversion-target-add-legal-dialect
    (foreign-procedure "mlir_conversion_target_add_legal_dialect" (uptr string) void))

  ;; Mark a single named op as legal.
  ;; target:   ConversionTarget* uptr — mutated in place
  ;; ctx:      MLIRContext* uptr — needed to intern the OperationName
  ;; op-name:  string, e.g. "onnx.NoValue"
  (define mlir-conversion-target-add-legal-op
    (foreign-procedure "mlir_conversion_target_add_legal_op" (uptr uptr string) void))

  ;; Mark a named op as dynamically legal using a Scheme predicate.
  ;; target:   ConversionTarget* uptr — mutated in place
  ;; ctx:      MLIRContext* uptr
  ;; op-name:  string
  ;; callback: Scheme procedure (lambda (op-uptr) -> #t if legal, #f if not)
  ;;           GC-locked for the lifetime of the ConversionTarget.
  (define mlir-conversion-target-add-dynamically-legal-op
    (foreign-procedure "mlir_conversion_target_add_dynamically_legal_op"
                       (uptr uptr string scheme-object) void))

  ;; Mark all unknown ops as dynamically legal using a Scheme predicate.
  ;; Applied to ops whose dialect is not otherwise registered on the target.
  ;; target:   ConversionTarget* uptr — mutated in place
  ;; callback: Scheme procedure (lambda (op-uptr) -> #t if legal, #f if not)
  ;;           GC-locked for the lifetime of the ConversionTarget.
  (define mlir-conversion-target-mark-unknown-ops-dynamically-legal
    (foreign-procedure "mlir_conversion_target_mark_unknown_ops_dynamically_legal"
                       (uptr scheme-object) void))

  ;;===--------------------------------------------------------------------===;;
  ;; RewritePatternSet
  ;;===--------------------------------------------------------------------===;;

  ;; Allocate a RewritePatternSet for the given MLIRContext.
  ;; ctx:     MLIRContext* uptr
  ;; Returns: RewritePatternSet* uptr (must be freed with mlir-destroy-rewrite-pattern-set).
  (define mlir-create-rewrite-pattern-set
    (foreign-procedure "mlir_create_rewrite_pattern_set" (uptr) uptr))

  ;; Free a RewritePatternSet created by mlir-create-rewrite-pattern-set.
  ;; patterns: RewritePatternSet* uptr
  (define mlir-destroy-rewrite-pattern-set
    (foreign-procedure "mlir_destroy_rewrite_pattern_set" (uptr) void))

  ;; Apply full dialect conversion: rewrite all ops in module according to
  ;; the target and patterns.  Consumes the patterns set.
  ;; module:   ModuleOp Operation* uptr
  ;; target:   ConversionTarget* uptr
  ;; patterns: RewritePatternSet* uptr (consumed — do not use after this call)
  ;; Returns:  1 on success, 0 on failure (MLIR diagnostics already emitted).
  (define mlir-apply-full-conversion
    (foreign-procedure "mlir_apply_full_conversion" (uptr uptr uptr) int))

  ;;===--------------------------------------------------------------------===;;
  ;; Pattern Registration
  ;;===--------------------------------------------------------------------===;;

  ;; Register a Scheme-defined ConversionPattern for a named op.
  ;; patterns:       RewritePatternSet* uptr — pattern added in-place
  ;; op-name:        string op name to match, e.g. "onnx.Cast"
  ;; callback:       Scheme procedure called when op-name is encountered:
  ;;                   (lambda (op operands-ref rewriter type-converter) → #t/#f)
  ;;                   op:             Operation* uptr
  ;;                   operands-ref:   ArrayRef<Value>* uptr (converted operands)
  ;;                   rewriter:       ConversionPatternRewriter* uptr
  ;;                   type-converter: TypeConverter* uptr
  ;;                   Returns #t on success, #f to signal pattern failure.
  ;; type-converter: TypeConverter* uptr passed to the callback
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
