#!r6rs
;;===----------------------------------------------------------------------===;;
;;
;; Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
;; Licensed under the MIT License.
;;
;;===----------------------------------------------------------------------===;;
;;
;; (mlir core operation) — MLIR Operation primitives.
;;
;; Mirrors mlir/IR/Operation.h. All functions take an Operation* uptr.
;;
;;===----------------------------------------------------------------------===;;

(library (mlir core operation)
  (export
    mlir-operation-name
    mlir-operation-get-context
    mlir-operation-num-operands
    mlir-operation-num-results
    mlir-operation-get-operand
    mlir-operation-get-result
    mlir-operation-get-parent
    mlir-operation-get-operand-value
    mlir-operation-get-result-value
    mlir-operation-get-loc
    mlir-operation-get-block-argument
    mlir-operation-walk
    mlir-operation-get-attribute
    mlir-operation-get-attr
    mlir-operation-set-attribute!
    mlir-operation-has-attr?
    mlir-operation-set-operand
    mlir-operation-use-empty?
    mlir-operation-num-dps-inits
    mlir-operation-get-dps-init-value
    mlir-emit-error!
    mlir-emit-warning!
    mlir-emit-remark!)

  (import (rnrs)
          (only (chezscheme) foreign-procedure))

  (define mlir-operation-name
    (foreign-procedure "mlir_operation_get_name" (uptr) string))
  (define mlir-operation-get-context
    (foreign-procedure "mlir_operation_get_context" (uptr) uptr))
  (define mlir-operation-num-operands
    (foreign-procedure "mlir_operation_num_operands" (uptr) iptr))
  (define mlir-operation-num-results
    (foreign-procedure "mlir_operation_num_results" (uptr) iptr))
  (define mlir-operation-get-operand
    (foreign-procedure "mlir_operation_get_operand" (uptr iptr) uptr))
  (define mlir-operation-get-result
    (foreign-procedure "mlir_operation_get_result" (uptr iptr) uptr))
  (define mlir-operation-get-parent
    (foreign-procedure "mlir_operation_get_parent" (uptr) uptr))
  (define mlir-operation-get-operand-value
    (foreign-procedure "mlir_operation_get_operand_value" (uptr int) uptr))
  (define mlir-operation-get-result-value
    (foreign-procedure "mlir_operation_get_result_value" (uptr int) uptr))
  (define mlir-operation-get-loc
    (foreign-procedure "mlir_operation_get_loc" (uptr) uptr))
  (define mlir-operation-get-block-argument
    (foreign-procedure "mlir_operation_get_block_argument" (uptr int) uptr))
  (define mlir-operation-walk
    (foreign-procedure "mlir_operation_walk" (uptr scheme-object) void))

  (define mlir-operation-get-attribute
    (foreign-procedure "mlir_operation_get_attribute" (uptr string) uptr))
  (define mlir-operation-set-attribute!
    (foreign-procedure "mlir_operation_set_attribute" (uptr string uptr) void))
  (define (mlir-operation-has-attr? op name)
    (= 1 ((foreign-procedure "mlir_operation_has_attr" (uptr string) int) op name)))

  ;; Typed attribute getters — extract a Scheme value from a named attribute.
  (define %get-string-attr
    (foreign-procedure "mlir_operation_get_string_attr" (uptr string) string))
  (define %get-i64-attr
    (foreign-procedure "mlir_operation_get_integer_attr" (uptr string integer-64) integer-64))
  (define %get-i64-array-attr
    (foreign-procedure "mlir_operation_get_integer_array_attr" (uptr string) scheme-object))

  (define (mlir-operation-get-attr op name type . rest)
    (case type
      [(:string)    (%get-string-attr op name)]
      [(:i64)       (%get-i64-attr op name (if (null? rest) 0 (car rest)))]
      [(:i64-array) (%get-i64-array-attr op name)]
      [else (error 'mlir-operation-get-attr "unknown attr type" type)]))
  (define mlir-operation-set-operand
    (foreign-procedure "mlir_operation_set_operand" (uptr int uptr) void))
  (define (mlir-operation-use-empty? op)
    (= 1 ((foreign-procedure "mlir_operation_use_empty" (uptr) int) op)))
  (define mlir-operation-num-dps-inits
    (foreign-procedure "mlir_operation_num_dps_inits" (uptr) int))
  (define mlir-operation-get-dps-init-value
    (foreign-procedure "mlir_operation_get_dps_init_value" (uptr int) uptr))

  (define mlir-emit-error!
    (foreign-procedure "mlir_emit_error"   (uptr string) void))
  (define mlir-emit-warning!
    (foreign-procedure "mlir_emit_warning" (uptr string) void))
  (define mlir-emit-remark!
    (foreign-procedure "mlir_emit_remark"  (uptr string) void))

) ;; end library (mlir core operation)
