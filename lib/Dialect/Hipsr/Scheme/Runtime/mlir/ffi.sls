#!r6rs
;;===----------------------------------------------------------------------===;;
;;
;; Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
;; Licensed under the MIT License.
;;
;;===----------------------------------------------------------------------===;;
;;
;; MLIR FFI - Foreign Function Interface to MLIR C++ API
;;
;; This library provides the low-level FFI bindings to MLIR operations.
;; It is the only Chez-specific module; all other modules should use
;; standard R6RS and import from this library.
;;
;;===----------------------------------------------------------------------===;;

(library (mlir ffi)
  (export
    ;; Operation inspection
    mlir-operation-name
    mlir-operation-num-operands
    mlir-operation-num-results
    mlir-operation-get-operand
    mlir-operation-get-result
    mlir-operation-get-parent
    mlir-operation-get-operand-value
    mlir-operation-get-result-value
    mlir-operation-get-loc
    mlir-operation-get-block-argument

    ;; Operation traversal
    mlir-operation-walk
    mlir-operation-walk-rewrite

    ;; Logging
    mlir-log-trace
    mlir-log-debug
    mlir-log-info
    mlir-log-warning
    mlir-log-error
    mlir-log-fatal

    ;; Type system
    mlir-type-is-ranked-tensor
    mlir-type-get-element-type
    mlir-type-get-shape
    mlir-type-get-rank
    mlir-type-set-memory-space
    mlir-tensor-type-in-device-space
    mlir-value-get-type

    ;; Utility
    mlir-get-hipsr-context-arg
    mlir-operation-get-context
    mlir-context-is-dialect-loaded

    ;; Dialect conversion helpers
    mlir-populate-cast-conversion-patterns
    mlir-populate-return-conversion-patterns
    mlir-populate-func-type-conversion-pattern
    mlir-erase-dead-novalue-ops
    mlir-rewire-placeholder-inputs
    mlir-apply-dialect-conversion-onnx-to-hipsr

    ;; IR construction
    mlir-create-placeholder-op
    mlir-create-cast-op
    mlir-create-unrealized-conversion-cast

    ;; Pattern rewriting
    mlir-replace-op
    mlir-erase-op
    mlir-notify-match-failure
    )

  (import (chezscheme))

  ;;===--------------------------------------------------------------------===;;
  ;; Operation Inspection
  ;;===--------------------------------------------------------------------===;;

  (define mlir-operation-name
    (foreign-procedure "mlir_operation_get_name" (unsigned-64) string))

  (define mlir-operation-num-operands
    (foreign-procedure "mlir_operation_num_operands" (unsigned-64) iptr))

  (define mlir-operation-num-results
    (foreign-procedure "mlir_operation_num_results" (unsigned-64) iptr))

  (define mlir-operation-get-operand
    (foreign-procedure "mlir_operation_get_operand" (unsigned-64 iptr) unsigned-64))

  (define mlir-operation-get-result
    (foreign-procedure "mlir_operation_get_result" (unsigned-64 iptr) unsigned-64))

  (define mlir-operation-get-parent
    (foreign-procedure "mlir_operation_get_parent" (unsigned-64) unsigned-64))

  (define mlir-operation-get-operand-value
    (foreign-procedure "mlir_operation_get_operand_value" (unsigned-64 int) unsigned-64))

  (define mlir-operation-get-result-value
    (foreign-procedure "mlir_operation_get_result_value" (unsigned-64 int) unsigned-64))

  (define mlir-operation-get-loc
    (foreign-procedure "mlir_operation_get_loc" (unsigned-64) unsigned-64))

  (define mlir-operation-get-block-argument
    (foreign-procedure "mlir_operation_get_block_argument" (unsigned-64 int) unsigned-64))

  ;;===--------------------------------------------------------------------===;;
  ;; Operation Traversal
  ;;===--------------------------------------------------------------------===;;

  (define mlir-operation-walk
    (foreign-procedure "mlir_operation_walk" (unsigned-64 scheme-object) void))

  (define mlir-operation-walk-rewrite
    (foreign-procedure "mlir_operation_walk_rewrite" (unsigned-64 scheme-object) void))

  ;;===--------------------------------------------------------------------===;;
  ;; Logging
  ;;===--------------------------------------------------------------------===;;

  (define mlir-log-trace
    (foreign-procedure "mlir_log_trace" (string) void))

  (define mlir-log-debug
    (foreign-procedure "mlir_log_debug" (string) void))

  (define mlir-log-info
    (foreign-procedure "mlir_log_info" (string) void))

  (define mlir-log-warning
    (foreign-procedure "mlir_log_warning" (string) void))

  (define mlir-log-error
    (foreign-procedure "mlir_log_error" (string) void))

  (define mlir-log-fatal
    (foreign-procedure "mlir_log_fatal" (string) void))

  ;;===--------------------------------------------------------------------===;;
  ;; Type System
  ;;===--------------------------------------------------------------------===;;

  (define mlir-type-is-ranked-tensor
    (foreign-procedure "mlir_type_is_ranked_tensor" (unsigned-64) int))

  (define mlir-type-get-element-type
    (foreign-procedure "mlir_type_get_element_type" (unsigned-64) unsigned-64))

  (define mlir-type-get-shape
    (foreign-procedure "mlir_type_get_shape" (unsigned-64) scheme-object))

  (define mlir-type-get-rank
    (foreign-procedure "mlir_type_get_rank" (unsigned-64) int))

  (define mlir-type-set-memory-space
    (foreign-procedure "mlir_type_set_memory_space" (unsigned-64 int) unsigned-64))

  (define mlir-tensor-type-in-device-space
    (foreign-procedure "mlir_tensor_type_in_device_space" (unsigned-64) unsigned-64))

  (define mlir-value-get-type
    (foreign-procedure "mlir_value_get_type" (unsigned-64) unsigned-64))

  ;;===--------------------------------------------------------------------===;;
  ;; Utility Functions
  ;;===--------------------------------------------------------------------===;;

  (define mlir-get-hipsr-context-arg
    (foreign-procedure "mlir_get_hipsr_context_arg" (unsigned-64) unsigned-64))

  (define mlir-operation-get-context
    (foreign-procedure "mlir_operation_get_context" (unsigned-64) unsigned-64))

  (define mlir-context-is-dialect-loaded
    (foreign-procedure "mlir_context_is_dialect_loaded" (unsigned-64 string) int))

  ;;===--------------------------------------------------------------------===;;
  ;; Dialect Conversion Helpers
  ;;===--------------------------------------------------------------------===;;

  (define mlir-populate-cast-conversion-patterns
    (foreign-procedure "mlir_populate_cast_conversion_patterns"
                       (unsigned-64 unsigned-64 unsigned-64) void))

  (define mlir-populate-return-conversion-patterns
    (foreign-procedure "mlir_populate_return_conversion_patterns"
                       (unsigned-64 unsigned-64 unsigned-64) void))

  (define mlir-populate-func-type-conversion-pattern
    (foreign-procedure "mlir_populate_func_type_conversion_pattern"
                       (unsigned-64 unsigned-64) void))

  (define mlir-erase-dead-novalue-ops
    (foreign-procedure "mlir_erase_dead_novalue_ops" (unsigned-64) void))

  (define mlir-rewire-placeholder-inputs
    (foreign-procedure "mlir_rewire_placeholder_inputs" (unsigned-64) void))

  (define mlir-apply-dialect-conversion-onnx-to-hipsr
    (foreign-procedure "mlir_apply_dialect_conversion_onnx_to_hipsr" (unsigned-64) int))

  ;;===--------------------------------------------------------------------===;;
  ;; IR Construction
  ;;===--------------------------------------------------------------------===;;

  (define mlir-create-placeholder-op
    (foreign-procedure "mlir_create_placeholder_op"
                       (unsigned-64 unsigned-64 unsigned-64 int) unsigned-64))

  (define mlir-create-cast-op
    (foreign-procedure "mlir_create_cast_op"
                       (unsigned-64 unsigned-64 unsigned-64 unsigned-64) unsigned-64))

  (define mlir-create-unrealized-conversion-cast
    (foreign-procedure "mlir_create_unrealized_conversion_cast"
                       (unsigned-64 unsigned-64) unsigned-64))

  ;;===--------------------------------------------------------------------===;;
  ;; Pattern Rewriting
  ;;===--------------------------------------------------------------------===;;

  (define mlir-replace-op
    (foreign-procedure "mlir_replace_op" (unsigned-64 unsigned-64) int))

  (define mlir-erase-op
    (foreign-procedure "mlir_erase_op" (unsigned-64) int))

  (define mlir-notify-match-failure
    (foreign-procedure "mlir_notify_match_failure" (unsigned-64 string) void))

) ;; end library (mlir ffi)
