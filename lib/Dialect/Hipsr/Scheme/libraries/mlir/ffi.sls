#!r6rs
;;===----------------------------------------------------------------------===;;
;;
;; Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
;; Licensed under the MIT License.
;;
;;===----------------------------------------------------------------------===;;
;;
;; @file ffi.sls
;; @brief MLIR FFI - Foreign Function Interface to MLIR C++ API
;;
;; This library provides the low-level FFI bindings to MLIR operations.
;; It is the only Chez-specific module; all other modules should use
;; standard R6RS and import from this library.
;;
;; @section Pointer Representation
;;
;; MLIR pointers (Operation*, Value*, Type*, etc.) are represented as
;; Scheme exact integers using the `unsigned-64` FFI type.
;;
;; On the C++ side, these pointers are converted using:
;; - C++ → Scheme: `Sunsigned64(reinterpret_cast<uint64_t>(ptr))`
;; - Scheme → C++: `reinterpret_cast<T*>(Sunsigned64_value(scheme_val))`
;;
;; `Sunsigned64()` always creates a bignum (never fixnum), preserving all
;; 64 bits exactly without bit shifting. This is CRITICAL - do NOT use
;; `Sinteger()` for pointers as it may shift bits if the value fits in
;; fixnum range, losing the top 3 bits.
;;
;; @section Type Safety
;;
;; These FFI functions provide NO type checking on pointer arguments.
;; Passing a Value* where an Operation* is expected will cause undefined
;; behavior (likely a crash). The type system is enforced only by careful
;; programming conventions.
;;
;;===----------------------------------------------------------------------===;;

(library (mlir ffi)

  ;;===--------------------------------------------------------------------===;;
  ;; Foreign Type Definitions
  ;;===--------------------------------------------------------------------===;;

  ;; ArrayRef-like type for passing arrays across FFI boundary
  ;; Mirrors C++ ArrayRef<mlir::Value>: pointer + length
  (define-ftype ValueArrayRef
    (struct
      [data (* void)]   ; Value* pointer (stored as void* since we can't import C++ types)
      [size unsigned-64]))  ; Number of elements

  (export
    ;; Foreign types
    ValueArrayRef
    ;; ValueArrayRef accessors
    value-array-ref-size
    value-array-ref-at
    ;; Operation inspection
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

    ;; Dialect conversion framework primitives
    mlir-create-type-converter
    mlir-destroy-type-converter
    mlir-type-converter-add-device-memory-conversions
    mlir-create-conversion-target
    mlir-destroy-conversion-target
    mlir-conversion-target-add-illegal-onnx
    mlir-conversion-target-add-legal-hipsr
    mlir-conversion-target-add-legal-common-ops
    mlir-conversion-target-add-dynamically-legal-func
    mlir-conversion-target-mark-unknown-ops-nested-legal
    mlir-create-rewrite-pattern-set
    mlir-destroy-rewrite-pattern-set
    mlir-apply-full-conversion

    ;; Dialect conversion helpers
    mlir-populate-cast-conversion-patterns
    mlir-populate-return-conversion-patterns
    mlir-populate-func-type-conversion-pattern
    mlir-erase-dead-novalue-ops
    mlir-rewire-placeholder-inputs

    ;; IR construction
    mlir-create-placeholder-op
    mlir-create-cast-op
    mlir-create-generic-op
    mlir-operation-get-result-value-from-op
    mlir-create-unrealized-conversion-cast

    ;; Pattern rewriting
    mlir-replace-op
    mlir-erase-op
    mlir-notify-match-failure

    ;; Pattern registration (for Scheme-defined patterns)
    mlir-register-conversion-pattern
    )

  (import (chezscheme))

  ;;===--------------------------------------------------------------------===;;
  ;; Operation Inspection
  ;;===--------------------------------------------------------------------===;;

  ;;; @brief Get the name of an MLIR operation (e.g., "onnx.Cast")
  ;;; @param op-ptr Operation* as unsigned-64
  ;;; @return String containing operation name
  (define mlir-operation-name
    (foreign-procedure "mlir_operation_get_name" (unsigned-64) string))

  ;;; @brief Get the MLIRContext from an operation
  ;;; @param op-ptr Operation* as unsigned-64
  ;;; @return MLIRContext* as unsigned-64
  (define mlir-operation-get-context
    (foreign-procedure "mlir_operation_get_context" (unsigned-64) unsigned-64))

  ;;; @brief Get the number of operands for an operation
  ;;; @param op-ptr Operation* as unsigned-64
  ;;; @return Number of operands as signed pointer-sized integer
  (define mlir-operation-num-operands
    (foreign-procedure "mlir_operation_num_operands" (unsigned-64) iptr))

  ;;; @brief Get the number of results for an operation
  ;;; @param op-ptr Operation* as unsigned-64
  ;;; @return Number of results as signed pointer-sized integer
  (define mlir-operation-num-results
    (foreign-procedure "mlir_operation_num_results" (unsigned-64) iptr))

  ;;; @brief Get an operand OpOperand* from an operation by index
  ;;; @param op-ptr Operation* as unsigned-64
  ;;; @param index Operand index as signed pointer-sized integer
  ;;; @return OpOperand* as unsigned-64
  (define mlir-operation-get-operand
    (foreign-procedure "mlir_operation_get_operand" (unsigned-64 iptr) unsigned-64))

  ;;; @brief Get a result OpResult* from an operation by index
  ;;; @param op-ptr Operation* as unsigned-64
  ;;; @param index Result index as signed pointer-sized integer
  ;;; @return OpResult* as unsigned-64
  (define mlir-operation-get-result
    (foreign-procedure "mlir_operation_get_result" (unsigned-64 iptr) unsigned-64))

  ;;; @brief Get the parent operation
  ;;; @param op-ptr Operation* as unsigned-64
  ;;; @return Parent Operation* as unsigned-64, or 0 if no parent
  (define mlir-operation-get-parent
    (foreign-procedure "mlir_operation_get_parent" (unsigned-64) unsigned-64))

  ;;; @brief Get an operand Value* from an operation by index
  ;;; @param op-ptr Operation* as unsigned-64
  ;;; @param index Operand index as int
  ;;; @return Value* as unsigned-64
  (define mlir-operation-get-operand-value
    (foreign-procedure "mlir_operation_get_operand_value" (unsigned-64 int) unsigned-64))

  ;;; @brief Get a result Value* from an operation by index
  ;;; @param op-ptr Operation* as unsigned-64
  ;;; @param index Result index as int
  ;;; @return Value* as unsigned-64
  (define mlir-operation-get-result-value
    (foreign-procedure "mlir_operation_get_result_value" (unsigned-64 int) unsigned-64))

  ;;; @brief Get the location of an operation
  ;;; @param op-ptr Operation* as unsigned-64
  ;;; @return Location* as unsigned-64
  (define mlir-operation-get-loc
    (foreign-procedure "mlir_operation_get_loc" (unsigned-64) unsigned-64))

  ;;; @brief Get a block argument Value* from an operation
  ;;; @param op-ptr Operation* as unsigned-64
  ;;; @param index Argument index as int
  ;;; @return BlockArgument (Value*) as unsigned-64
  (define mlir-operation-get-block-argument
    (foreign-procedure "mlir_operation_get_block_argument" (unsigned-64 int) unsigned-64))

  ;;===--------------------------------------------------------------------===;;
  ;; Operation Traversal
  ;;===--------------------------------------------------------------------===;;

  ;;; @brief Walk all operations in the tree, calling callback for each
  ;;; @param op-ptr Root Operation* as unsigned-64
  ;;; @param callback Scheme procedure taking one argument (Operation* as unsigned-64)
  ;;; @note Callback is passed as scheme-object (GC-tracked), NOT unsigned-64
  (define mlir-operation-walk
    (foreign-procedure "mlir_operation_walk" (unsigned-64 scheme-object) void))

  ;;; @brief Walk all operations with rewriter support
  ;;; @param op-ptr Root Operation* as unsigned-64
  ;;; @param callback Scheme procedure taking one argument (Operation* as unsigned-64)
  ;;; @note Callback is passed as scheme-object (GC-tracked), NOT unsigned-64
  ;;; @note Callback can use mlir-replace-op/mlir-erase-op during walk
  (define mlir-operation-walk-rewrite
    (foreign-procedure "mlir_operation_walk_rewrite" (unsigned-64 scheme-object) void))

  ;;===--------------------------------------------------------------------===;;
  ;; Logging
  ;;===--------------------------------------------------------------------===;;

  ;;; @brief Log a trace-level message
  ;;; @param message String to log
  (define mlir-log-trace
    (foreign-procedure "mlir_log_trace" (string) void))

  ;;; @brief Log a debug-level message
  ;;; @param message String to log
  (define mlir-log-debug
    (foreign-procedure "mlir_log_debug" (string) void))

  ;;; @brief Log an info-level message
  ;;; @param message String to log
  (define mlir-log-info
    (foreign-procedure "mlir_log_info" (string) void))

  ;;; @brief Log a warning-level message
  ;;; @param message String to log
  (define mlir-log-warning
    (foreign-procedure "mlir_log_warning" (string) void))

  ;;; @brief Log an error-level message
  ;;; @param message String to log
  (define mlir-log-error
    (foreign-procedure "mlir_log_error" (string) void))

  ;;; @brief Log a fatal-level message
  ;;; @param message String to log
  (define mlir-log-fatal
    (foreign-procedure "mlir_log_fatal" (string) void))

  ;;===--------------------------------------------------------------------===;;
  ;; Type System
  ;;===--------------------------------------------------------------------===;;

  ;;; @brief Check if a type is a ranked tensor type
  ;;; @param type-ptr Type* as unsigned-64
  ;;; @return 1 if ranked tensor, 0 otherwise
  (define mlir-type-is-ranked-tensor
    (foreign-procedure "mlir_type_is_ranked_tensor" (unsigned-64) int))

  ;;; @brief Get the element type of a shaped type (tensor, memref, etc.)
  ;;; @param type-ptr Type* as unsigned-64
  ;;; @return Element Type* as unsigned-64
  (define mlir-type-get-element-type
    (foreign-procedure "mlir_type_get_element_type" (unsigned-64) unsigned-64))

  ;;; @brief Get the shape dimensions of a shaped type
  ;;; @param type-ptr Type* as unsigned-64
  ;;; @return Scheme list of integers representing shape (e.g., '(1 3 224 224))
  ;;; @note Returns a Scheme object (list), NOT an unsigned-64 pointer
  (define mlir-type-get-shape
    (foreign-procedure "mlir_type_get_shape" (unsigned-64) scheme-object))

  ;;; @brief Get the rank (number of dimensions) of a shaped type
  ;;; @param type-ptr Type* as unsigned-64
  ;;; @return Rank as int
  (define mlir-type-get-rank
    (foreign-procedure "mlir_type_get_rank" (unsigned-64) int))

  ;;; @brief Create a new type with specified memory space attribute
  ;;; @param type-ptr Type* as unsigned-64
  ;;; @param space Memory space identifier as int
  ;;; @return New Type* with updated memory space as unsigned-64
  (define mlir-type-set-memory-space
    (foreign-procedure "mlir_type_set_memory_space" (unsigned-64 int) unsigned-64))

  ;;; @brief Convert a tensor type to device memory space
  ;;; @param type-ptr Type* as unsigned-64
  ;;; @return New Type* in device memory space as unsigned-64
  (define mlir-tensor-type-in-device-space
    (foreign-procedure "mlir_tensor_type_in_device_space" (unsigned-64) unsigned-64))

  ;;; @brief Get the type of a value
  ;;; @param value-ptr Value* as unsigned-64
  ;;; @return Type* as unsigned-64
  (define mlir-value-get-type
    (foreign-procedure "mlir_value_get_type" (unsigned-64) unsigned-64))

  ;;===--------------------------------------------------------------------===;;
  ;; Utility Functions
  ;;===--------------------------------------------------------------------===;;

  ;;; @brief Get the HipSR context argument from a function operation
  ;;; @param op-ptr Function Operation* as unsigned-64
  ;;; @return BlockArgument (Value*) representing the context argument as unsigned-64
  (define mlir-get-hipsr-context-arg
    (foreign-procedure "mlir_get_hipsr_context_arg" (unsigned-64) unsigned-64))

  ;;; @brief Get the MLIRContext from an operation
  ;;; @param op-ptr Operation* as unsigned-64
  ;;; @return MLIRContext* as unsigned-64
  (define mlir-operation-get-context
    (foreign-procedure "mlir_operation_get_context" (unsigned-64) unsigned-64))

  ;;===--------------------------------------------------------------------===;;
  ;; Dialect Conversion Framework Primitives
  ;;===--------------------------------------------------------------------===;;

  ;;; @brief Create a new TypeConverter for dialect conversion
  ;;; @return TypeConverter* as unsigned-64
  (define mlir-create-type-converter
    (foreign-procedure "mlir_create_type_converter" () unsigned-64))

  ;;; @brief Destroy a TypeConverter
  ;;; @param converter-ptr TypeConverter* as unsigned-64
  (define mlir-destroy-type-converter
    (foreign-procedure "mlir_destroy_type_converter" (unsigned-64) void))

  ;;; @brief Add device memory space conversions to a TypeConverter
  ;;; @param converter-ptr TypeConverter* as unsigned-64
  ;;; @note Registers conversions for tensor types to device memory space
  (define mlir-type-converter-add-device-memory-conversions
    (foreign-procedure "mlir_type_converter_add_device_memory_conversions" (unsigned-64) void))

  ;;; @brief Create a ConversionTarget for dialect conversion
  ;;; @param context-ptr MLIRContext* as unsigned-64
  ;;; @return ConversionTarget* as unsigned-64
  (define mlir-create-conversion-target
    (foreign-procedure "mlir_create_conversion_target" (unsigned-64) unsigned-64))

  ;;; @brief Destroy a ConversionTarget
  ;;; @param target-ptr ConversionTarget* as unsigned-64
  (define mlir-destroy-conversion-target
    (foreign-procedure "mlir_destroy_conversion_target" (unsigned-64) void))

  ;;; @brief Mark all ONNX dialect operations as illegal in conversion target
  ;;; @param target-ptr ConversionTarget* as unsigned-64
  ;;; @note Operations marked illegal must be converted during dialect conversion
  (define mlir-conversion-target-add-illegal-onnx
    (foreign-procedure "mlir_conversion_target_add_illegal_onnx" (unsigned-64) void))

  ;;; @brief Mark all HipSR dialect operations as legal in conversion target
  ;;; @param target-ptr ConversionTarget* as unsigned-64
  ;;; @note Operations marked legal can remain unchanged during conversion
  (define mlir-conversion-target-add-legal-hipsr
    (foreign-procedure "mlir_conversion_target_add_legal_hipsr" (unsigned-64) void))

  ;;; @brief Mark common operations (func, return, etc.) as legal
  ;;; @param target-ptr ConversionTarget* as unsigned-64
  (define mlir-conversion-target-add-legal-common-ops
    (foreign-procedure "mlir_conversion_target_add_legal_common_ops" (unsigned-64) void))

  ;;; @brief Mark func operations as dynamically legal (checked per-operation)
  ;;; @param target-ptr ConversionTarget* as unsigned-64
  ;;; @param converter-ptr TypeConverter* as unsigned-64 (used for signature checks)
  ;;; @note Dynamic legality checks if function signatures use only legal types
  (define mlir-conversion-target-add-dynamically-legal-func
    (foreign-procedure "mlir_conversion_target_add_dynamically_legal_func"
                       (unsigned-64 unsigned-64) void))

  ;;; @brief Mark unknown operations as legal if nested in legal contexts
  ;;; @param target-ptr ConversionTarget* as unsigned-64
  ;;; @note Allows unknown ops inside legal regions without conversion
  (define mlir-conversion-target-mark-unknown-ops-nested-legal
    (foreign-procedure "mlir_conversion_target_mark_unknown_ops_nested_legal" (unsigned-64) void))

  ;;; @brief Create a RewritePatternSet for dialect conversion
  ;;; @param context-ptr MLIRContext* as unsigned-64
  ;;; @return RewritePatternSet* as unsigned-64
  (define mlir-create-rewrite-pattern-set
    (foreign-procedure "mlir_create_rewrite_pattern_set" (unsigned-64) unsigned-64))

  ;;; @brief Destroy a RewritePatternSet
  ;;; @param patterns-ptr RewritePatternSet* as unsigned-64
  (define mlir-destroy-rewrite-pattern-set
    (foreign-procedure "mlir_destroy_rewrite_pattern_set" (unsigned-64) void))

  ;;; @brief Apply full dialect conversion to a module
  ;;; @param module-ptr Module Operation* as unsigned-64
  ;;; @param target-ptr ConversionTarget* as unsigned-64
  ;;; @param patterns-ptr RewritePatternSet* as unsigned-64
  ;;; @return 0 on success, non-zero on failure
  (define mlir-apply-full-conversion
    (foreign-procedure "mlir_apply_full_conversion"
                       (unsigned-64 unsigned-64 unsigned-64) int))

  ;;===--------------------------------------------------------------------===;;
  ;; Dialect Conversion Helpers
  ;;===--------------------------------------------------------------------===;;

  ;;; @brief Add Cast operation conversion patterns
  ;;; @param patterns-ptr RewritePatternSet* as unsigned-64
  ;;; @param converter-ptr TypeConverter* as unsigned-64
  ;;; @param context-ptr MLIRContext* as unsigned-64
  ;;; @note Populates patterns for converting hipsr.cast operations
  (define mlir-populate-cast-conversion-patterns
    (foreign-procedure "mlir_populate_cast_conversion_patterns"
                       (unsigned-64 unsigned-64 unsigned-64) void))

  ;;; @brief Add Return operation conversion patterns
  ;;; @param patterns-ptr RewritePatternSet* as unsigned-64
  ;;; @param converter-ptr TypeConverter* as unsigned-64
  ;;; @param context-ptr MLIRContext* as unsigned-64
  ;;; @note Populates patterns for converting func.return operations
  (define mlir-populate-return-conversion-patterns
    (foreign-procedure "mlir_populate_return_conversion_patterns"
                       (unsigned-64 unsigned-64 unsigned-64) void))

  ;;; @brief Add function type conversion patterns
  ;;; @param patterns-ptr RewritePatternSet* as unsigned-64
  ;;; @param converter-ptr TypeConverter* as unsigned-64
  ;;; @note Populates patterns for converting function signatures
  (define mlir-populate-func-type-conversion-pattern
    (foreign-procedure "mlir_populate_func_type_conversion_pattern"
                       (unsigned-64 unsigned-64) void))

  ;;; @brief Erase dead operations with no values
  ;;; @param module-ptr Module Operation* as unsigned-64
  ;;; @note Cleanup pass to remove operations marked as dead during conversion
  (define mlir-erase-dead-novalue-ops
    (foreign-procedure "mlir_erase_dead_novalue_ops" (unsigned-64) void))

  ;;; @brief Rewire placeholder operation inputs after conversion
  ;;; @param module-ptr Module Operation* as unsigned-64
  ;;; @note Connects placeholder operations to their actual operands
  (define mlir-rewire-placeholder-inputs
    (foreign-procedure "mlir_rewire_placeholder_inputs" (unsigned-64) void))

  ;;===--------------------------------------------------------------------===;;
  ;; IR Construction
  ;;===--------------------------------------------------------------------===;;

  ;;; @brief Create a hipsr.placeholder operation
  ;;; @param ctx-ptr MLIRContext* as unsigned-64
  ;;; @param input-value Input Value* as unsigned-64
  ;;; @param result-type Result Type* as unsigned-64
  ;;; @param placeholder-type Placeholder type enum as int
  ;;; @return Created Operation* as unsigned-64
  ;;; @note Placeholder ops are temporary and get replaced during conversion
  (define mlir-create-placeholder-op
    (foreign-procedure "mlir_create_placeholder_op"
                       (unsigned-64 unsigned-64 unsigned-64 int) unsigned-64))

  ;;; @brief Create a hipsr.cast operation
  ;;; @param ctx-ptr MLIRContext* as unsigned-64
  ;;; @param input-value Input Value* as unsigned-64
  ;;; @param output-value Output Value* as unsigned-64 (typically from placeholder)
  ;;; @param result-type Result Type* as unsigned-64
  ;;; @return Created Operation* as unsigned-64
  (define mlir-create-cast-op
    (foreign-procedure "mlir_create_cast_op"
                       (unsigned-64 unsigned-64 unsigned-64 unsigned-64) unsigned-64))

  ;;; @brief Create a generic MLIR operation by name
  ;;; @param op-name Operation name as string (e.g., "hipsr.add")
  ;;; @param operands-list Scheme list of Value* (as unsigned-64)
  ;;; @param result-types-list Scheme list of Type* (as unsigned-64)
  ;;; @return Created Operation* as unsigned-64
  ;;; @note Lists are passed as scheme-object (GC-tracked), NOT unsigned-64
  (define mlir-create-generic-op
    (foreign-procedure "mlir_create_generic_op"
                       (string scheme-object scheme-object) unsigned-64))

  ;;; @brief Get a result Value* from an operation by index
  ;;; @param op-ptr Operation* as unsigned-64
  ;;; @param index Result index as int
  ;;; @return Value* as unsigned-64
  (define mlir-operation-get-result-value-from-op
    (foreign-procedure "mlir_operation_get_result_value_from_op"
                       (unsigned-64 int) unsigned-64))

  ;;; @brief Create an unrealized_conversion_cast operation
  ;;; @param input-value Input Value* as unsigned-64
  ;;; @param result-type Result Type* as unsigned-64
  ;;; @return Created Operation* as unsigned-64
  ;;; @note Used for type conversions that will be resolved later
  (define mlir-create-unrealized-conversion-cast
    (foreign-procedure "mlir_create_unrealized_conversion_cast"
                       (unsigned-64 unsigned-64) unsigned-64))

  ;;===--------------------------------------------------------------------===;;
  ;; Pattern Rewriting
  ;;===--------------------------------------------------------------------===;;

  ;;; @brief Replace an operation with another operation's results
  ;;; @param old-op-ptr Operation* to replace as unsigned-64
  ;;; @param new-op-ptr Operation* whose results replace old-op as unsigned-64
  ;;; @return 0 on success, non-zero on failure
  ;;; @note Old operation is marked for deletion, new operation provides results
  (define mlir-replace-op
    (foreign-procedure "mlir_replace_op" (unsigned-64 unsigned-64) int))

  ;;; @brief Erase an operation from the IR
  ;;; @param op-ptr Operation* to erase as unsigned-64
  ;;; @return 0 on success, non-zero on failure
  ;;; @note Operation must have no uses remaining
  (define mlir-erase-op
    (foreign-procedure "mlir_erase_op" (unsigned-64) int))

  ;;; @brief Notify pattern matching system of a match failure
  ;;; @param op-ptr Operation* that failed to match as unsigned-64
  ;;; @param reason Failure reason as string
  ;;; @note Used for debugging pattern matching failures
  (define mlir-notify-match-failure
    (foreign-procedure "mlir_notify_match_failure" (unsigned-64 string) void))

  ;;===--------------------------------------------------------------------===;;
  ;; Pattern Registration
  ;;===--------------------------------------------------------------------===;;

  ;;; @brief Register a Scheme-defined conversion pattern
  ;;; @param patterns-ptr RewritePatternSet* as unsigned-64
  ;;; @param op-name Operation name to match (e.g., "onnx.Cast")
  ;;; @param callback Scheme procedure for pattern rewriting
  ;;; @param type-converter-ptr TypeConverter* as unsigned-64
  ;;; @note Callback signature: (lambda (op operands-ref rewriter type-converter) ...)
  ;;;       - op: Operation* being converted (unsigned-64)
  ;;;       - operands-ref: ValueArrayRef* pointer (unsigned-64)
  ;;;       - rewriter: ConversionPatternRewriter* (unsigned-64)
  ;;;       - type-converter: TypeConverter* (unsigned-64)
  ;;;       Returns: #t on successful rewrite, #f on match failure
  ;;; @note Callback is passed as scheme-object (GC-tracked) and locked with Slock_object
  ;;; @note Mirrors C++ OpConversionPattern::matchAndRewrite signature
  (define mlir-register-conversion-pattern
    (foreign-procedure "mlir_register_conversion_pattern"
                       (unsigned-64 string scheme-object unsigned-64) void))

  ;;===--------------------------------------------------------------------===;;
  ;; ValueArrayRef Accessors
  ;;===--------------------------------------------------------------------===;;

  ;;; @brief Get the size of a ValueArrayRef
  ;;; @param ref-ptr ValueArrayRef* as unsigned-64
  ;;; @return Number of elements
  (define (value-array-ref-size ref-ptr)
    (ftype-ref ValueArrayRef (size) ref-ptr))

  ;;; @brief Get Value* at index from ValueArrayRef
  ;;; @param ref-ptr ValueArrayRef* as unsigned-64
  ;;; @param index Zero-based index
  ;;; @return Value* as unsigned-64
  (define (value-array-ref-at ref-ptr index)
    (let* ([data-ptr (ftype-ref ValueArrayRef (data) ref-ptr)]
           [offset (* index 8)])  ; Assuming 64-bit pointers
      ;; Read pointer at offset
      (foreign-ref 'unsigned-64 data-ptr offset)))

) ;; end library (mlir ffi)
