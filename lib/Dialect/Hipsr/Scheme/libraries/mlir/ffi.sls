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
;; Scheme exact integers using the `uptr` FFI type.
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

  (export
    ;; ValueArrayRef accessors (ftype itself is not exported, only accessors)
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

    ;; Value operations
    mlir-value-get-defining-op
    mlir-value-get-type
    mlir-value-is-block-argument
    mlir-value-get-result-number

    ;; Operation attributes and mutation
    mlir-operation-set-attr
    mlir-operation-set-operand
    mlir-operation-use-empty
    mlir-operation-num-dps-inits
    mlir-operation-get-dps-init-value

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
    mlir-type-get-encoding
    mlir-value-get-type

    ;; Utility
    mlir-operation-get-context

    ;; Dialect conversion framework — lifecycle
    mlir-create-type-converter
    mlir-destroy-type-converter
    mlir-create-conversion-target
    mlir-destroy-conversion-target
    mlir-create-rewrite-pattern-set
    mlir-destroy-rewrite-pattern-set
    mlir-apply-full-conversion

    ;; Dialect conversion framework — generic configuration
    mlir-type-converter-add-conversion
    mlir-type-converter-is-legal-type
    mlir-type-converter-is-legal
    mlir-type-converter-is-signature-legal
    mlir-conversion-target-add-illegal-dialect
    mlir-conversion-target-add-legal-dialect
    mlir-conversion-target-add-legal-op
    mlir-conversion-target-add-dynamically-legal-op
    mlir-conversion-target-mark-unknown-ops-dynamically-legal

    ;; Dialect conversion helpers (generic MLIR utilities)
    mlir-populate-func-type-conversion-pattern

    ;; Builder — explicit rewriter-based op construction
    mlir-build-op
    mlir-set-insertion-point-before
    mlir-set-insertion-point-to-block-end
    mlir-op-get-region
    mlir-region-create-block
    mlir-block-get-argument
    mlir-get-shape-shape-type
    mlir-operation-get-result           ; get result value from op at index

    ;; Pattern rewriting
    mlir-replace-op     ; (rewriter old-op new-value) → int
    mlir-erase-op       ; (rewriter op) → int
    mlir-op-erase       ; (op) → void  — direct erase, no rewriter needed
    mlir-notify-match-failure

    ;; Pattern registration (for Scheme-defined patterns)
    mlir-register-conversion-pattern

    ;; Resource management
    with-raii
    with-type-converter
    with-conversion-target
    with-rewrite-pattern-set
    )

  (import (chezscheme))

  ;;===--------------------------------------------------------------------===;;
  ;; Foreign Type Definitions
  ;;===--------------------------------------------------------------------===;;

  ;; ArrayRef-like type for passing arrays across FFI boundary
  ;; Mirrors C++ ArrayRef<mlir::Value>: pointer + length
  (define-ftype ValueArrayRef
    (struct
      [data uptr]   ; Value* pointer (stored as uptr since we can't import C++ types)
      [size uptr]))  ; Number of elements

  ;;===--------------------------------------------------------------------===;;
  ;; Operation Inspection
  ;;===--------------------------------------------------------------------===;;

  ;;; @brief Get the name of an MLIR operation (e.g., "onnx.Cast")
  ;;; @param op-ptr Operation* as uptr
  ;;; @return String containing operation name
  (define mlir-operation-name
    (foreign-procedure "mlir_operation_get_name" (uptr) string))

  ;;; @brief Get the MLIRContext from an operation
  ;;; @param op-ptr Operation* as uptr
  ;;; @return MLIRContext* as uptr
  (define mlir-operation-get-context
    (foreign-procedure "mlir_operation_get_context" (uptr) uptr))

  ;;; @brief Get the number of operands for an operation
  ;;; @param op-ptr Operation* as uptr
  ;;; @return Number of operands as signed pointer-sized integer
  (define mlir-operation-num-operands
    (foreign-procedure "mlir_operation_num_operands" (uptr) iptr))

  ;;; @brief Get the number of results for an operation
  ;;; @param op-ptr Operation* as uptr
  ;;; @return Number of results as signed pointer-sized integer
  (define mlir-operation-num-results
    (foreign-procedure "mlir_operation_num_results" (uptr) iptr))

  ;;; @brief Get an operand OpOperand* from an operation by index
  ;;; @param op-ptr Operation* as uptr
  ;;; @param index Operand index as signed pointer-sized integer
  ;;; @return OpOperand* as uptr
  (define mlir-operation-get-operand
    (foreign-procedure "mlir_operation_get_operand" (uptr iptr) uptr))

  ;;; @brief Get a result OpResult* from an operation by index
  ;;; @param op-ptr Operation* as uptr
  ;;; @param index Result index as signed pointer-sized integer
  ;;; @return OpResult* as uptr
  (define mlir-operation-get-result
    (foreign-procedure "mlir_operation_get_result" (uptr iptr) uptr))

  ;;; @brief Get the defining operation of a value
  ;;; @param value-ptr Value* as uptr
  ;;; @return Operation* as uptr, or 0 for block arguments
  (define mlir-value-get-defining-op
    (foreign-procedure "mlir_value_get_defining_op" (uptr) uptr))

  ;;; @brief Check whether a value is a block argument
  ;;; @param value-ptr Value* as uptr
  ;;; @return 1 if block argument, 0 if op result
  (define mlir-value-is-block-argument
    (foreign-procedure "mlir_value_is_block_argument" (uptr) int))

  ;;; @brief Get the result index of an OpResult value
  ;;; @param value-ptr Value* as uptr
  ;;; @return Result index, or -1 if value is a block argument
  (define mlir-value-get-result-number
    (foreign-procedure "mlir_value_get_result_number" (uptr) int))

  ;;; @brief Get the parent operation
  ;;; @param op-ptr Operation* as uptr
  ;;; @return Parent Operation* as uptr, or 0 if no parent
  (define mlir-operation-get-parent
    (foreign-procedure "mlir_operation_get_parent" (uptr) uptr))

  ;;; @brief Get an operand Value* from an operation by index
  ;;; @param op-ptr Operation* as uptr
  ;;; @param index Operand index as int
  ;;; @return Value* as uptr
  (define mlir-operation-get-operand-value
    (foreign-procedure "mlir_operation_get_operand_value" (uptr int) uptr))

  ;;; @brief Get a result Value* from an operation by index
  ;;; @param op-ptr Operation* as uptr
  ;;; @param index Result index as int
  ;;; @return Value* as uptr
  (define mlir-operation-get-result-value
    (foreign-procedure "mlir_operation_get_result_value" (uptr int) uptr))

  ;;; @brief Get the location of an operation
  ;;; @param op-ptr Operation* as uptr
  ;;; @return Location* as uptr
  (define mlir-operation-get-loc
    (foreign-procedure "mlir_operation_get_loc" (uptr) uptr))

  ;;; @brief Get a block argument Value* from an operation
  ;;; @param op-ptr Operation* as uptr
  ;;; @param index Argument index as int
  ;;; @return BlockArgument (Value*) as uptr
  (define mlir-operation-get-block-argument
    (foreign-procedure "mlir_operation_get_block_argument" (uptr int) uptr))

  ;;===--------------------------------------------------------------------===;;
  ;; Operation Traversal
  ;;===--------------------------------------------------------------------===;;

  ;;; @brief Walk all operations in the tree, calling callback for each
  ;;; @param op-ptr Root Operation* as uptr
  ;;; @param callback Scheme procedure taking one argument (Operation* as uptr)
  ;;; @note Callback is passed as scheme-object (GC-tracked), NOT uptr
  (define mlir-operation-walk
    (foreign-procedure "mlir_operation_walk" (uptr scheme-object) void))

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
  ;;; @param type-ptr Type* as uptr
  ;;; @return 1 if ranked tensor, 0 otherwise
  (define mlir-type-is-ranked-tensor
    (foreign-procedure "mlir_type_is_ranked_tensor" (uptr) int))

  ;;; @brief Get the element type of a shaped type (tensor, memref, etc.)
  ;;; @param type-ptr Type* as uptr
  ;;; @return Element Type* as uptr
  (define mlir-type-get-element-type
    (foreign-procedure "mlir_type_get_element_type" (uptr) uptr))

  ;;; @brief Get the shape dimensions of a shaped type
  ;;; @param type-ptr Type* as uptr
  ;;; @return Scheme list of integers representing shape (e.g., '(1 3 224 224))
  ;;; @note Returns a Scheme object (list), NOT an uptr pointer
  (define mlir-type-get-shape
    (foreign-procedure "mlir_type_get_shape" (uptr) scheme-object))

  ;;; @brief Get the rank (number of dimensions) of a shaped type
  ;;; @param type-ptr Type* as uptr
  ;;; @return Rank as int
  (define mlir-type-get-rank
    (foreign-procedure "mlir_type_get_rank" (uptr) int))

  ;;; @brief Create a new type with specified memory space attribute
  ;;; @param type-ptr Type* as uptr
  ;;; @param space Memory space identifier as int
  ;;; @return New Type* with updated memory space as uptr
  (define mlir-type-set-memory-space
    (foreign-procedure "mlir_type_set_memory_space" (uptr int) uptr))

  ;;; @brief Convert a tensor type to device memory space
  ;;; @param type-ptr Type* as uptr
  ;;; @return New Type* in device memory space as uptr
  (define mlir-tensor-type-in-device-space
    (foreign-procedure "mlir_tensor_type_in_device_space" (uptr) uptr))

  ;;; @brief Get the type of a value
  ;;; @param value-ptr Value* as uptr
  ;;; @return Type* as uptr
  (define mlir-value-get-type
    (foreign-procedure "mlir_value_get_type" (uptr) uptr))

  ;;; @brief Get the encoding attribute of a RankedTensorType (0 if none)
  ;;; @param type-ptr Type* as uptr
  ;;; @return Attribute* as uptr, or 0 if not a ranked tensor or has no encoding
  (define mlir-type-get-encoding
    (foreign-procedure "mlir_type_get_encoding" (uptr) uptr))

  ;;===--------------------------------------------------------------------===;;
  ;; Utility Functions
  ;;===--------------------------------------------------------------------===;;

  ;;; @brief Get the HipSR context argument from a function operation
  ;;; @param op-ptr Function Operation* as uptr
  ;;; @return BlockArgument (Value*) representing the context argument as uptr
  (define mlir-get-hipsr-context-arg
    (foreign-procedure "mlir_get_hipsr_context_arg" (uptr) uptr))

  ;;===--------------------------------------------------------------------===;;
  ;; Dialect Conversion Framework Primitives
  ;;===--------------------------------------------------------------------===;;

  ;;; @brief Create a new TypeConverter for dialect conversion
  ;;; @return TypeConverter* as uptr
  (define mlir-create-type-converter
    (foreign-procedure "mlir_create_type_converter" () uptr))

  ;;; @brief Destroy a TypeConverter
  ;;; @param converter-ptr TypeConverter* as uptr
  (define mlir-destroy-type-converter
    (foreign-procedure "mlir_destroy_type_converter" (uptr) void))

  ;;; @brief Add a Scheme type-conversion callback to a TypeConverter
  ;;; @param converter-ptr TypeConverter* as uptr
  ;;; @param callback Scheme procedure: (lambda (type-uptr) -> type-uptr or #f)
  ;;; @note Returns #f from callback means "not handled"; return a Type* uptr to convert
  (define mlir-type-converter-add-conversion
    (foreign-procedure "mlir_type_converter_add_conversion" (uptr scheme-object) void))

  ;;; @brief Check if a single type is legal according to a TypeConverter
  ;;; @param converter-ptr TypeConverter* as uptr
  ;;; @param type-ptr Type* as uptr
  ;;; @return 1 if legal, 0 otherwise
  (define mlir-type-converter-is-legal-type
    (foreign-procedure "mlir_type_converter_is_legal_type" (uptr uptr) int))

  ;;; @brief Check if all operand/result types of an op are legal
  ;;; @param converter-ptr TypeConverter* as uptr
  ;;; @param op-ptr Operation* as uptr
  ;;; @return 1 if legal, 0 otherwise
  (define mlir-type-converter-is-legal
    (foreign-procedure "mlir_type_converter_is_legal" (uptr uptr) int))

  ;;; @brief Check if a func op's signature is legal according to a TypeConverter
  ;;; @param converter-ptr TypeConverter* as uptr
  ;;; @param func-op-ptr func::FuncOp Operation* as uptr
  ;;; @return 1 if legal, 0 otherwise
  (define mlir-type-converter-is-signature-legal
    (foreign-procedure "mlir_type_converter_is_signature_legal" (uptr uptr) int))

  ;;; @brief Add device memory space conversions to a TypeConverter
  ;;; @param converter-ptr TypeConverter* as uptr
  ;;; @note Registers conversions for tensor types to device memory space
  (define mlir-type-converter-add-device-memory-conversions
    (foreign-procedure "mlir_type_converter_add_device_memory_conversions" (uptr) void))

  ;;; @brief Create a ConversionTarget for dialect conversion
  ;;; @param context-ptr MLIRContext* as uptr
  ;;; @return ConversionTarget* as uptr
  (define mlir-create-conversion-target
    (foreign-procedure "mlir_create_conversion_target" (uptr) uptr))

  ;;; @brief Destroy a ConversionTarget
  ;;; @param target-ptr ConversionTarget* as uptr
  (define mlir-destroy-conversion-target
    (foreign-procedure "mlir_destroy_conversion_target" (uptr) void))

  ;;; @brief Mark a dialect illegal by name
  (define mlir-conversion-target-add-illegal-dialect
    (foreign-procedure "mlir_conversion_target_add_illegal_dialect" (uptr string) void))

  ;;; @brief Mark a dialect legal by name
  (define mlir-conversion-target-add-legal-dialect
    (foreign-procedure "mlir_conversion_target_add_legal_dialect" (uptr string) void))

  ;;; @brief Mark a specific op legal by name (overrides dialect-level legality)
  ;;; @param ctx-ptr MLIRContext* as uptr (needed to construct OperationName)
  (define mlir-conversion-target-add-legal-op
    (foreign-procedure "mlir_conversion_target_add_legal_op" (uptr uptr string) void))

  ;;; @brief Mark a specific op dynamically legal with a Scheme callback
  ;;; @param callback Scheme procedure: (lambda (op-uptr) -> bool)
  (define mlir-conversion-target-add-dynamically-legal-op
    (foreign-procedure "mlir_conversion_target_add_dynamically_legal_op"
                       (uptr uptr string scheme-object) void))

  ;;; @brief Mark all unknown ops dynamically legal with a Scheme callback
  ;;; @param callback Scheme procedure: (lambda (op-uptr) -> bool)
  (define mlir-conversion-target-mark-unknown-ops-dynamically-legal
    (foreign-procedure "mlir_conversion_target_mark_unknown_ops_dynamically_legal"
                       (uptr scheme-object) void))

  ;;; @brief Mark all ONNX dialect operations as illegal in conversion target
  ;;; @param target-ptr ConversionTarget* as uptr
  ;;; @note Operations marked illegal must be converted during dialect conversion
  (define mlir-conversion-target-add-illegal-onnx
    (foreign-procedure "mlir_conversion_target_add_illegal_onnx" (uptr) void))

  ;;; @brief Mark all HipSR dialect operations as legal in conversion target
  ;;; @param target-ptr ConversionTarget* as uptr
  ;;; @note Operations marked legal can remain unchanged during conversion
  (define mlir-conversion-target-add-legal-hipsr
    (foreign-procedure "mlir_conversion_target_add_legal_hipsr" (uptr) void))

  ;;; @brief Mark common operations (func, return, etc.) as legal
  ;;; @param target-ptr ConversionTarget* as uptr
  (define mlir-conversion-target-add-legal-common-ops
    (foreign-procedure "mlir_conversion_target_add_legal_common_ops" (uptr) void))

  ;;; @brief Mark func operations as dynamically legal (checked per-operation)
  ;;; @param target-ptr ConversionTarget* as uptr
  ;;; @param converter-ptr TypeConverter* as uptr (used for signature checks)
  ;;; @note Dynamic legality checks if function signatures use only legal types
  (define mlir-conversion-target-add-dynamically-legal-func
    (foreign-procedure "mlir_conversion_target_add_dynamically_legal_func"
                       (uptr uptr) void))

  ;;; @brief Mark unknown operations as legal if nested in legal contexts
  ;;; @param target-ptr ConversionTarget* as uptr
  ;;; @note Allows unknown ops inside legal regions without conversion
  (define mlir-conversion-target-mark-unknown-ops-nested-legal
    (foreign-procedure "mlir_conversion_target_mark_unknown_ops_nested_legal" (uptr) void))

  ;;; @brief Create a RewritePatternSet for dialect conversion
  ;;; @param context-ptr MLIRContext* as uptr
  ;;; @return RewritePatternSet* as uptr
  (define mlir-create-rewrite-pattern-set
    (foreign-procedure "mlir_create_rewrite_pattern_set" (uptr) uptr))

  ;;; @brief Destroy a RewritePatternSet
  ;;; @param patterns-ptr RewritePatternSet* as uptr
  (define mlir-destroy-rewrite-pattern-set
    (foreign-procedure "mlir_destroy_rewrite_pattern_set" (uptr) void))

  ;;; @brief Apply full dialect conversion to a module
  ;;; @param module-ptr Module Operation* as uptr
  ;;; @param target-ptr ConversionTarget* as uptr
  ;;; @param patterns-ptr RewritePatternSet* as uptr
  ;;; @return 0 on success, non-zero on failure
  (define mlir-apply-full-conversion
    (foreign-procedure "mlir_apply_full_conversion"
                       (uptr uptr uptr) int))

  ;;===--------------------------------------------------------------------===;;
  ;; Dialect Conversion Helpers
  ;;===--------------------------------------------------------------------===;;

  ;;; @brief Add Cast operation conversion patterns
  ;;; @param patterns-ptr RewritePatternSet* as uptr
  ;;; @param converter-ptr TypeConverter* as uptr
  ;;; @param context-ptr MLIRContext* as uptr
  ;;; @note Populates patterns for converting hipsr.cast operations
  (define mlir-populate-cast-conversion-patterns
    (foreign-procedure "mlir_populate_cast_conversion_patterns"
                       (uptr uptr uptr) void))

  ;;; @brief Add Return operation conversion patterns
  ;;; @param patterns-ptr RewritePatternSet* as uptr
  ;;; @param converter-ptr TypeConverter* as uptr
  ;;; @param context-ptr MLIRContext* as uptr
  ;;; @note Populates patterns for converting func.return operations
  (define mlir-populate-return-conversion-patterns
    (foreign-procedure "mlir_populate_return_conversion_patterns"
                       (uptr uptr uptr) void))

  ;;; @brief Add function type conversion patterns
  ;;; @param patterns-ptr RewritePatternSet* as uptr
  ;;; @param converter-ptr TypeConverter* as uptr
  ;;; @note Populates patterns for converting function signatures
  (define mlir-populate-func-type-conversion-pattern
    (foreign-procedure "mlir_populate_func_type_conversion_pattern"
                       (uptr uptr) void))

  ;;; @brief Erase dead operations with no values
  ;;; @param module-ptr Module Operation* as uptr
  ;;; @note Cleanup pass to remove operations marked as dead during conversion
  (define mlir-erase-dead-novalue-ops
    (foreign-procedure "mlir_erase_dead_novalue_ops" (uptr) void))

  ;;; @brief Rewire placeholder operation inputs after conversion
  ;;; @param module-ptr Module Operation* as uptr
  ;;; @note Connects placeholder operations to their actual operands
  (define mlir-rewire-placeholder-inputs
    (foreign-procedure "mlir_rewire_placeholder_inputs" (uptr) void))

  ;;===--------------------------------------------------------------------===;;
  ;; IR Construction
  ;;===--------------------------------------------------------------------===;;

  ;;; @brief Create an MLIR operation at the current rewriter insertion point.
  ;;; @param rewriter Rewriter* as uptr (passed explicitly by pattern callback)
  ;;; @param loc-op   Operation* whose location is used for the new op
  ;;; @param op-name  Operation name string (e.g., "hipsr.cast")
  ;;; @param operands Scheme list of Value* uptrs
  ;;; @param result-types Scheme list of Type* uptrs
  ;;; @return Created Operation* as uptr
  (define mlir-build-op
    (foreign-procedure "mlir_build_op"
                       (uptr uptr string scheme-object scheme-object) uptr))

  ;;; @brief Set rewriter insertion point to immediately before an operation.
  (define mlir-set-insertion-point-before
    (foreign-procedure "mlir_set_insertion_point_before" (uptr uptr) void))

  ;;; @brief Set rewriter insertion point to the end of a block.
  (define mlir-set-insertion-point-to-block-end
    (foreign-procedure "mlir_set_insertion_point_to_block_end" (uptr uptr) void))

  ;;; @brief Get the i-th region of an operation. Returns Region* as uptr.
  (define mlir-op-get-region
    (foreign-procedure "mlir_op_get_region" (uptr int) uptr))

  ;;; @brief Create a block in a region with given arg types. Sets IP to its end.
  ;;; @param rewriter Rewriter* as uptr
  ;;; @param region   Region* as uptr
  ;;; @param arg-types Scheme list of Type* uptrs
  ;;; @return Block* as uptr
  (define mlir-region-create-block
    (foreign-procedure "mlir_region_create_block" (uptr uptr scheme-object) uptr))

  ;;; @brief Get the i-th argument of a block as a Value* uptr.
  (define mlir-block-get-argument
    (foreign-procedure "mlir_block_get_argument" (uptr int) uptr))

  ;;; @brief Get the shape::ShapeType from an MLIRContext.
  (define mlir-get-shape-shape-type
    (foreign-procedure "mlir_get_shape_shape_type" (uptr) uptr))

  ;;; @brief Set an integer attribute on an operation
  ;;; @param op-ptr Operation* as uptr
  ;;; @param attr-name Attribute name as string
  ;;; @param value Integer value
  (define mlir-operation-set-attr
    (foreign-procedure "mlir_operation_set_attr" (uptr string iptr) void))

  ;;; @brief Set the i-th operand of an operation to a new value
  ;;; @param op-ptr Operation* as uptr
  ;;; @param index Operand index as int
  ;;; @param value New Value* as uptr
  (define mlir-operation-set-operand
    (foreign-procedure "mlir_operation_set_operand" (uptr int uptr) void))

  ;;; @brief Check whether all results of an operation have no uses
  ;;; @param op-ptr Operation* as uptr
  ;;; @return 1 if use-empty, 0 otherwise
  (define mlir-operation-use-empty
    (foreign-procedure "mlir_operation_use_empty" (uptr) int))

  ;;; @brief Get the number of DPS init (outs) operands of an operation
  ;;; @param op-ptr Operation* as uptr
  ;;; @return Count, or 0 if not a DPS op
  (define mlir-operation-num-dps-inits
    (foreign-procedure "mlir_operation_num_dps_inits" (uptr) int))

  ;;; @brief Get the Value* of the i-th DPS init (outs) operand
  ;;; @param op-ptr Operation* as uptr
  ;;; @param index Init index as int
  ;;; @return Value* as uptr, or 0 if out of range
  (define mlir-operation-get-dps-init-value
    (foreign-procedure "mlir_operation_get_dps_init_value" (uptr int) uptr))

  ;;===--------------------------------------------------------------------===;;
  ;; Pattern Rewriting
  ;;===--------------------------------------------------------------------===;;

  ;;; @brief Replace an operation with a value. Takes explicit rewriter.
  ;;; @param rewriter Rewriter* as uptr
  ;;; @param old-op   Operation* to replace as uptr
  ;;; @param new-val  Replacement Value* as uptr
  (define mlir-replace-op
    (foreign-procedure "mlir_replace_op" (uptr uptr uptr) int))

  ;;; @brief Erase an operation. Takes explicit rewriter.
  ;;; @param rewriter Rewriter* as uptr
  ;;; @param op       Operation* to erase as uptr
  (define mlir-erase-op
    (foreign-procedure "mlir_erase_op" (uptr uptr) int))

  ;;; @brief Erase an operation directly without a rewriter.
  ;;; For post-pass cleanup outside a ConversionPattern callback.
  (define mlir-op-erase
    (foreign-procedure "mlir_op_erase" (uptr) void))

  ;;; @brief Notify pattern matching system of a match failure
  ;;; @param op-ptr Operation* that failed to match as uptr
  ;;; @param reason Failure reason as string
  ;;; @note Used for debugging pattern matching failures
  (define mlir-notify-match-failure
    (foreign-procedure "mlir_notify_match_failure" (uptr string) void))

  ;;===--------------------------------------------------------------------===;;
  ;; Pattern Registration
  ;;===--------------------------------------------------------------------===;;

  ;;; @brief Register a Scheme-defined conversion pattern
  ;;; @param patterns-ptr RewritePatternSet* as uptr
  ;;; @param op-name Operation name to match (e.g., "onnx.Cast")
  ;;; @param callback Scheme procedure for pattern rewriting
  ;;; @param type-converter-ptr TypeConverter* as uptr
  ;;; @note Callback signature: (lambda (op operands-ref rewriter type-converter) ...)
  ;;;       - op: Operation* being converted (uptr)
  ;;;       - operands-ref: ValueArrayRef* pointer (uptr)
  ;;;       - rewriter: ConversionPatternRewriter* (uptr)
  ;;;       - type-converter: TypeConverter* (uptr)
  ;;;       Returns: #t on successful rewrite, #f on match failure
  ;;; @note Callback is passed as scheme-object (GC-tracked) and locked with Slock_object
  ;;; @note Mirrors C++ OpConversionPattern::matchAndRewrite signature
  (define mlir-register-conversion-pattern
    (foreign-procedure "mlir_register_conversion_pattern"
                       (uptr string scheme-object uptr) void))

  ;;===--------------------------------------------------------------------===;;
  ;; ValueArrayRef Accessors
  ;;===--------------------------------------------------------------------===;;

  ;;; @brief Get the size of a ValueArrayRef
  ;;; @param ref-ptr ValueArrayRef* as uptr
  ;;; @return Number of elements
  (define (value-array-ref-size ref-ptr)
    (let ([ptr (make-ftype-pointer ValueArrayRef ref-ptr)])
      (ftype-ref ValueArrayRef (size) ptr)))

  ;;; @brief Get Value* at index from ValueArrayRef
  ;;; @param ref-ptr ValueArrayRef* as uptr
  ;;; @param index Zero-based index
  ;;; @return Value* as uptr
  (define (value-array-ref-at ref-ptr index)
    (let* ([ptr (make-ftype-pointer ValueArrayRef ref-ptr)]
           [data-ptr (ftype-ref ValueArrayRef (data) ptr)]
           [offset (* index 8)])  ; Assuming 64-bit pointers
      ;; Read pointer at offset
      (foreign-ref 'uptr data-ptr offset)))

  ;;===--------------------------------------------------------------------===;;
  ;; Resource Management
  ;;===--------------------------------------------------------------------===;;

  ;;; @brief RAII-style resource management via dynamic-wind
  ;;; @syntax (with-raii ((var ctor dtor) ...) body ...)
  ;;; Each resource is created by ctor, bound to var, and destroyed by (dtor var)
  ;;; on exit — whether normal, exception, or continuation escape.
  (define-syntax with-raii
    (syntax-rules ()
      [(_ () body ...)
       (begin body ...)]
      [(_ ((val ctor dtor) rest ...) body ...)
       (let ([val ctor])
         (dynamic-wind
           void
           (lambda () (with-raii (rest ...) body ...))
           (lambda () (dtor val))))]))

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

) ;; end library (mlir ffi)
