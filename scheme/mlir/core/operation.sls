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

  ;; Return the registered name of the operation (e.g. "onnx.Cast").
  ;; op: Operation* uptr
  ;; Returns: interned string; valid for the lifetime of the operation
  (define mlir-operation-name
    (foreign-procedure "mlir_operation_get_name" (uptr) string))

  ;; Return the MLIRContext that owns the operation.
  ;; op: Operation* uptr
  ;; Returns: MLIRContext* uptr
  (define mlir-operation-get-context
    (foreign-procedure "mlir_operation_get_context" (uptr) uptr))

  ;; Return the number of operands of the operation.
  ;; op: Operation* uptr
  ;; Returns: operand count as iptr (signed platform integer)
  (define mlir-operation-num-operands
    (foreign-procedure "mlir_operation_num_operands" (uptr) iptr))

  ;; Return the number of results of the operation.
  ;; op: Operation* uptr
  ;; Returns: result count as iptr
  (define mlir-operation-num-results
    (foreign-procedure "mlir_operation_num_results" (uptr) iptr))

  ;; Return the i-th operand as an OpOperand* wrapped in a C-API MlirValue.
  ;; op:    Operation* uptr
  ;; index: 0-based operand index (iptr)
  ;; Returns: C-API value handle uptr; 0 if index out of range
  ;; Note: use mlir-operation-get-operand-value to get the raw Value* opaque ptr
  (define mlir-operation-get-operand
    (foreign-procedure "mlir_operation_get_operand" (uptr iptr) uptr))

  ;; Return the i-th result as a C-API MlirValue handle.
  ;; op:    Operation* uptr
  ;; index: 0-based result index (iptr)
  ;; Returns: C-API value handle uptr; 0 if index out of range
  ;; Note: use mlir-operation-get-result-value for the raw Value* opaque ptr
  (define mlir-operation-get-result
    (foreign-procedure "mlir_operation_get_result" (uptr iptr) uptr))

  ;; Return the parent operation, or 0 if at the module root.
  ;; op: Operation* uptr
  ;; Returns: parent Operation* uptr, or 0
  (define mlir-operation-get-parent
    (foreign-procedure "mlir_operation_get_parent" (uptr) uptr))

  ;; Return the i-th operand as a raw Value* opaque pointer.
  ;; op:    Operation* uptr
  ;; index: 0-based operand index (int)
  ;; Returns: Value* opaque ptr uptr; 0 if null or out of range
  (define mlir-operation-get-operand-value
    (foreign-procedure "mlir_operation_get_operand_value" (uptr int) uptr))

  ;; Return the i-th result as a raw Value* opaque pointer.
  ;; op:    Operation* uptr
  ;; index: 0-based result index (int)
  ;; Returns: Value* opaque ptr uptr; 0 if null or out of range
  (define mlir-operation-get-result-value
    (foreign-procedure "mlir_operation_get_result_value" (uptr int) uptr))

  ;; Return the source location of the operation as a Location opaque ptr.
  ;; op: Operation* uptr
  ;; Returns: Location opaque ptr uptr (pass to mlir-emit-error! etc.)
  (define mlir-operation-get-loc
    (foreign-procedure "mlir_operation_get_loc" (uptr) uptr))

  ;; Return the i-th argument of the enclosing func.func, walking up to find it.
  ;; op:    Operation* uptr — any op inside the function
  ;; index: 0-based argument index (int)
  ;; Returns: Value* opaque ptr uptr; 0 if no enclosing func.func or out of range
  (define mlir-operation-get-block-argument
    (foreign-procedure "mlir_operation_get_block_argument" (uptr int) uptr))

  ;; Walk the operation tree in pre-order, calling callback on each op.
  ;; op:       Operation* uptr — root of the walk
  ;; callback: Scheme procedure (lambda (op-uptr) ...) called for each visited op
  ;; The callback is GC-locked for the duration of the walk.
  (define mlir-operation-walk
    (foreign-procedure "mlir_operation_walk" (uptr scheme-object) void))

  ;; Return the named attribute as an opaque Attribute pointer, or 0 if absent.
  ;; op:   Operation* uptr
  ;; name: attribute name string
  ;; Returns: Attribute opaque ptr uptr; 0 if the attribute is not set
  (define mlir-operation-get-attribute
    (foreign-procedure "mlir_operation_get_attribute" (uptr string) uptr))

  ;; Set a named attribute on the operation from an opaque Attribute pointer.
  ;; op:   Operation* uptr
  ;; name: attribute name string
  ;; attr: Attribute opaque ptr uptr (from make-mlir-attribute or mlir-operation-get-attribute)
  (define mlir-operation-set-attribute!
    (foreign-procedure "mlir_operation_set_attribute" (uptr string uptr) void))

  ;; Return #t if the operation has the named attribute, #f otherwise.
  ;; op:   Operation* uptr
  ;; name: attribute name string
  (define (mlir-operation-has-attr? op name)
    (= 1 ((foreign-procedure "mlir_operation_has_attr" (uptr string) int) op name)))

  ;; Typed attribute getters — extract a Scheme value from a named attribute.

  ;; Return the string value of a StringAttr, or "" if absent or wrong type.
  ;; op:   Operation* uptr
  ;; name: attribute name string
  (define %get-string-attr
    (foreign-procedure "mlir_operation_get_string_attr" (uptr string) string))

  ;; Return the integer value of an IntegerAttr, or default-val if absent.
  ;; op:          Operation* uptr
  ;; name:        attribute name string
  ;; default-val: value returned when the attribute is absent (integer-64)
  (define %get-i64-attr
    (foreign-procedure "mlir_operation_get_integer_attr" (uptr string integer-64) integer-64))

  ;; Return a DenseI64ArrayAttr (or ArrayAttr of integers) as a Scheme list of integers.
  ;; op:   Operation* uptr
  ;; name: attribute name string
  ;; Returns: Scheme list of exact integers; '() if absent or wrong type
  (define %get-i64-array-attr
    (foreign-procedure "mlir_operation_get_integer_array_attr" (uptr string) scheme-object))

  ;; Dispatch typed attribute getter by type keyword.
  ;; op:   Operation* uptr
  ;; name: attribute name string
  ;; type: ':string | ':i64 | ':i64-array
  ;; rest: optional default for ':i64 (default 0)
  ;; Raises on unknown type keyword.
  (define (mlir-operation-get-attr op name type . rest)
    (case type
      [(:string)    (%get-string-attr op name)]
      [(:i64)       (%get-i64-attr op name (if (null? rest) 0 (car rest)))]
      [(:i64-array) (%get-i64-array-attr op name)]
      [else (error 'mlir-operation-get-attr "unknown attr type" type)]))

  ;; Replace the i-th operand of the operation with a new value.
  ;; op:    Operation* uptr
  ;; index: 0-based operand index (int)
  ;; value: new Value* opaque ptr uptr
  (define mlir-operation-set-operand
    (foreign-procedure "mlir_operation_set_operand" (uptr int uptr) void))

  ;; Return #t if all results of the operation have no uses (op is dead).
  ;; op: Operation* uptr
  (define (mlir-operation-use-empty? op)
    (= 1 ((foreign-procedure "mlir_operation_use_empty" (uptr) int) op)))

  ;; Return the number of DPS (destination-passing style) init operands.
  ;; op: Operation* uptr — must implement DestinationStyleOpInterface
  ;; Returns: 0 if the op does not implement DPS
  (define mlir-operation-num-dps-inits
    (foreign-procedure "mlir_operation_num_dps_inits" (uptr) int))

  ;; Return the i-th DPS init operand as a Value* opaque ptr.
  ;; op:    Operation* uptr
  ;; index: 0-based init index (int)
  ;; Returns: Value* opaque ptr uptr; 0 if out of range or not a DPS op
  (define mlir-operation-get-dps-init-value
    (foreign-procedure "mlir_operation_get_dps_init_value" (uptr int) uptr))

  ;; Emit an error diagnostic attached to op through MLIR's diagnostic engine.
  ;; Falls back to mlir_log_error when op is 0.
  ;; Does not raise a Scheme exception — callers propagate failure explicitly.
  ;; op:  Operation* uptr (or 0 for unattached diagnostic)
  ;; msg: diagnostic message string
  (define mlir-emit-error!
    (foreign-procedure "mlir_emit_error"   (uptr string) void))

  ;; Emit a warning diagnostic attached to op. Falls back to mlir_log_warning.
  ;; op:  Operation* uptr (or 0)
  ;; msg: diagnostic message string
  (define mlir-emit-warning!
    (foreign-procedure "mlir_emit_warning" (uptr string) void))

  ;; Emit a remark diagnostic attached to op. Falls back to mlir_log_info.
  ;; op:  Operation* uptr (or 0)
  ;; msg: diagnostic message string
  (define mlir-emit-remark!
    (foreign-procedure "mlir_emit_remark"  (uptr string) void))

) ;; end library (mlir core operation)
