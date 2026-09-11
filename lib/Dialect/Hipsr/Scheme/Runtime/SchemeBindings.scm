;; MLIR-Scheme Bridge Library
;; Provides Scheme access to MLIR IR through foreign function interface

;; Define foreign procedures to call C functions
;; Pass pointers as unsigned-64 (64-bit pointers on this architecture)
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

;; Walk operation tree - callback is (lambda (op) ...)
(define mlir-operation-walk
  (foreign-procedure "mlir_operation_walk" (unsigned-64 scheme-object) void))

;; Logging functions
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

;;===----------------------------------------------------------------------===;;
;; Phase 1: Type System FFI
;;===----------------------------------------------------------------------===;;

;; Check if a Type is a RankedTensorType
(define mlir-type-is-ranked-tensor
  (foreign-procedure "mlir_type_is_ranked_tensor" (unsigned-64) int))

;; Get element type of a tensor type
(define mlir-type-get-element-type
  (foreign-procedure "mlir_type_get_element_type" (unsigned-64) unsigned-64))

;; Get shape of a ranked tensor type (returns list of dimensions)
(define mlir-type-get-shape
  (foreign-procedure "mlir_type_get_shape" (unsigned-64) scheme-object))

;; Get rank of a ranked tensor type
(define mlir-type-get-rank
  (foreign-procedure "mlir_type_get_rank" (unsigned-64) int))

;; Get type from a Value
(define mlir-value-get-type
  (foreign-procedure "mlir_value_get_type" (unsigned-64) unsigned-64))

;;===----------------------------------------------------------------------===;;
;; Phase 2: Operation/Value Navigation FFI
;;===----------------------------------------------------------------------===;;

;; Get parent operation
(define mlir-operation-get-parent
  (foreign-procedure "mlir_operation_get_parent" (unsigned-64) unsigned-64))

;; Get operand Value from operation by index
(define mlir-operation-get-operand-value
  (foreign-procedure "mlir_operation_get_operand_value" (unsigned-64 int) unsigned-64))

;; Get result Value from operation by index
(define mlir-operation-get-result-value
  (foreign-procedure "mlir_operation_get_result_value" (unsigned-64 int) unsigned-64))

;; Get location from operation
(define mlir-operation-get-loc
  (foreign-procedure "mlir_operation_get_loc" (unsigned-64) unsigned-64))

;; Get block argument from parent function
(define mlir-operation-get-block-argument
  (foreign-procedure "mlir_operation_get_block_argument" (unsigned-64 int) unsigned-64))

;;===----------------------------------------------------------------------===;;
;; High-level Scheme API
;;===----------------------------------------------------------------------===;;

;; Process a single operation - gets called from C++ for each operation
(define (process-operation op)
  (let ((name (mlir-operation-name op))
        (num-operands (mlir-operation-num-operands op))
        (num-results (mlir-operation-num-results op)))
    (display "Operation: \"")
    (display name)
    (display "\"")

    ;; Print operand pointers
    (when (> num-operands 0)
      (display " | Operands[")
      (display num-operands)
      (display "]: ")
      (let loop ((i 0))
        (when (< i num-operands)
          (let ((operand (mlir-operation-get-operand op i)))
            (display operand)
            (when (< (+ i 1) num-operands)
              (display ", "))
            (loop (+ i 1))))))

    ;; Print result pointers
    (when (> num-results 0)
      (display " | Results[")
      (display num-results)
      (display "]: ")
      (let loop ((i 0))
        (when (< i num-results)
          (let ((result (mlir-operation-get-result op i)))
            (display result)
            (when (< (+ i 1) num-results)
              (display ", "))
            (loop (+ i 1))))))

    (display "\n")))

;; Pass hooks

(define (pass-initialize module-name)
  (display "\n=== Scheme MLIR Pass ===\n")
  (display "Module: ")
  (display module-name)
  (display "\n\n"))

(define (pass-finalize)
  (display "=== End Scheme Pass ===\n\n"))
