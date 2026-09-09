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

;; High-level Scheme API

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
