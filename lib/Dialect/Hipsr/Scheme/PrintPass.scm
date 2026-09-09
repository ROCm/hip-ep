;; Pure Scheme MLIR Pass - Print Operations
;; This is a complete MLIR pass written entirely in Scheme
;; Demonstrates the MLIR logging API with different verbosity levels
;; Extensively uses rime loop macro for functional iteration

(import (rnrs) (rime) (rime loop))

;; Helper: format operands list using rime loop
(define (format-operands op num-operands)
  (if (zero? num-operands)
      ""
      (format " | Operands[~a]: ~a"
              num-operands
              (loop :for i :from 0 :below num-operands
                    :join-string (number->string (mlir-operation-get-operand op i))
                    :seperator ", "))))

;; Helper: format results list using rime loop
(define (format-results op num-results)
  (if (zero? num-results)
      ""
      (format " | Results[~a]: ~a"
              num-results
              (loop :for i :from 0 :below num-results
                    :join-string (number->string (mlir-operation-get-result op i))
                    :seperator ", "))))

;; Helper: format operation details
(define (format-operation op)
  (let ((name (mlir-operation-name op))
        (num-operands (mlir-operation-num-operands op))
        (num-results (mlir-operation-num-results op)))
    (format "Operation: ~s~a~a"
            name
            (format-operands op num-operands)
            (format-results op num-results))))

;; Entry point called by C++ - receives the module operation
(define (run-pass module-op)
  (mlir-log-info "Starting Pure Scheme MLIR Pass")
  (mlir-log-debug (format "Module: ~a" (mlir-operation-name module-op)))

  ;; Walk all operations and log with trace level
  (mlir-operation-walk module-op
    (lambda (op)
      (mlir-log-trace (format-operation op))))

  (mlir-log-info "Completed Pure Scheme MLIR Pass"))
