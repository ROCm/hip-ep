;; Pure Scheme MLIR Pass - Print Operations
;; This is a complete MLIR pass written entirely in Scheme
;; Demonstrates the MLIR logging API with different verbosity levels

;; Import rime loop for functional iteration
(import (rime loop))

;; Helper: format operands list using rime loop
(define (format-operands op num-operands)
  (loop :initially := ""
        :for i :from 0 :to (- num-operands 1)
        :with operand := (mlir-operation-get-operand op i)
        :join-string operand :seperator ", "
        :finally (if (zero? num-operands)
                     :return-value
                     (format " | Operands[~a]: ~a" num-operands :return-value))))

;; Helper: format results list using rime loop
(define (format-results op num-results)
  (loop :initially := ""
        :for i :from 0 :to (- num-results 1)
        :with result := (mlir-operation-get-result op i)
        :join-string result :seperator ", "
        :finally (if (zero? num-results)
                     :return-value
                     (format " | Results[~a]: ~a" num-results :return-value))))

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
