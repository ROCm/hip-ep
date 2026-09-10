;; Pure Scheme MLIR Pass - Print Operations
;; This is a complete MLIR pass written entirely in Scheme
;; Demonstrates the MLIR logging API with different verbosity levels

;; Import rime loop for functional iteration
(import (rime loop))

;; Helper: format operands list using rime loop
(define (format-operands op num-operands)
  (if (zero? num-operands)
      ""
      (let ((operands (loop ((for i (up-from 0 (to num-operands)))
                             (listing operand (mlir-operation-get-operand op i)))
                        => operand)))
        (format " | Operands[~a]: ~a"
                num-operands
                (loop ((for op (in-list operands))
                       (listing str (number->string op)))
                   => (string-join str ", "))))))

;; Helper: format results list using rime loop
(define (format-results op num-results)
  (if (zero? num-results)
      ""
      (let ((results (loop ((for i (up-from 0 (to num-results)))
                            (listing result (mlir-operation-get-result op i)))
                       => result)))
        (format " | Results[~a]: ~a"
                num-results
                (loop ((for res (in-list results))
                       (listing str (number->string res)))
                   => (string-join str ", "))))))

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
