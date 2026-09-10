;; Pure Scheme MLIR Pass - Print Operations
;; This is a complete MLIR pass written entirely in Scheme
;; Demonstrates the MLIR logging API with different verbosity levels

;; Import rime loop for functional iteration
(import (rime loop))

;; Helper: join strings
(define (string-join strs sep)
  (if (null? strs)
      ""
      (let loop-inner ((rest (cdr strs)) (acc (car strs)))
        (if (null? rest)
            acc
            (loop-inner (cdr rest)
                        (string-append acc sep (car rest)))))))

;; Helper: format operands list using rime loop
(define (format-operands op num-operands)
  (if (zero? num-operands)
      ""
      (let ((operands (loop ((for i (up-from 0 (to num-operands))))
                         (listing (mlir-operation-get-operand op i)))))
        (format " | Operands[~a]: ~a"
                num-operands
                (string-join (loop ((for operand (in-list operands)))
                               (listing (number->string operand)))
                             ", ")))))

;; Helper: format results list using rime loop
(define (format-results op num-results)
  (if (zero? num-results)
      ""
      (let ((results (loop ((for i (up-from 0 (to num-results))))
                       (listing (mlir-operation-get-result op i)))))
        (format " | Results[~a]: ~a"
                num-results
                (string-join (loop ((for result (in-list results)))
                               (listing (number->string result)))
                             ", ")))))

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
