;; Pure Scheme MLIR Pass - Print Operations
;; This is a complete MLIR pass written entirely in Scheme
;; Demonstrates the MLIR logging API with different verbosity levels
;; Extensively uses rime loop macro for functional iteration

(import (chezscheme))
;; TODO: Re-enable rime loop when compatibility issues are resolved
;; (import (rime loop))

;; Helper: build list 0..n-1
(define (iota n)
  (let loop ((i 0) (acc '()))
    (if (>= i n)
        (reverse acc)
        (loop (+ i 1) (cons i acc)))))

;; Helper: join strings
(define (string-join strs sep)
  (if (null? strs)
      ""
      (let loop ((rest (cdr strs)) (acc (car strs)))
        (if (null? rest)
            acc
            (loop (cdr rest)
                  (string-append acc sep (car rest)))))))

;; Helper: format operands list
(define (format-operands op num-operands)
  (if (zero? num-operands)
      ""
      (let ((operands (map (lambda (i) (mlir-operation-get-operand op i))
                           (iota num-operands))))
        (format " | Operands[~a]: ~a"
                num-operands
                (string-join (map number->string operands) ", ")))))

;; Helper: format results list
(define (format-results op num-results)
  (if (zero? num-results)
      ""
      (let ((results (map (lambda (i) (mlir-operation-get-result op i))
                          (iota num-results))))
        (format " | Results[~a]: ~a"
                num-results
                (string-join (map number->string results) ", ")))))

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
