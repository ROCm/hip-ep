;; Pure Scheme MLIR Pass - Print Operations
;; This is a complete MLIR pass written entirely in Scheme
;; Demonstrates the MLIR logging API with different verbosity levels

;; Helper: build a list of integers from 0 to n-1
(define (iota n)
  (let loop ((i 0) (acc '()))
    (if (>= i n)
        (reverse acc)
        (loop (+ i 1) (cons i acc)))))

;; Helper: join strings with separator
(define (string-join strs sep)
  (if (null? strs)
      ""
      (let loop ((rest (cdr strs)) (acc (car strs)))
        (if (null? rest)
            acc
            (loop (cdr rest)
                  (string-append acc sep (car rest)))))))

;; Entry point called by C++ - receives the module operation
(define (run-pass module-op)
  (mlir-log-info "Starting Pure Scheme MLIR Pass")
  (mlir-log-debug (string-append "Module: " (mlir-operation-name module-op)))

  ;; Walk all operations and print details
  (mlir-operation-walk module-op
    (lambda (op)
      (let* ((name (mlir-operation-name op))
             (num-operands (mlir-operation-num-operands op))
             (num-results (mlir-operation-num-results op))
             (operands (if (> num-operands 0)
                          (map (lambda (i) (mlir-operation-get-operand op i))
                               (iota num-operands))
                          '()))
             (results (if (> num-results 0)
                         (map (lambda (i) (mlir-operation-get-result op i))
                              (iota num-results))
                         '())))

        (mlir-log-debug
          (string-append "Operation: \"" name "\""
                        (if (null? operands)
                            ""
                            (string-append " | Operands["
                                          (number->string num-operands)
                                          "]: "
                                          (string-join (map number->string operands) ", ")))
                        (if (null? results)
                            ""
                            (string-append " | Results["
                                          (number->string num-results)
                                          "]: "
                                          (string-join (map number->string results) ", "))))))))

  (mlir-log-info "Completed Pure Scheme MLIR Pass"))
