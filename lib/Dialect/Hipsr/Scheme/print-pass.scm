;; Pure Scheme MLIR Pass - Print Operations
;; This is a complete MLIR pass written entirely in Scheme

;; Entry point called by C++ - receives the module operation
(define (run-pass module-op)
  (display "\n=== Pure Scheme MLIR Pass ===\n")
  (display "Module: ")
  (display (mlir-operation-name module-op))
  (display "\n\n")

  ;; Walk all operations and print details
  (mlir-operation-walk module-op
    (lambda (op)
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

        (display "\n"))))

  (display "\n=== End Pure Scheme Pass ===\n\n"))
