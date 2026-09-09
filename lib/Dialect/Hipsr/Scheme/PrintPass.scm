;; Pure Scheme MLIR Pass - Print Operations
;; This is a complete MLIR pass written entirely in Scheme
;; Demonstrates the MLIR logging API with different verbosity levels

(import (rnrs) (rime))

;; Entry point called by C++ - receives the module operation
(define (run-pass module-op)
  (mlir-log-info "Starting Pure Scheme MLIR Pass")
  (mlir-log-debug (format "Module: ~a" (mlir-operation-name module-op)))

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
          (format "Operation: ~s~a~a"
                  name
                  (if (null? operands)
                      ""
                      (format " | Operands[~a]: ~a"
                              num-operands
                              (string-join (map number->string operands) ", ")))
                  (if (null? results)
                      ""
                      (format " | Results[~a]: ~a"
                              num-results
                              (string-join (map number->string results) ", "))))))))

  (mlir-log-info "Completed Pure Scheme MLIR Pass"))
