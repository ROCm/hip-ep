;; MLIR-Scheme Bridge Library
;; Provides Scheme functions for working with MLIR operations

;; Format an operation as a human-readable string
(define (format-operation op-name num-operands num-results generic-form)
  (string-append
    "Operation: \"" op-name "\"\n"
    "  Operands: " (number->string num-operands) "\n"
    "  Results: " (number->string num-results) "\n"
    "  Generic form: " generic-form "\n\n"))

;; Get operation signature
(define (operation-signature op-name num-operands num-results)
  (string-append op-name " : "
    (number->string num-operands) " -> "
    (number->string num-results)))

;; Check if operation is a cast
(define (is-cast-op? op-name)
  (string=? op-name "onnx.Cast"))

;; More utilities can be added here for pattern matching, etc.
