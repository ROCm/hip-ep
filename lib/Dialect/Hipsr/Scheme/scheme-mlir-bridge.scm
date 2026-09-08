;; MLIR-Scheme Bridge Library
;; Provides Scheme functions for working with MLIR operations

;; Print operation information
;; Called for each operation during the pass
(define (process-operation op-name num-operands num-results generic-form)
  (display "Operation: \"")
  (display op-name)
  (display "\"\n")
  (display "  Operands: ")
  (display num-operands)
  (display "\n")
  (display "  Results: ")
  (display num-results)
  (display "\n")
  (display "  Generic form: ")
  (display generic-form)
  (display "\n\n"))

;; Pass initialization
;; Called once when the pass starts
(define (pass-initialize module-name)
  (display "\n=== Scheme-based MLIR Printer ===\n")
  (display "Module: ")
  (display module-name)
  (display "\n\n"))

;; Pass finalization
;; Called once when the pass ends
(define (pass-finalize)
  (display "=== End Scheme Printer ===\n\n"))

;; Utility functions below

;; Format an operation as a string (returns string instead of printing)
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
