#!r6rs
;;===----------------------------------------------------------------------===;;
;; Unit Tests for Pattern Macro (compile-time only, no FFI)
;;===----------------------------------------------------------------------===;;

(library (test pattern-macro-test)
  (export run-tests)

  (import (rnrs (6))
          (test test-framework)
          (mlir pattern-macro))

  ;; Define test patterns at top level
  (define-conversion-pattern test-cast
    :match "onnx.Cast"
    :rewrite (lambda (op operands-ref rewriter type-converter) 'success))

  (define-conversion-pattern test-add
    :match "onnx.Add"
    :rewrite (lambda (op operands-ref rewriter type-converter)
               (+ op operands-ref rewriter type-converter)))

  (define (run-tests)
    (test-begin "pattern-macro")

    ;; Test: Macro expands to a procedure
    (test-assert "macro generates a procedure"
      (procedure? test-cast))

    (test-assert "test-add is a procedure"
      (procedure? test-add))

    ;; Note: Cannot test execution without FFI runtime (mlir-operation-name)

    (test-end))

) ;; end library
