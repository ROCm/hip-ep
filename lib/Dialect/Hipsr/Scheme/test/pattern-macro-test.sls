#!r6rs
;;===----------------------------------------------------------------------===;;
;; Test define-conversion-pattern macro
;;===----------------------------------------------------------------------===;;

(library (test pattern-macro-test)
  (export run-tests)

  (import (rnrs (6))
          (test test-framework)
          (prefix (test mock-ffi) ffi:)  ; Import mock FFI first
          (mlir pattern-macro))

  ;; Re-export mocked FFI functions at top level so pattern-macro can see them
  (define mlir-operation-name ffi:mlir-operation-name)

  ;; Test: Define patterns using the macro
  (define-conversion-pattern my-test-pattern
    :match "test.op"
    :rewrite (lambda (op operands-ref rewriter type-converter)
               'success))

  (define-conversion-pattern cast-pattern
    :match "onnx.Cast"  
    :rewrite (lambda (op operands-ref rewriter type-converter)
               (list 'cast op operands-ref rewriter type-converter)))

  (define (run-tests)
    (test-begin "define-conversion-pattern")

    ;; Test 1: Macro generates a procedure
    (test-assert "generates a procedure"
      (procedure? my-test-pattern))

    ;; Test 2: Pattern matches correct op name
    (test-equal "matches correct op (op=1 is test.op)"
      (my-test-pattern 1 0 0 0)
      'success)

    ;; Test 3: Pattern returns #f for non-matching op
    (test-equal "returns #f for non-matching op (op=999)"
      (my-test-pattern 999 0 0 0)
      #f)

    ;; Test 4: Pattern calls rewrite with all 4 parameters
    (test-equal "passes all 4 parameters to rewrite"
      (cast-pattern 2 'operands 'rewriter 'type-converter)
      '(cast 2 operands rewriter type-converter))

    (test-end))

) ;; end library
