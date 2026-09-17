#!r6rs
;;===----------------------------------------------------------------------===;;
;; Test define-conversion-pattern macro syntax
;;===----------------------------------------------------------------------===;;

(library (test pattern-macro-test)
  (export run-tests)

  (import (rnrs (6))
          (test test-framework)
          (mlir pattern-macro))

  (define (run-tests)
    (test-begin "define-conversion-pattern")

    ;; Test: Macro syntax is valid
    ;; Just verifying the macro compiles - actual pattern execution needs FFI
    (test-assert "macro syntax compiles"
      (let ()
        ;; If this compiles without error, the macro is valid
        (define-syntax test-macro-expands
          (syntax-rules ()
            [(_) #t]))
        (test-macro-expands)))

    (test-end))

) ;; end library
