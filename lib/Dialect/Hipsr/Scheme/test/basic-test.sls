#!r6rs
;;===----------------------------------------------------------------------===;;
;; Basic Tests (No FFI Required)
;;===----------------------------------------------------------------------===;;

(library (test basic-test)
  (export run-tests)

  (import (rnrs (6))
          (test test-framework))

  (define (run-tests)
    (test-begin "basic")

    ;; Test test framework itself
    (test-equal "1 + 1 = 2"
      (+ 1 1)
      2)

    (test-assert "list? works"
      (list? '(1 2 3)))

    (test-assert "procedure? works"
      (procedure? (lambda (x) x)))

    (test-equal "string-append works"
      (string-append "hello" " " "world")
      "hello world")

    (test-end))

) ;; end library
