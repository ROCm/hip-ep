#!r6rs
;;===----------------------------------------------------------------------===;;
;; Framework Test - Verify test framework works
;;===----------------------------------------------------------------------===;;

(library (test framework-test)
  (export run-tests)

  (import (rnrs (6))
          (test test-framework))

  (define (run-tests)
    (test-begin "framework")

    ;; Single test to verify framework works
    (test-equal "hello test"
      (string-append "hello" " " "world")
      "hello world")

    (test-end))

) ;; end library
