#!r6rs
;;===----------------------------------------------------------------------===;;
;; Basic Test - Verify test framework works
;;===----------------------------------------------------------------------===;;

(library (test basic-test)
  (export run-tests)

  (import (rnrs (6))
          (test test-framework))

  (define (run-tests)
    (test-begin "basic")

    ;; Single test to verify framework works
    (test-equal "hello test"
      (string-append "hello" " " "world")
      "hello world")

    (test-end))

) ;; end library
