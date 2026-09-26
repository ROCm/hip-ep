#!r6rs
;; Use the test-macro

(import (rnrs)
        (test-debug-ast-minimal))

;; Test 1: With :debug flag
(test-macro :debug my-test "foo" "bar")

;; Test 2: Without :debug flag
(test-macro my-test-2 "baz")

;; Verify
(display "Test 1: ")
(display (dummy-record? my-test))
(newline)

(display "Test 2: ")
(display my-test-2)
(newline)
