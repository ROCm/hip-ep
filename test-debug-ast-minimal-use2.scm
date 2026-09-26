#!r6rs
;; Simpler test - just expand and check what happens

(import (rnrs)
        (test-debug-ast-minimal))

;; Test 1: With :debug flag
(test-macro :debug my-test "foo" "bar")

;; Just display it
(display my-test)
(newline)
