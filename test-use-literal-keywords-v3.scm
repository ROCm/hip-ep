#!r6rs
;; Test case V3
(import (rnrs)
        (test-literal-keywords-v3))

;; This works - literals match in same library
(display "Test 1: ")
(display (test-macro x = 42))
(newline)

(display "Test 2: ")
(display (test-macro y : Int))
(newline)

(display "Test 3: ")
(display (test-macro String -> Bool))
(newline)
