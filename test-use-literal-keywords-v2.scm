#!r6rs
;; Test case V2: Use macro with defined keywords
(import (rnrs)
        (test-literal-keywords-v2))

;; Test 1: Use macro with = literal
(display "Test 1: ")
(display (test-macro x = 42))
(newline)

;; Test 2: Use macro with : literal
(display "Test 2: ")
(display (test-macro y : Int))
(newline)

;; Test 3: Use macro with -> literal
(display "Test 3: ")
(display (test-macro String -> Bool))
(newline)
