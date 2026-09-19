#!r6rs
;; Test case: Use the macro from another file
(import (rnrs)
        (test-literal-keywords))

;; Test 1: Use macro with = literal
(display "Test 1: ")
(display (test-macro x = 42))
(newline)
;; Expected: (assign x 42)

;; Test 2: Use macro with : literal
(display "Test 2: ")
(display (test-macro y : Int))
(newline)
;; Expected: (typed y Int)

;; Test 3: Use macro with -> literal
(display "Test 3: ")
(display (test-macro String -> Bool))
(newline)
;; Expected: (arrow String Bool)
