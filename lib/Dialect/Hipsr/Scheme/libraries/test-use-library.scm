(import (chezscheme) (test keyword-library))

;; Test: should match first pattern
(display "Test from library: ")
(display (test-keyword test1 :match stuff))
(newline)
