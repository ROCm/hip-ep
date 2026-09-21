#!/usr/bin/env scheme-script
(import (rnrs)
        (for (rename (rime loop) (:with :rime-with)) expand)
        (only (chezscheme) format))

;; Test EXACT pattern from expert

(format #t "Test 1 - Should return #t (found):\n")
(let ([result1 (loop :for item :in '(d e f)
                     :when (eq? item 'e)
                     :break #t
                     :finally #f)])
  (format #t "  Result: ~a\n" result1))

(format #t "\nTest 2 - Should return #f (not found):\n")
(let ([result2 (loop :for item :in '(d e f)
                     :when (eq? item 'z)
                     :break #t
                     :finally #f)])
  (format #t "  Result: ~a\n" result2))
