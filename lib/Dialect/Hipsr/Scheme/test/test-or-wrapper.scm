#!/usr/bin/env scheme-script
(import (rnrs)
        (for (rename (rime loop) (:with :rime-with)) expand)
        (only (chezscheme) format))

;; Test using (or ... #f) to convert #<void> to #f

(format #t "Test 1 - Should return #t (found):\n")
(let ([result1 (or (loop :for item :in '(d e f)
                         :when (eq? item 'e)
                         :break #t)
                   #f)])
  (format #t "  Result: ~a\n" result1))

(format #t "\nTest 2 - Should return #f (not found):\n")
(let ([result2 (or (loop :for item :in '(d e f)
                         :when (eq? item 'z)
                         :break #t)
                   #f)])
  (format #t "  Result: ~a\n" result2))
