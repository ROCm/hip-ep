#!/usr/bin/env scheme-script
(import (rnrs)
        (for (rename (rime loop) (:with :rime-with)) expand)
        (only (chezscheme) format))

;; GOAL: Understand what rime loop actually returns

(format #t "=== Test 1: :break with value ===\n")
(let ([result (loop :for i :from 1 :to 5
                    :when (= i 3)
                    :break 'found-three)])
  (format #t "Result: ~a\n" result)
  (format #t "Is truthy? ~a\n" (if result "YES" "NO")))

(format #t "\n=== Test 2: No break (loop completes) ===\n")
(let ([result (loop :initially := #f 
                    :for i :from 1 :to 5
                    :when (= i 99)
                    :break 'never-happens)])
  (format #t "Result: ~a\n" result)
  (format #t "Is truthy? ~a\n" (if result "YES" "NO")))

(format #t "\n=== Test 3: Nested loop - inner breaks ===\n")
(let ([result (loop :for i :from 1 :to 3
                    :rime-with inner := (loop :for j :from 1 :to 3
                                              :when (= j 2)
                                              :break 'found-j2)
                    :do (format #t "  i=~a inner=~a\n" i inner))])
  (format #t "Result: ~a\n" result))

(format #t "\n=== Test 4: Nested loop - find first match ===\n")
(let ([result (loop :for i :from 1 :to 3
                    :break i :if (loop :for j :from 1 :to 3
                                       :when (= j 2)
                                       :break #t))])
  (format #t "Result: ~a\n" result))
