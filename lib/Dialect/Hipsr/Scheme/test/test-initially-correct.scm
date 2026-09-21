#!/usr/bin/env scheme-script
(import (rnrs)
        (for (rename (rime loop) (:with :rime-with)) expand)
        (only (chezscheme) format))

;; Test :initially := syntax

(format #t "=== Test 1: :initially := with accumulator ===\n")
(let ([result (loop :initially := 0
                    :for i :from 1 :to 5
                    := (+ = i))])
  (format #t "Result: ~a (sum of 1..5)\n" result))

(format #t "\n=== Test 2: :initially := #f, break with #t ===\n")
(let ([result (loop :initially := #f
                    :for i :from 1 :to 5
                    :when (= i 3)
                    :break #t)])
  (format #t "Result: ~a (should be #t)\n" result))

(format #t "\n=== Test 3: :initially := #f, no break ===\n")
(let ([result (loop :initially := #f
                    :for i :from 1 :to 5
                    :when (= i 99)
                    :break #t)])
  (format #t "Result: ~a (should be #f)\n" result))
