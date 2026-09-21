#!/usr/bin/env scheme-script
(import (rnrs)
        (for (rename (rime loop) (:with :rime-with)) expand)
        (only (chezscheme) format))

;; Test nested loop with :break :if and :finally

(define (test-nested-loop search-value)
  (format #t "\nSearching for ~a:\n" search-value)
  (let ([outer-list '((a b c) (d e f) (g h i))])
    (let ([result
           (loop :for outer-idx :from 0
                 :for inner-list :in outer-list
                 :rime-with inner-result := (loop :for item :in inner-list
                                                   :break #t :if (eq? item search-value)
                                                   :finally #f)
                 :do (format #t "  outer-idx=~a inner-result=~a\n" outer-idx inner-result)
                 :break outer-idx :if inner-result
                 :finally #f)])
      (format #t "  Result: ~a\n" result)
      result)))

;; Test cases
(test-nested-loop 'e)    ; Should find at index 1
(test-nested-loop 'a)    ; Should find at index 0
(test-nested-loop 'z)    ; Should not find, return #f
