#!/usr/bin/env scheme-script
(import (rnrs)
        (for (rename (rime loop) (:with :rime-with)) expand)
        (only (chezscheme) format))

;; Test just the inner loop

(define (test-inner search-value list)
  (format #t "Searching for ~a in ~a:\n" search-value list)
  (let ([result (loop :for item :in list
                      :when (eq? item search-value)
                      :break #t
                      :finally #f)])
    (format #t "  Result: ~a\n" result)
    result))

(test-inner 'e '(d e f))  ; Should return #t
(test-inner 'a '(d e f))  ; Should return #f
