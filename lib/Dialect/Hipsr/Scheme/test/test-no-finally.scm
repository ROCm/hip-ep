#!/usr/bin/env scheme-script
(import (rnrs)
        (for (rename (rime loop) (:with :rime-with)) expand)
        (only (chezscheme) format))

;; Test WITHOUT :finally to see what happens

(define (test-no-finally search-value list)
  (format #t "Searching for ~a in ~a (NO :finally):\n" search-value list)
  (let ([result (loop :for item :in list
                      :when (eq? item search-value)
                      :break #t)])
    (format #t "  Result: ~a\n" result)
    result))

(test-no-finally 'e '(d e f))  ; Should break with #t
(test-no-finally 'a '(d e f))  ; Should NOT break, returns ?
