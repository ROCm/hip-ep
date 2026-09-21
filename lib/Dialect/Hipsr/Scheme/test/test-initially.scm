#!/usr/bin/env scheme-script
(import (rnrs)
        (for (rename (rime loop) (:with :rime-with)) expand)
        (only (chezscheme) format))

;; Test WITH :initially

(define (test-initially search-value list)
  (format #t "Searching for ~a in ~a (WITH :initially):\n" search-value list)
  (let ([result (loop :initially (result #f)
                      :for item :in list
                      :when (eq? item search-value)
                      :break #t)])
    (format #t "  Result: ~a\n" result)
    result))

(test-initially 'e '(d e f))  ; Should break with #t
(test-initially 'a '(d e f))  ; Should return #f from :initially
