#!r6rs
(library (test keyword-library)
  (export test-keyword)
  (import (rnrs (6)))

  (define-syntax test-keyword
    (lambda (stx)
      (syntax-case stx (:match)
        [(_ fname :match . rest)
         (begin
           (display "MATCHED with :match keyword\n")
           #'(quote matched))]
        
        [(_ fname . rest)
         (begin
           (display "MATCHED without :match keyword\n")
           #'(quote fallback))])))
)
