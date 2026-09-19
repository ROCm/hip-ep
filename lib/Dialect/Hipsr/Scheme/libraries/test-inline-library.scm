(import (chezscheme))

(library (test inline)
  (export test-keyword)
  (import (rnrs (6)))

  (define-syntax test-keyword
    (lambda (stx)
      (syntax-case stx (:match)
        [(_ fname :match . rest)
         #'(quote matched-with-match)]
        
        [(_ fname . rest)
         #'(quote matched-without-match)]))))

(import (test inline))

;; Test: should match first pattern
(display "Test: ")
(display (test-keyword test1 :match stuff))
(newline)
