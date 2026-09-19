(import (chezscheme))

;; Minimal test: does :match keyword matching work?
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

;; Test 1: should match first pattern
(display "Test 1: ")
(display (test-keyword test1 :match stuff))
(newline)

;; Test 2: should match second pattern  
(display "Test 2: ")
(display (test-keyword test1 stuff))
(newline)
