#!r6rs
(library (minimal keyword-test)
  (export test-macro =)
  (import (except (rnrs (6)) =))
  
  ;; Define = as auxiliary keyword
  (define-syntax =
    (lambda (x)
      (syntax-violation '= "misplaced auxiliary keyword" x)))
  
  ;; Simple macro that uses = as literal
  (define-syntax test-macro
    (lambda (x)
      (syntax-case x (=)
        [(_ a = b)
         #'(list 'matched 'a 'b)]
        [(_ . rest)
         #'(list 'no-match)]))))
