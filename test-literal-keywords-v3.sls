#!r6rs
;; Minimal test case V3: Exclude = from rnrs, then define it

(library (test-literal-keywords-v3)
  (export test-macro = : ->)
  (import (except (rnrs) =))  ;; <- Exclude R6RS =

  ;; Now define our keywords
  (define-syntax = (lambda (x) (syntax-violation 'keyword "misplaced" x)))
  (define-syntax : (lambda (x) (syntax-violation 'keyword "misplaced" x)))
  (define-syntax -> (lambda (x) (syntax-violation 'keyword "misplaced" x)))

  ;; Use them as literals in syntax-case
  (define-syntax test-macro
    (lambda (x)
      (syntax-case x (= : ->)
        [(_ lhs = rhs)
         #'(list 'assign 'lhs 'rhs)]

        [(_ name : type)
         #'(list 'typed 'name 'type)]

        [(_ from -> to)
         #'(list 'arrow 'from 'to)]))))
