#!r6rs
;; Minimal test case V2: Define keywords as syntax violations before using as literals

(library (test-literal-keywords-v2)
  (export test-macro = : ->)
  (import (rnrs))

  ;; Define keywords as syntax violations (like we currently do)
  (define-syntax = (lambda (x) (syntax-violation 'keyword "misplaced" x)))
  (define-syntax : (lambda (x) (syntax-violation 'keyword "misplaced" x)))
  (define-syntax -> (lambda (x) (syntax-violation 'keyword "misplaced" x)))

  ;; Now use them as literals
  (define-syntax test-macro
    (lambda (x)
      (syntax-case x (= : ->)  ;; <- Use as literals
        [(_ lhs = rhs)
         #'(list 'assign 'lhs 'rhs)]

        [(_ name : type)
         #'(list 'typed 'name 'type)]

        [(_ from -> to)
         #'(list 'arrow 'from 'to)]))))
