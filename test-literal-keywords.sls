#!r6rs
;; Minimal test case: Can we use =, :, -> as literals in syntax-case
;; when they're exported from a library?

(library (test-literal-keywords)
  (export test-macro = : ->)
  (import (rnrs))

  ;; Question: Do we need to define these as syntax violations?
  ;; The guide says any symbol can be a literal, but what about
  ;; when the symbol is exported/imported across libraries?

  ;; Attempt 1: DON'T define them, just use as literals
  (define-syntax test-macro
    (lambda (x)
      (syntax-case x (= : ->)  ;; <- Use symbols as literals
        [(_ lhs = rhs)
         #'(list 'assign 'lhs 'rhs)]

        [(_ name : type)
         #'(list 'typed 'name 'type)]

        [(_ from -> to)
         #'(list 'arrow 'from 'to)]))))
