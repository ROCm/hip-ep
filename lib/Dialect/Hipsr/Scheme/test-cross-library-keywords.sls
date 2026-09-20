#!r6rs
;; Minimal reproduction of cross-library keyword matching issue

;; Library A: Defines keywords and exports a macro
(library (test-lib-a)
  (export my-macro :foo :bar)
  (import (rnrs) (for (test-lib-b) expand))

  ;; Define keywords
  (define-syntax :foo (lambda (x) (syntax-violation 'keyword "misplaced" x)))
  (define-syntax :bar (lambda (x) (syntax-violation 'keyword "misplaced" x)))

  ;; Macro that calls parser from lib-b
  (define-syntax my-macro
    (lambda (stx)
      ;; How do I pass keywords to lib-b so syntax-case literal matching works there?
      (parse-it stx))))

;; Library B: Parser with syntax-case literal matching
(library (test-lib-b)
  (export parse-it)
  (import (rnrs))

  (define (parse-it stx)
    ;; This syntax-case needs to match :foo and :bar from lib-a
    ;; But they're not in scope here!
    (syntax-case stx (:foo :bar)  ; <-- ERROR: :foo and :bar unbound
      [(_ :foo x) #'(define result 'foo-case)]
      [(_ :bar x) #'(define result 'bar-case)]
      [_ #'(define result 'no-match)])))

;; Usage
(library (test-use)
  (export test)
  (import (rnrs) (test-lib-a))

  (my-macro :foo 42)  ; Should expand to (define result 'foo-case)

  (define test result))
