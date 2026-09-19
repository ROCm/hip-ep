(import (chezscheme) (for (only (chezscheme) syntax->list) expand))

(define-syntax test-macro
  (lambda (stx)
    ;; Add AST record
    (define-record-type (test-ast make-test-ast test-ast?)
      (fields (mutable fname)))
    
    (define (parse-to-ast whole-stx)
      (syntax-case whole-stx ()
        [(_ . rest)
         (parse-rest #'rest (make-test-ast #f))]))
    
    (define (parse-rest rest ast)
      (syntax-case rest (:match :rewrite :with)
        [(fname :match (match-op ...) :rewrite root :with (rewrite-op ...) . where-rest)
         (and (identifier? #'fname)
              (identifier? #'root)
              (not (null? (syntax->list #'(match-op ...))))
              (not (null? (syntax->list #'(rewrite-op ...)))))
         (begin
           (test-ast-fname-set! ast #'fname)
           #`(define #,(test-ast-fname ast) (lambda () 'success)))]
        
        [_ (syntax-violation 'test-macro "Pattern doesn't match" rest)]))
    
    (parse-to-ast stx)))

(test-macro test1 :match ((%out = "test.op" (%in) () : (!t) -> !t)) :rewrite %out :with ((%new = "new.op" (%in) () : (!t) -> !t)))

(display "test1 result: ")
(display (test1))
(newline)
