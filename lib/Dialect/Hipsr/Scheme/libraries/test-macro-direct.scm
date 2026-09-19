(import (chezscheme) (for (only (chezscheme) syntax->list) expand))

;; Copy the EXACT macro from pattern-macro.sls but standalone
(define-syntax test-macro
  (lambda (stx)
    (define (parse-to-ast whole-stx)
      (syntax-case whole-stx ()
        [(_ . rest)
         (parse-rest #'rest)]))
    
    (define (parse-rest rest)
      (syntax-case rest (:match :rewrite :with)
        [(fname :match (match-op ...) :rewrite root :with (rewrite-op ...) . where-rest)
         (and (identifier? #'fname)
              (identifier? #'root)
              (not (null? (syntax->list #'(match-op ...))))
              (not (null? (syntax->list #'(rewrite-op ...)))))
         #`(define #,#'fname (lambda () 'success))]
        
        [_ (syntax-violation 'test-macro "Pattern doesn't match" rest)]))
    
    (parse-to-ast stx)))

(test-macro test1 :match ((%out = "test.op" (%in) () : (!t) -> !t)) :rewrite %out :with ((%new = "new.op" (%in) () : (!t) -> !t)))

(display "test1 result: ")
(display (test1))
(newline)
