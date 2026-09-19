(import (chezscheme) (for (only (chezscheme) syntax->list) expand))

(define-record-type (ast-pattern-expand make-ast-pattern-expand ast-pattern-expand?)
  (fields (mutable function-name)
          (mutable root-var)
          (mutable root-op-name)
          (mutable match)
          (mutable rewrite)
          (mutable where)
          (mutable debug-ast?)
          (mutable debug-matching?)))

(define-syntax debug-parse
  (lambda (stx)
    (define (parse-rest rest ast)
      (display "DEBUG parse-rest called with:")
      (newline)
      (display (syntax->datum rest))
      (newline)
      (syntax-case rest (:debug-ast :debug-matching :match :rewrite :with :where)
        [(fname :match (match-op ...) :rewrite root :with (rewrite-op ...) . where-rest)
         (and (identifier? #'fname)
              (identifier? #'root)
              (not (null? (syntax->list #'(match-op ...))))
              (not (null? (syntax->list #'(rewrite-op ...)))))
         (begin
           (display "MATCHED main pattern!")
           (newline)
           #'(list 'success 'fname 'root))]
        
        [_ 
         (begin
           (display "FELL THROUGH to catch-all!")
           (newline)
           #'(list 'fail))]))
    
    (syntax-case stx ()
      [(_ . rest)
       (parse-rest #'rest (make-ast-pattern-expand #f #f #f '() '() '() #f #f))])))

(display (debug-parse test1 :match ((%out = "test.op" (%in) () : (!t) -> !t)) :rewrite %out :with ((%new = "new.op" (%in) () : (!t) -> !t))))
(newline)
