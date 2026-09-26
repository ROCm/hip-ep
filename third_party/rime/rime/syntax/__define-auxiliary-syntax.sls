#!r6rs
(library (rime syntax __define-auxiliary-syntax)
  (export define-auxiliary-syntax)
  (import (rnrs (6)))
  (define-syntax define-auxiliary-syntax
    (syntax-rules ()
        [(_ name ...)
         (begin
           (define-syntax name
             (lambda (e)
               (syntax-case e ()
                 [(k . v)(syntax-violation 'name "misplaced auxiliary keyword" #'k #'v)]))) ...)
         ])))
