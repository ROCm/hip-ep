#!r6rs
(library (rime rime-98 __get-environment-variable)
  (export get-environment-variable)
  (import (rnrs (6))
          (rnrs eval (6))
          (for (rime rime-0) expand))
  (cond-expand
   [(library (guile))
    (define get-environment-variable (eval 'getenv (environment '(guile))))]
   [(library (chezscheme))
    (define get-environment-variable (eval 'getenv (environment '(chezscheme))))]
   [else (define (getenv var)
           (raise-continuable
            (condition
             (make-warning)
             (make-message-condition
              "getenv is not implemented")))
           #f)]))
