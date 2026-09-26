(library (rime files __is-directory)
  (export is-directory?)
  (import (rime rime-0)
          (rnrs (6))
          (rnrs eval (6))
          )
  (cond-expand
   [(library (chezscheme))
    (define is-directory? (eval 'file-directory? (environment '(chezscheme))))]
   [(library (guile))
    (define stat (eval 'stat (environment '(guile))))
    (define stat:type (eval 'stat:type (environment '(guile))))
    (define (is-directory? directory-name)
      (eq? (stat:type (stat directory-name)) 'directory))]
   [else
    (define (is-directory? diretory-name)
      (raise
       (condition
        (make-error)
        (make-who-condition 'is-directory?)
        (make-message-condition "not implemented")
        )))])  )
