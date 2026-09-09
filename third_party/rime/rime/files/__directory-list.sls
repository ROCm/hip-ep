#!r6rs
(library (rime files __directory-list)
  (export directory-list)
  (import (rime rime-0)
          (rnrs (6))
          (rnrs eval (6))
          )
  (cond-expand
   [(library (chezscheme))
    (define $directory-list (eval 'directory-list (environment '(chezscheme))))
    (define $directory-separator (string ((eval 'directory-separator
                                                (environment '(chezscheme))))))
    (define (directory-list directory-name)
      (map (lambda (f)
             (string-append directory-name $directory-separator f))
           ($directory-list directory-name)))]
   [(library (ice-9 ftw))
    (define scandir (eval 'scandir (environment '(ice-9 ftw))))
    (define $directory-separator "/") ;; TODO
    (define (directory-list directory-name)
      (map
       (lambda (f)
         (string-append directory-name $directory-separator f))
       (scandir directory-name (lambda (name)
                                 (cond
                                  [(string=? name ".") #f]
                                  [(string=? name "..") #f]
                                  [else #t])))))]
   [else
    (define (directory-list diretory-name)
      (raise
       (condition
        (make-error)
        (make-who-condition 'diretory-list)
        (make-message-condition "not implemented")
        )))]))
