#!r6rs
(library (test rime loop list-test)
  (export main)
  (import (rnrs (6))
          (rime unit-test)
          (rime loop)
          )
  (define-test test:for-in-list-void
    (CHECK equal? (loop :for i :in '(a b c)) (if #f 0)))
  (define-test test:for-in-list-do
    (CHECK equal?
           (loop :for i :in '(a b c)
                 :do (map display (list ">>>>>> i=" i "\n"))
                 :do (map display (list ">>>>>> (symbol->string i) ;; => " (symbol->string i) "\n")))
           (if #f 0)))
  (define (main)
    (run-all-tests)))
