#!r6rs
(library (test rime loop upfrom-2-no-end-test)
  (export main)
  (import (rnrs (6))
          (rime unit-test)
          (rime loop)
          )
  (define (main)
    (run-all-tests))

  (define-test
    test-for-upfrom
    (CHECK equal? (loop :for i :upfrom 1
                        :for sym :in '(a b c)
                        :collect (cons i sym))
           '((1 . a) (2 . b) (3 . c)))
    (CHECK equal? (loop :for i :upfrom 1 :by 2
                        :for sym :in '(a b c)
                        :collect (cons i sym))
           '((1 . a) (3 . b) (5 . c)))))
