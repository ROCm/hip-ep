#!r6rs
(library (test rime loop two-for-test)
  (export main)
  (import (rnrs (6))
          (rime unit-test)
          (rime loop)
          )
  (define-test test:for-ij-from-0-to-3
    (CHECK equal? (loop :for i :from 0 :to 3
                        :for j :from 0 :to 4
                        :collect (list i j))
           '((0 0) (1 1) (2 2) (3 3))))
  (define (main)
    (run-all-tests)))
