#!r6rs
(library (test rime loop for-i-in-vector-123)
  (export main)
  (import (rnrs (6))
          (rime unit-test)
          (rime loop)
          )
  (define-test
    test:for-i-in-vector-123
    (CHECK equal? (loop :for i :in-vector (vector 1 2 3)
                        :collect i)
           '(1 2 3))
    (CHECK equal? (loop :for i :across (vector 1 2 3)
                        :collect i)
           '(1 2 3)))
  (define (main)
    (run-all-tests)))
