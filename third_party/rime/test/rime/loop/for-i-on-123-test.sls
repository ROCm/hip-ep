#!r6rs
(library (test rime loop for-i-on-123-test)
  (export main)
  (import (rnrs (6))
          (rime unit-test)
          (rime loop)
          )
  (define-test test:for-i-on-123
    (lambda ()
      (CHECK equal? (loop :for i :on '(1 2 3)
                          :collect i)
             '((1 2 3) (2 3) (3)))))
  (define (main)
    (run-all-tests)))
