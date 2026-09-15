#!r6rs
(library (test rime loop append-test)
  (export main)
  (import (rnrs (6))
          (rime unit-test)
          (rime loop)
          )
  (define (main)
    (run-all-tests))
  (define-test
    test-collect-append-flat-map
    (CHECK equal? (loop :for i :in '((a b c) (d e f))
                        :append i)
           '(a b c d e f))))
