#!r6rs
(library (test rime loop repeat-3-test)
  (export main)
  (import (rnrs (6))
          (rime unit-test)
          (rime loop)
          )
  (define (main)
    (run-all-tests))
  (define-test
    repeat-3
    (CHECK equal? (loop :repeat 3
                        :collect 103)
           '(103 103 103))))
