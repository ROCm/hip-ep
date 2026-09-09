#!r6rs
(library (test rime loop with-test)
  (export main)
  (import (rnrs (6))
          (rime unit-test)
          (rime loop)
          )
  (define (main)
    (run-all-tests))
  (define-test
    test-basic-with
    (CHECK equal? (loop :for i :upfrom 0 :to 2
                        :with i+ := (+ 1 i)
                        :with i++ := (+ 1 i+)
                        :collect i++)
           '(2 3 4))
    ))
