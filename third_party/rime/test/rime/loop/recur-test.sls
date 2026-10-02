#!r6rs
(library (test rime loop recur-test)
  (export main)
  (import (rnrs (6))
          (rime unit-test)
          (rime loop)
          )
  (define (main)
    (run-all-tests))
  (define-test
    test-basic-recur
    (CHECK equal? (loop :for i :upfrom 1 :to 10
                        :recur x0 := 0 :then x
                        :recur x := 1 :then (fx+ x0 x)
                        :collect x)
           '(1 1 2 3 5 8 13 21 34 55))
    ))
