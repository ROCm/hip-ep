#!r6rs
(library (test rime loop count-test)
  (export main)
  (import (rnrs (6))
          (rime unit-test)
          (rime loop)
          )
  (define (main)
    (run-all-tests))
  (define-test
    test-count
    (CHECK equal? (loop :for i :upfrom 0 :to 10
                        :count)
           11)
    (CHECK equal? (loop :for i :upfrom 0 :to 10
                        :count
                        :finally (+ :return-value 100))
           111)
    ))
