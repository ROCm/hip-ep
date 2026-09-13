#!r6rs
(library (test rime loop break-if-test)
  (export main)
  (import (rnrs (6))
          (rime unit-test)
          (rime loop)
          )
  (define (main)
    (run-all-tests))
  (define-test
    test-break-2
    (CHECK equal? (loop :for i :upfrom 0 :to 10
                        :break i :if (= i 2))
           2)))
