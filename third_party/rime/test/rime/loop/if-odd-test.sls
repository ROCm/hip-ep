#!r6rs
(library (test rime loop if-odd-test)
  (export main)
  (import (rnrs (6))
          (rime unit-test)
          (rime loop)
          )
  (define (main)
    (run-all-tests))

  (define-test
    test-if-odd?
    (CHECK equal? (loop :for i :upfrom 1 :to 10
                        :if (odd? i)
                        :collect i)
           '(1 3 5 7 9))
    (CHECK equal? (loop :for i :upfrom 1 :to 10
                        :when (odd? i)
                        :collect i)
           '(1 3 5 7 9))
    (CHECK equal? (loop :for i :upfrom 1 :to 10
                        :unless (odd? i)
                        :collect i)
           '(2 4 6 8 10))
    ))
