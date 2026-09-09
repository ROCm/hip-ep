#!r6rs
(library (test rime loop finally-test)
  (export main)
  (import (rnrs (6))
          (rime unit-test)
          (rime loop)
          )
  (define (main)
    (run-all-tests))
  (define-test
    test-basic-finally
    (CHECK equal? (loop :for i :upfrom 0 :to 10
                        :if (odd? i)
                        :with i+ := (+ 1 i)
                        :with i++ := (+ 1 i+)
                        :collect i++ :into acc123
                        :finally (map (lambda (x) (+ 100 x)) acc123))
           '(103 105 107 109 111))

    (CHECK equal? (loop :for i :upfrom 0 :to 10
                        :if (odd? i)
                        :with i+ := (+ 1 i)
                        :with i++ := (+ 1 i+)
                        :collect i++ :if (fx>=? i++ 7) :into acc123
                        :finally (map (lambda (x) (+ 100 x)) acc123))
           '(107 109 111))
    (CHECK equal? (loop :for i :upfrom 0 :to 10
                        :if (odd? i)
                        :with i+ := (+ 1 i)
                        :with i++ := (+ 1 i+)
                        :collect i++ :into acc123 :if (fx>=? i++ 7)
                        :finally (map (lambda (x) (+ 100 x)) acc123))
           '(107 109 111))
    ))
