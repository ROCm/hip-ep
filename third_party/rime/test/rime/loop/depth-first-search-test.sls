#!r6rs
(library (test rime loop depth-first-search-test)
  (export main)
  (import (rnrs (6))
          (rime unit-test)
          (rime loop)
          )
  (define (main)
    (run-all-tests))
  (define-test
    test-depth-first-travel
    (let ()
      (CHECK equal? (loop :name recur
                          :for i :in '(0 1 2 (1 11 (20 21 22) 12) 3 4 5)
                          :do (recur i) :if (list? i)
                          :collect i :unless (list? i)
                          )
             '(0 1 2 1 11 20 21 22 12 3 4 5))
      )))
