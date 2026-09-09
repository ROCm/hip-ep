#!r6rs
(library (test rime loop collect-inside-finally-test)
  (export main)
  (import (rnrs (6))
          (rime unit-test)
          (rime loop)
          )
  (define (main)
    (run-all-tests))
  (define-test
    test-collect-at-finally
    (let ()
      (CHECK equal? (loop :for i :in '(1 2 3)
                          :with x := (fx+ i 1)
                          :collect i
                          :finally
                          :collect x
                          )
             '(1 2 3 4))
      )))
