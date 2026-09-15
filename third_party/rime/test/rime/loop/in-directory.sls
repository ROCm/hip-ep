#!r6rs
(library (test rime loop directory-test)
  (export main)
  (import (rnrs (6))
          (rime unit-test)
          (rime loop)
          )
  (define (main)
    (run-all-tests))
  (define-test
    test-basic-directory
    (CHECK equal? (loop :for file :in-directory "test"
                        :collect file)
           '())
    ))
