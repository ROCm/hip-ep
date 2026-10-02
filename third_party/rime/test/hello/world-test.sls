#!r6rs
(library (test hello world-test)
  (export main)
  (import (rnrs (6))
          (rime unit-test))

  (define-test hello-world-test
    (CHECK equal? (+ 1 1) 2)
    1)
  (define (main)
    (run-all-tests))
  )
