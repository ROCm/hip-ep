#!r6rs
(library (test rime string __string-join-test)
  (export main)
  (import (rnrs (6))
          (rime unit-test)
          (rime string __string-join)
          )

  (define-test test-string-join
    (CHECK equal?
           (string-join "::" (list "std" "vector" "type"))
           "std::vector::type"))
  (define (main)
    (run-all-tests)))
