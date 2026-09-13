(library (test rime string __string-find-all-test)
  (export main)
  (import (rnrs (6))
          (rime string __string-find-all)
          (rime unit-test))
  (define-test test-string-find-all
    (CHECK equal?
           (string-find-all (lambda (s index) (char-upper-case? (string-ref s index))) "AbCdE")
           '(0 2 4))
    )
  (define (main)
    (run-all-tests)))
