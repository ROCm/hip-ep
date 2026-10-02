#!r6rs
(library (test rime files __directory-list-test)
  (export main)
  (import (rnrs (6))
          (rime files __directory-list)
          (rime unit-test))

  (define (main)
    (run-all-tests))
  ;;(define-test hello-test
  ;;  (CHECK equal? 
  ;;         (length (list-sort string<=? (directory-list "test")))
  ;;         5))
  )
