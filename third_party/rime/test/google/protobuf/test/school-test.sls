#!r6rs
(library (test google protobuf test school-test)
  (export main)
  (import (rnrs (6))
          (rime protobuf)
          (test google protobuf test school)
          (rime logging)
          (rime unit-test))
  (define test-school (list google-protobuf-test-school))
  (define-test create-school
    ;; load the library
    (let ([school (make-message '(google protobuf test School))])
      (logger :info " school=" (school ':debug-string))))
  (define (main)
    (run-all-tests)))
