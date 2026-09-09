#!r6rs
(library (test rime protobuf private x-define-proto-record-test)
  (export main)
  (import (rnrs (6))
          (rime protobuf private x-define-proto-record)
          (rime protobuf private record)
          (rime unit-test)
          )
  (define-test test-x-define-proto-record
    (map syntax->datum
         (r-proto.define-record-type
          (make-r-proto
           #'(define-proto
               (library (google protobuf test school))
               (package (google protobuf test))
               (message
                (google protobuf test School)
                ((optional sint32 id 1 (option))
                 (optional string name 2 (option)))))))))
  (define (main)
    (run-all-tests)
    ))
