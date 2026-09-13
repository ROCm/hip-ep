#!r6rs
(library (test google protobuf test school)
  (export google-protobuf-test-school)
  (import (rnrs (6))
          (rime protobuf define-proto)
          (rime protobuf private message))
  (define-proto
    (library (google protobuf test school))
    (package (google protobuf test))
    (message
     (google protobuf test School)
     ((required sint32 id 1 (option))
      (optional string name 2 (option))))))
