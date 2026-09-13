#!r6rs
(library (rime protobuf define-proto)
  (export define-proto)
  (import (rnrs (6))
          (for (rime protobuf private x-define-proto) expand))
  (define-syntax define-proto x-define-proto)
  (begin "HELLO"))
