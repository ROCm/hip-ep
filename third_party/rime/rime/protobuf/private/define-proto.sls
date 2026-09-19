#!r6rs
(library (rime protobuf private define-proto)
  (export define-proto)
  (import (rnrs (6)))
  (import (for (rime protobuf private x-define-proto) expand))
  (define-syntax define-proto x-define-proto))
