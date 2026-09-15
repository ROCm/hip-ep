#!r6rs
(library (rime protobuf private x-define-proto)
  (export x-define-proto)
  (import (rnrs (6))
          (rime pretty-print)
          (rime logging)
          (rime protobuf private record)
          (rime protobuf private x-define-proto-record)
          (rime protobuf private x-define-proto-deserialize)
          )
  (define (x-define-init r-proto)
    (with-syntax ([init-name  (r-proto-lib-init-name r-proto)])
      ;; chezscheme requires refering a identifier to initialize a library
      #'(define init-name #f)))
  (define (x-define-proto e)
    (let ([proto (make-r-proto e)])
      (let ([output
             #`(begin
                 #,(x-define-init proto)
                 #,@(apply append (map r-field.define-derialize (r-proto-messages proto)))
                 #,@(r-proto.define-record-type proto)
                 )])
        (and #t (logger :trace
          " output:\n"
          (call-with-string-output-port
           (lambda (port)
             (pretty-print (syntax->datum output) port)))
          "\n"
          ))
        output))))
