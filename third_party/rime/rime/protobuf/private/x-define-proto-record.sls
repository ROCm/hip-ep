#!r6rs
(library (rime protobuf private x-define-proto-record)
  (export r-proto.define-record-type
          ;; chez complain reference unexported identifier
          not-present)
  (import (rnrs (6))
          (rnrs bytevectors (6))
          (rime loop)
          (rime protobuf private record)
          (rime protobuf private message)
          (rime protobuf private vec))
  (define not-present 'NOT-PRESENT)

  (define (r-proto.define-record-type r-proto)
    (loop :for r-proto-message :in (r-proto-messages r-proto)
          (:loop :for code :in (r-message.define-record-type r-proto-message)
                 :collect code)
          ))

  (define (r-message.define-record-type r-proto-message)
    (let ([r-fields  (r-message-fields r-proto-message)])
      (with-syntax ([record (r-message-s-record r-proto-message)]
                    [ctor (r-message-s-ctor r-proto-message)]
                    [pred (r-message-s-predictor r-proto-message)]
                    [handle (car (generate-temporaries (list (r-message-s-handle r-proto-message))))]
                    [(field-name ...) (map r-field-name r-fields)]
                    [(field-getter ...) (map r-field-s-getter r-fields)]
                    [(field-setter ...) (map r-field-s-setter r-fields)]
                    [(field-label ...) (map r-field-label r-fields)]
                    [(field-type ...) (map r-field-type r-fields)]
                    [name  (r-message-name-as-syntax r-proto-message)]
                    [deserilize  (r-message-s-deserialize r-proto-message)]
                    )
        (with-syntax ([(field-default-value ...)
                       (map field-default-value #'(field-label ...) #'(field-type ...))])
          (apply
           list
           #'(define-record-type (record ctor pred)
               (fields (mutable field-name field-getter field-setter) ...)
               (parent protobuf-message)
               (protocol
                (lambda (new)
                  (letrec* ([tmp-ctor (lambda ()
                                        ((new meta) field-default-value ...))]
                            [meta
                              (make-protobuf-meta-message
                               tmp-ctor
                               deserilize)])
                    (register-message-ctor 'name meta)
                    tmp-ctor))))
           (loop :for field :in r-fields
                 :if (eq? (syntax->datum (r-field-label field)) 'optional)
                 :collect (with-syntax ([field-has? (r-field-s-has field)]
                                        [field-get (r-field-s-getter field)])
                            #'(define (field-has? message)
                                (not (eq? (field-get message) not-present))))))))))

  (define (field-default-value label type)
    (with-syntax ([label label]
                  [type type])
      (syntax-case #'(label type)
          (optional repeated required double float
                    int64 uint64 fixed64 fixed32
                    bool string
                    message enum
                    bytes
                    uint32
                    sfixed64
                    sfixed32
                    sint32
                    sint64
                    )
        [(required double) 0.0]
        [(required float) 0.0]
        [(required int64) 0]
        [(required uint64) 0]
        [(required fixed64) 0]
        [(required fixed32) 0]
        [(required bool) #f]
        [(required string) ""]
        [(required (message <MESSAGE-FULL-NAME> ...))
         #'(make-message '(<MESSAGE-FULL-NAME> ...))]
        [(required bytes)
         (make-bytevector 0)]
        [(required uint32) 0]
        [(required (enum m ...)) 0]
        [(required sfixed32) 0]
        [(required sfixed64) 0]
        [(required sint32) 0]
        [(required sint64) 0]
        [(optional _) #'not-present]
        [(repeated _) #'(vec)]
        [(otherwise ...) (raise (call-with-string-output-port
                                 (lambda (port)
                                   (for-each
                                    (lambda (x)
                                      (display x port))
                                    (list " (field-default-value)"
                                          " TODO "
                                          (syntax->datum #'(label type)))) "\n"
                                   )))]))))
