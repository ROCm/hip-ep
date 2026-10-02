(library (rime protobuf private x-define-proto-deserialize)
  (export r-field.define-derialize)
  (import (rnrs (6))
          (rime loop)
          (rime protobuf private record)
          (rime logging)
          (rime protobuf private io)
          (rime protobuf private vec)
          (rime protobuf private message))

  (define (r-field.define-derialize r-proto-message)
    (with-syntax ([deserialize (r-message-s-deserialize r-proto-message)])
      (let ([vars (generate-temporaries (list #'deserialize #'deserialize))])
        (with-syntax ([message (car vars)]
                      [port (cadr vars)])
         (with-syntax ([(deserialize-clauses ...)
                        (loop :for field :in (r-message-fields r-proto-message)
                              :collect (define-deserialize-field #'message #'port field))])
           (list
            #'(define deserialize
                (lambda (message port)
                  (let loop ()
                    (unless (port-eof? port)
                      (let* ([pos (and (port-has-port-position? port) (port-position port))]
                             [tag (read-varint port)]
                             [decode-number (fxarithmetic-shift-right tag 3)]
                             [wired-type (fxand tag #x07)]
                             )
                        (and #t (logger :trace
                                 " deserialize " 'name
                                 " tag = " tag
                                 " decode-number = " decode-number
                                 " wired-type= = " wired-type
                                 " pos = (" pos " " (and (port-has-port-position? port) (port-position port)) ")"
                                 ))
                        (case decode-number
                          deserialize-clauses ...
                          [else (raise
                                 (condition
                                  (make-warning)
                                  (make-message-condition "unknown field:")
                                  (make-irritants-condition (list decode-number tag (port-position port)))))])
                        message)
                      (loop)))))))))))

  (define (define-deserialize-field s-message s-port r-field)
    (with-syntax ([message s-message]
                  [port s-port]
                  [message.field (r-field-s-getter r-field)]
                  [message.has.field? (r-field-s-has r-field)]
                  [message.set.field (r-field-s-setter r-field)]
                  [number (r-field-number r-field)])
      (let ([label (syntax->datum (r-field-label r-field))]
            [type (syntax->datum (r-field-type r-field))]
            [option (syntax->datum (r-field-option r-field))])
        (define (assert-not-implemented)
          #'(raise
             (condition
              (make-warning)
              (make-message-condition "not implemented")
              (make-irritants-condition (list label type option)))))

        (define (set-value value)
          (case label
            [(required optional)
             #`(message.set.field message #,value)]
            [(repeated)
             #`(vec-append (message.field message) #,value)]
            [else (assert-not-implemented)]))
        (define (build-body)
          (case type
            [(int32)
             (cond
              [(equal? (assq 'packed (cdr option)) #t)
               (set-value #'(read-packed-int32 port))]
              [else (set-value #'(read-int32 port))])
             ]
            [(int32) (set-value #'(read-int32 port))]
            [(sint32) (set-value #'(read-sint32 port))]
            [(string) (set-value #'(read-string port))]
            [else
             (cond
              [(pair? type)
               (case (car type)
                 [(message)
                  (with-syntax ([message-type (datum->syntax s-message (cadr type))])
                    (case label
                      [(optional)
                       #'(let* ([len (read-varint port)]
                                [port (make-sub-input-port port len)])
                           (unless (message.has.field? message)
                             (message.set.field message ((make-message 'message-type) ':self)))
                           ((invoke-message (message.field message)) ':<< port))]
                      [(requied)
                       #'(let* ([len (read-varint port)]
                                [port (make-sub-input-port port len)])
                           ((invoke-message (message.field message)) ':<< port))]
                      [(repeated)
                       #'(let* ([len (read-varint port)]
                                [port (make-sub-input-port port len)])
                           ((invoke-message (message.field message)) ':<< port))]))]
                 [(enum)
                  (let ([enum-type (assq 'enum (cdr type))])
                    (set-value #'(read-int32 port)))]
                 [else (assert-not-implemented)])]
              [else (assert-not-implemented)])]))
        (with-syntax ([body (build-body)])
          #'[(number) body])))))
