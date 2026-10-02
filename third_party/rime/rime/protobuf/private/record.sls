#!r6rs
(library (rime protobuf private record)
  (export r-proto?
          make-r-proto
          (rename (r-proto-lib-name* r-proto-lib-name))
          r-proto-messages
          r-proto-s-define-proto
          r-proto-lib-init-name
          r-proto-s-proto-handle
          (rename (r-message-name*  r-message-name)
                  (r-message-name r-message-name-as-syntax))
          r-message-fields
          r-message-s-handle
          r-message-s-record
          r-message-s-ctor
          r-message-s-predictor
          r-message-s-deserialize
          r-field-seq
          r-field-label
          r-field-type
          r-field-name
          r-field-number
          r-field-option
          r-field-s-getter
          r-field-s-setter
          r-field-s-has
          )
  (import (rnrs (6))
          (rime loop)
          (rime logging))

  (define (r-proto-lib-name* r-proto)
    (syntax->datum (r-proto-lib-name r-proto)))

  (define (r-proto-lib-init-name r-proto)
    (datum->syntax
     (r-proto-s-define-proto r-proto)
     (string->symbol
      (apply string-append
             (loop :for name :in (r-proto-lib-name* r-proto)
                   :collect sep
                   :collect (symbol->string name)
                   :with sep := "-" :initially "")))))

  (define-record-type r-proto
    (fields lib-name              ; and id for debugger purpose
            package               ; list of symbols, for package name
            messages              ; ::vector<r-message>
            s-define-proto        ; use for create id in the top scope
            s-proto-handle        ; use for create id in the top scope
            )
    (protocol
     (lambda (new)
       (lambda (proto)
         (syntax-case proto (package library)
           [(define-proto
              (library (lib-name ...))
              (package (package-name ...))
              messages ...)
            (new
             #'(lib-name ...)
             #'(package-name ...)
             (loop :for m :in #'(messages ...)
                   :collect (make-r-message #'define-proto m))
             #'define-proto
             (construct-name #'define-proto "proto-handle")
             )
            ])))))

  (define (r-message-name* message)
    (syntax->datum (r-message-name message)))
  (define-record-type r-message
    (fields name         ; list<symbol> for full name of the message
            fields       ; ::vector<r-field>
            s-handle     ; syntax for the compile time value
            s-record     ; syntax identifier for generated record type
            s-ctor ; syntax identifier for generated record constructor
            s-predictor ; syntax identifier for generated record constructor
            s-deserialize               ; member function deserialize
            )
    (protocol
     (lambda (new)
       (lambda (define-proto m)
         (syntax-case m (message)
           [(message
             (package ... message-name)
             (fields ...))
            (let ([name #'(package ... message-name)])
              (with-syntax ([full-name (apply construct-name define-proto
                                              (loop :for n :in #'(package ... message-name)
                                                    :collect seq
                                                    :collect n
                                                    :with seq := "-" :initially ""))])
                (with-syntax ([s-handle
                               (construct-name #'full-name 'handle- #'full-name)]
                              [s-record
                               (construct-name #'full-name 'record- #'full-name)]
                              [s-ctor
                               (construct-name #'full-name 'make- #'full-name)]
                              [s-predictor
                               (construct-name #'full-name #'full-name '?)]
                              [s-deserialize
                               (construct-name #'full-name #'full-name "-deserialize")]
                              )
                  (new name
                       (loop :for seq :upfrom 0
                             :for a-field :in #'(fields ...)
                             :collect (make-r-field #'full-name seq a-field))
                       #'s-handle
                       #'s-record
                       #'s-ctor
                       #'s-predictor
                       #'s-deserialize)))
              )])))))

  (define-record-type r-field
    (fields seq                 ; int
            label               ; symbol: optional, repeated, required
            type                ; symbols protobuf types
            name                ; symbol for field name
            number              ; int for field number
            option              ; ...?
            s-getter            ; syntax identifier for field-accsor
            s-has               ; syntax identifier for field-accsor
            s-setter            ; syntax identifier for field-mutator
            )
    (protocol
     (lambda (new)
       (lambda (full-name seq field)
         (syntax-case field ()
           [(label type name number option)
            (let ([s-getter (construct-name full-name full-name '- #'name)]
                  [s-has (if
                          (eq? (syntax->datum #'label) 'optional)
                          (construct-name full-name full-name '- "has-" #'name "?") #f)]
                  [s-setter (construct-name full-name full-name '- #'name "-set!")])
              (new seq
                   #'label
                   #'type
                   #'name
                   #'number
                   #'option
                   s-getter
                   s-has
                   s-setter))]))))
    )

  (define (construct-name template-identifier . args)
    (datum->syntax
     template-identifier
     (string->symbol
      (apply string-append
             (map (lambda (x)
                    (if (string? x)
                        x
                        (symbol->string (syntax->datum x))))
                  args)))))
  )
