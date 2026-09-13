#!r6rs
(library (rime loop with)
  (export loop/core/with)
  (import (rnrs (6))
          (rime loop plugin)
          (rime loop keywords))
  (define (make-with-plugin props)
    (let* ([s-var (assq-id ':into props #f)]
           [s-expr (assq-id ':= props #'(if #f 0))]
           [s-initially (assq-id ':initially props #'(if #f 0))]
           )
      (lambda (method . args)
        (with-syntax ([var s-var]
                      [expr s-expr]
                      [initially s-initially]
                      )
          (case method
            [(debug)
             (object-to-string
              ":with " (syntax->datum #'var)
              " :initially := " (syntax->datum #'initially)
              " := " (syntax->datum #'expr)
              )]
            [(setup)
             (list #'(var initially))]

            [(iteration-body)
             (with-syntax ([(rest-body ...) (car args)])
               (list #'(begin
                         [set! var expr]
                         rest-body ...)))]
            [else (apply default-plugin #'make-with-plugin method args)])))))

  (define (loop/core/with original-e)
    (let loop ([e original-e])
      (syntax-case e (:with := :into :initially)

        [(k :with var rest ...)
         (identifier? #'var)
         (loop #'(k (:with (:into . var)) rest ...))
         ]

        [(k :with rest ...)
         (with-syntax ([return-value (loop-return-value #'k)])
           (loop #'(k (:with (:into . return-value)) rest ...)))
         ]

        [(k (:with (prop . value) ...) := expr rest ...)
         (loop #'(k (:with (:= . expr) (prop . value) ...)  rest ...))
         ]

        [(k (:with (prop . value) ...) :initially expr rest ...)
         (loop #'(k (:with (:initially . expr) (prop . value) ...)  rest ...))
         ]

        [(k (:with (prop . value) ...) rest ...)
         (values (make-with-plugin #'((prop . value) ...)) #'(k rest ...))
         ]

        [(k rest ...)
         (values #f e)
         ]))))
