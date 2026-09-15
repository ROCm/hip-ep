#!r6rs
(library (rime loop recur)
  (export loop/core/recur)
  (import (rnrs (6))
          (rime loop plugin)
          (rime loop keywords))
  (define (make-recur-plugin props)
    (let* ([s-var (assq-id ':into props)]
           [s-expr (assq-id ':expr props)]
           [s-then (assq-id ':then props s-var)]
           )
      (lambda (method . args)
        (with-syntax ([var s-var]
                      [expr s-expr]
                      [then s-then]
                      )
          (case method
            [(debug)
             (object-to-string
              ":recur " (syntax->datum #'var)
              " := " (syntax->datum #'expr)
              " :then " (syntax->datum #'then)
              )]
            [(recur)
             (list #'(var expr))]
            [(iteration)
             (list #'(var var then))]

            [else (apply default-plugin #'make-recur-plugin method args)])))))

  (define (loop/core/recur original-e)
    (let loop ([e original-e])
      (syntax-case e (:recur := :then :into :expr)
        [(k :recur var := expr  rest ...)
         (identifier? #'var)
         (loop #'(k (:recur (:into . var) (:expr . expr)) rest ...))
         ]

        [(k (:recur (prop . value) ...) :then expr rest ...)
         (loop #'(k (:recur (:then . expr) (prop . value) ...)  rest ...))
         ]

        [(k (:recur (prop . value) ...) rest ...)
         (values (make-recur-plugin #'((prop . value) ...)) #'(k rest ...))
         ]

        [(k rest ...)
         (values #f e)
         ]))))
