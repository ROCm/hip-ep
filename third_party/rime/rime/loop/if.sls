#!r6rs
(library (rime loop if)
  (export loop/core/if)
  (import (rnrs (6))
          (rime loop plugin)
          (rime loop keywords))
  (define (make-if-plugin s-expr)
    (let ()
      (lambda (method . args)
        (with-syntax ([expr s-expr])
          (case method
            [(debug)
             (object-to-string
              ":if " (syntax->datum #'expr))]
            [(iteration-body)
             (with-syntax ([(rest-body ...) (car args)])
               (list #'(if expr
                           (begin
                             ;; support empty rest-body
                             0 rest-body ...))))]
            [else (apply default-plugin #'make-if-plugin method args)])))))
  (define (loop/core/if e)
    (syntax-case e (:if :when :unless)
      [(k :if expr rest ...)
       (begin
         (values (make-if-plugin #'expr) #'(k rest ...)))]
      [(k :when expr rest ...)
       (begin
         (values (make-if-plugin #'expr) #'(k rest ...)))]
      [(k :unless expr rest ...)
       (begin
         (values (make-if-plugin #'(not expr)) #'(k rest ...)))]
      [(k rest ...)
       (values #f e)
       ])))
