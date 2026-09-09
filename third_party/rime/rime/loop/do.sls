#!r6rs
(library (rime loop do)
  (export loop/core/do)
  (import (rnrs (6))
          (rime loop plugin)
          (rime loop keywords))
  (define (make-do-plugin s-expr s-cond)
    (let ()
      (lambda (method . args)
        (with-syntax ([expr s-expr])
          (case method
            [(debug)
             (object-to-string
              ":do " (syntax->datum #'expr))]
            [(iteration-body)
             (with-syntax ([cond-expr s-cond]
                           [(rest-body ...) (car args)])
               (list #'(begin
                         (when cond-expr
                           expr)
                         rest-body ...)))]
            [else (apply default-plugin #'make-do-plugin method args)])))))
  (define (loop/core/do original-e)
    (let loop ([e original-e])
      (syntax-case e (:do :if :when :unless)
        [(k :do expr :if cond-expr rest ...)
         (begin
           (values (make-do-plugin #'expr #'cond-expr)
                   #'(k rest ...)))]
        [(k :do expr :when cond-expr rest ...)
         (loop #'(k :do expr :if cond-expr rest ...))]
        [(k :do expr :unless cond-expr rest ...)
         (loop #'(k :do expr :if (not cond-expr) rest ...))]
        [(k :do expr rest ...)
         (loop #'(k :do expr :if #t rest ...))]
        [(k rest ...)
         (values #f e)
         ]))))
