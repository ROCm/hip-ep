#!r6rs
(library (rime loop list)
  (export loop/core/list)
  (import (rnrs (6))
          (rime loop plugin)
          (rime loop keywords))

  (define (make-for-as-list s-var s-expr in/on)
    (let ([s-expr-recur-var (new-sym s-var "list:-in-recur-list")]
          [s-expr-loop-var (new-sym s-var "list:-in-loop-list")])
      (lambda (method . args)
        (with-syntax ([var s-var]
                      [expr-recur-var s-expr-recur-var]
                      [expr-loop-var s-expr-loop-var]
                      [expr s-expr])
          (case method
            [(debug)
             (object-to-string
              ":for " (syntax->datum #'var)
              " :in " (syntax->datum #'expr))]
            [(recur)
             (list #'[expr-recur-var expr])]
            [(iteration)
             (list #'[expr-loop-var expr-recur-var (cdr expr-loop-var)])]
            [(inner-if-true)
             (list (if (eq? in/on 'in)
                       #'[var (car expr-loop-var)]
                       #'[var expr-loop-var]))]
            [(continue-condition)
             #'(not (null? expr-loop-var))]
            [else (apply default-plugin #'make-for-as-list method args)])))))

  (define (loop/core/list e)
    (syntax-case e (:for :in :on)
      [(k :for i :in expr1 rest ...)
       (begin
         (values (make-for-as-list #'i #'expr1 'in)
                 #'(k rest ...)))]
      [(k :for i :on expr1 rest ...)
       (begin
         (values (make-for-as-list #'i #'expr1 'on)
                 #'(k rest ...)))]
      [(k rest ...)
       (values #f e)
       ])))
