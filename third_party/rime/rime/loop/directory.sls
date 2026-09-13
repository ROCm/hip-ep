#!r6rs
(library (rime loop directory)
  (export loop/core/directory)
  (import (rnrs (6))
          (rime loop plugin)
          (rime loop keywords)
          (rime files __directory-list))
  (define (make-directory-plugin props)
    (let* ([s-var (assq-id ':into props)]
           [s-expr (assq-id ':expr props)]
           [s-expr-recur-var (new-sym s-var "list:-in-recur-list")]
           [s-expr-loop-var (new-sym s-var "list:-in-loop-list")]
           )
      (lambda (method . args)
        (with-syntax ([var s-var]
                      [expr s-expr]
                      [expr-recur-var s-expr-recur-var]
                      [expr-loop-var s-expr-loop-var]
                      )
          (case method
            [(debug)
             (object-to-string
              ":for " (syntax->datum #'var)
              " :in-directory " (syntax->datum #'expr)
              )]
            [(recur)
             (list #'[expr-recur-var expr])]
            [(iteration)
             (list #'[expr-loop-var (directory-list expr-recur-var) (cdr expr-loop-var)])]
            [(inner-if-true)
             (list #'[var (car expr-loop-var)])]
            [(continue-condition)
             #'(not (null? expr-loop-var))]
            [else (apply default-plugin #'make-directory-plugin method args)])))))

  (define (loop/core/directory original-e)
    (let loop ([e original-e])
      (syntax-case e (:for :in-directory)
        [(k :for var :in-directory expr  rest ...)
         (identifier? #'var)
         (loop #'(k (:in-directory (:into . var) (:expr . expr)) rest ...))
         ]

        [(k (:directory (prop . value) ...) rest ...)
         (values (make-directory-plugin #'((prop . value) ...)) #'(k rest ...))
         ]

        [(k rest ...)
         (values #f e)
         ]))))
