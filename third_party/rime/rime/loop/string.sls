#!r6rs
(library (rime loop string)
  (export loop/core/string)
  (import (rnrs (6))
          (rime loop plugin)
          (rime loop keywords))
  (define (make-string-plugin props)
    (let ([s-var (assq-id ':into props)]
          [s-expr (assq-id ':expr props)]
          [s-offset (assq-id ':offset props #'0)]
          [s-reverse (assq-id ':reverse props #f)])
      (lambda (method . args)
        (with-syntax ([expr s-expr]
                      [var s-var]
                      [offset s-offset]
                      [reverse s-reverse]
                      [expr-var (new-sym s-var "-string-expr")]
                      [len (new-sym s-var "string-length-string-expr")]
                      [var-index (new-sym s-var "-string-index")])
          (case method
            [(debug)
             (object-to-string
              ":for " (syntax->datum #'var)
              " :in-string "
              " :reverse " (syntax->datum #'reverse)
              " :offset " (syntax->datum #'offset)
              " :in-string " (cons '~s (syntax->datum #'expr)))]
            [(recur)
             (list #'(expr-var expr))]
            [(outer-iteration)
             (list #'[len (string-length expr-var)])]
            [(iteration)
             (list #'[var-index offset (+ 1 var-index)])]
            [(inner-if-true)
             (list
              (if s-reverse
                  #'[var (string-ref expr-var (fx- (fx- len var-index) 1))]
                  #'[var (string-ref expr-var var-index)]))]
            [(continue-condition)
             #'(< var-index len)]
            [else (apply default-plugin #'make-string-plugin method args)])))))

  (define (loop/core/string original-e)
    (let loop [(e original-e)]
      (syntax-case e (:for :into :expr :reverse :in-string :offset)
        [(k :for var :in-string expr rest ...)
         (loop #'(k (:in-string (:into . var) (:expr . expr)) rest ...))
         ]

        [(k (:in-string (prop . value) ...) :reverse rest ...)
         (loop #'(k (:in-string (:reverse . #t) (prop . value) ...) rest ...))
         ]

        [(k (:in-string (prop . value) ...) :offset offset rest ...)
         (loop #'(k (:in-string (:offset . offset) (prop . value) ...) rest ...))
         ]
        [(k (:in-string (prop . value) ...) rest ...)
         (begin
           (values (make-string-plugin #'((prop . value) ...))
                   #'(k rest ...)))]
        [(k rest ...)
         (values #f e)
         ]))))
