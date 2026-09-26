#!r6rs
(library (rime loop hashtable)
  (export loop/core/hashtable)
  (import (rnrs (6))
          (rime loop plugin)
          (rime loop keywords))
  (define (make-hashtable-plugin s-key s-value s-expr)
    (let ([s-expr-var (cond
                       [s-key  (new-sym s-key "-expr-var")]
                       [s-value  (new-sym s-value "-expr-value")]
                       [else (syntax-violation
                              'make-hashtable-plugin
                              "either key or value must be specified"
                              #'(s-key s-value s-expr))])]
          [s-keys
           (if s-key (new-sym s-key "-keys-var") #f)]
          [s-values
           (if s-value (new-sym s-value "-values-var") #f)]
          [s-var-index
           (new-sym (or s-key s-value) "-index")]
          [s-hash-size
           (new-sym (or s-key s-value) "-hash-size")])
      (lambda (method . args)
        (with-syntax ([key s-key]
                      [value s-value]
                      [expr s-expr]
                      [expr-var s-expr-var]
                      [keys s-keys]
                      [values s-values]
                      [var-index s-var-index]
                      [hash-size-expr (or (and s-keys #`(vector-length #,s-keys))
                                          (and s-values #`(vector-length #,s-values)))]
                      [hash-size s-hash-size])
          (case method
            [(debug)
             (object-to-string
              ":for "
              "(" (syntax->datum #'key) "  " (syntax->datum #'value) ")"
              " :in-hashtable " (syntax->datum #'expr))]
            [(recur)
             (list #'(expr-var expr))]
            [(outer-iteration)
             (remove #f
                     (list
                      #'(keys (hashtable-keys expr-var))
                      #'(hash-size hash-size-expr)))]
            [(iteration)
             (list #'[var-index 0 (fx+ 1 var-index)])]
            [(inner-if-true)
             (remove #f
                     (list (and s-key #'[key (vector-ref keys var-index)])
                           (and s-value #'[value (hashtable-ref expr-var key #f)])))]
            [(continue-condition)
             #'(fx<? var-index hash-size)]
            [(iteration)
             (with-syntax ([(rest-body ...) (car args)])
               (list #'(begin expr rest-body ...)))]
            [else (apply default-plugin #'make-hashtable-plugin method args)])))))
  (define (loop/core/hashtable e)
    (syntax-case e (:in-hashtable)
      [(k :for (key value) :in-hashtable expr rest ...)
       (and (identifier? #'key) (identifier? #'value))
       (begin
         (values (make-hashtable-plugin #'key #'value #'expr)
                 #'(k rest ...)))]
      [(k :for key :in-hashtable expr rest ...)
       (and (identifier? #'key))
       (begin
         (values (make-hashtable-plugin #'key #f #'expr)
                 #'(k rest ...)))]
      [(k rest ...)
       (values #f e)
       ])))
