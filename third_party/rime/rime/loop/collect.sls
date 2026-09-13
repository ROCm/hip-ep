#!r6rs
(library (rime loop collect)
  (export loop/core/collect)
  (import (rnrs (6))
          (rime loop plugin)
          (rime loop keywords)
          (for (rime loop list-collector)))

  (define (make-collect-list-plugin s-var s-expr append? s-cond-expr)
    (with-syntax ([var s-var]
                  [var-collector-and-extractor (new-sym s-var "collect-list:collector-and-extractor")]
                  [var-collector (new-sym s-var "collect-list:collector")]
                  [var-extractor (new-sym s-var "collect-list:extractor")]
                  [expr s-expr])
      (lambda (method . args)
        (case method
          [(debug)
           (object-to-string
            (if append? ":append" ":collect") " "
            (syntax->datum s-expr) " :into " (syntax->datum s-var))
           ]
          [(setup)
           (list
            #'(var '() #t) ;; weak = #t
            #'(var-collector-and-extractor (list-collector var))
            #'(var-collector (list-ref var-collector-and-extractor 0))
            #'(var-extractor (list-ref var-collector-and-extractor 1)))]

          [(iteration-body)
           (cons
            (with-syntax ([cond-expr s-cond-expr])
              (if (not append?)
                  #'(when cond-expr
                      (var-collector expr)
                      (set! var (var-extractor))
                      var)
                  #'(when cond-expr
                      (for-each var-collector expr)
                      (set! var (var-extractor))
                      var)))
            (car args))
           ]
          [else (apply default-plugin #'make-collect-plugin method args)]))))

  (define (make-collect-hash-table-plugin s-var s-expr s-cond-expr s-ctor)
    (lambda (method . args)
      (syntax-case s-expr ()
        [(key value)
         (with-syntax ([var s-var]
                       [expr s-expr]
                       [cond-expr s-cond-expr]
                       [ctor s-ctor])
           (case method
             [(debug)
              (object-to-string
               ":collect :as :hash-table "
               (syntax->datum s-expr) " :into " (syntax->datum s-var))
              " :ctor "  (syntax->datum s-ctor)
              " :if "  (syntax->datum s-cond-expr)
              ]
             [(setup)
              (list
               #'(var ctor))]

             [(iteration-body)
              (cons
               #'(when cond-expr
                   (hashtable-set! var key value))
               (car args))
              ]
             [else (apply default-plugin #'make-collect-plugin method args)]))]
        [else (syntax-violation 'make-collect-hash-table-plugin "expect key value pair" s-expr)])))

  (define (loop/core/collect original-e)
    (let loop ([e original-e])

      (syntax-case e (:collect
                      :append
                      :into
                      :if
                      :when
                      :unless
                      :expr
                      :as
                      :list
                      :hast-able
                      :make-hash-table
                      )
        [(k :collect expr rest ...)
         (loop #'(k (:collect (:append . #f) (:expr . expr)) rest ...))
         ]

        [(k :append expr rest ...)
         (loop #'(k (:collect (:append . #t) (:expr . expr)) rest ...))
         ]

        [(k (:collect (prop . value) ...) :if cond-expr rest ...)
         (loop #'(k (:collect (:if . cond-expr) (prop . value) ...) rest ...))
         ]

        [(k (:collect (prop . value) ...) :when cond-expr rest ...)
         (loop #'(k (:collect (:if . cond-expr) (prop . value) ...) rest ...))
         ]

        [(k (:collect (prop . value) ...) :unless cond-expr rest ...)
         (loop #'(k (:collect (:if . (not cond-expr)) (prop . value) ...) rest ...))
         ]

        [(k (:collect (prop . value) ...) :into var rest ...)
         (loop #'(k (:collect (:into . var) (prop . value) ...) rest ...))
         ]

        [(k (:collect (prop . value) ...) :as :list rest ...)
         (loop #'(k (:collect (:as . :list) (prop . value) ...) rest ...))
         ]

        [(k (:collect (prop . value) ...) :as :hash-table rest ...)
         (loop #'(k (:collect (:as . :hash-table)
                              (prop . value) ...) rest ...))
         ]

        [(k (:collect (prop . value) ...) :make-hash-table expr rest ...)
         (loop #'(k (:collect (:make-hash-table . expr)
                              (prop . value) ...) rest ...))
         ]

        [(k (:collect (prop . value) ...) rest ...)
         (let [(props #'((prop . value) ...))]
           (values (cond
                    [(keyword=? (assq-id ':as props #':list) #':list)
                     (make-collect-list-plugin
                      (assq-id ':into props (loop-return-value #'k))
                      (assq-id ':expr props #f)
                      (syntax->datum (assq-id ':append props #f))
                      (assq-id ':if props #t))]
                    [(keyword=? (assq-id ':as props #':list) #':hash-table)
                     (make-collect-hash-table-plugin
                      (assq-id ':into props (loop-return-value #'k))
                      (assq-id ':expr props #'(#f . #f))
                      (assq-id ':if props #t)
                      (assq-id ':make-hash-table props #'(make-eq-hashtable)))]
                    [else (raise "not go here")])
                   #'(k rest ...)))
         ]

        [(k rest ...)
         (values #f e)
         ]
        ))))
