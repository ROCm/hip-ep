(library (rime loop count)
  (export loop/core/count)
  (import (rnrs (6))
          (rime loop plugin)
          (rime loop keywords))
  (define (make-count-by-plugin props)
    (let ([s-key (assq-id ':by props #f)]
          [s-var (assq-id ':into props #f)]
          [s-cond-expr (assq-id ':if props #t)]
          [s-make-hash-table (assq-id ':make-hash-table props #'(make-eq-hashtable))])

      (with-syntax ([key s-key]
                    [var s-var]
                    [cond-expr s-cond-expr]
                    [make-hash-table s-make-hash-table]
                    )
        (lambda (method . args)
          (case method
            [(debug)
             (object-to-string
              ":count "
              (if s-key (list " :by "  (syntax->datum s-key))
                  "")
              " :into " (syntax->datum s-var)
              " :if " (syntax->datum s-cond-expr))
             ]

            [(setup)
             (remove
              #f
              (list
               (cond
                [s-key #'(var make-hash-table)]
                [else #'(var 0)])

               ))]

            [(iteration-body)
             (cons
              (cond
               [s-key
                #'(when cond-expr
                    (hashtable-update! var
                                       key
                                       (lambda (pre)
                                         (fx+ pre 1))
                                       0))
                ]
               [else
                #'(when cond-expr
                    (set! var (fx+ var 1)))])
              (car args))
             ]
            [else (apply default-plugin #'make-collect-plugin method args)])))))

  (define (loop/core/count original-e)
    (let loop ([e original-e])
      (syntax-case e (:count :group :by :if :when :unless :into :make-hash-table)
        [(k :count rest ...)
         (with-syntax ([return-value (loop-return-value #'k)])
           (loop #'(k (:count (:into . return-value)) rest ...)))]
        [(k (:count (prop . value) ...) :by key rest ...)
         (loop #'(k (:count (:by . key) (prop . value) ...) rest ...))
         ]

        [(k (:count (prop . value) ...) :if cond-expr rest ...)
         (loop #'(k (:count (:if . cond-expr) (prop . value) ...) rest ...))
         ]

        [(k (:count (prop . value) ...) :when cond-expr rest ...)
         (loop #'(k (:count (:if . cond-expr) (prop . value) ...) rest ...))
         ]

        [(k (:count (prop . value) ...) :unless cond-expr rest ...)
         (loop #'(k (:count (:if . (not cond-expr)) (prop . value) ...) rest ...))
         ]

        [(k (:group (prop . value) ...) :into var rest ...)
         (loop #'(k (:group (:into . var) (prop . value) ...) rest ...))]

        [(k (:count (prop . value) ...) :make-hash-table make-hash-table rest ...)
         (loop #'(k (:count (:make-hash-table . make-hash-table) (prop . value) ...) rest ...))
         ]

        [(k (:count (prop . value) ...) rest ...)
         (values (make-count-by-plugin #'((prop . value) ...))
                 #'(k rest ...))
         ]
        [(k rest ...)
         (values #f e)
         ]))))
