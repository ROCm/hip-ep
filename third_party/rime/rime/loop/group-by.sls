#!r6rs
(library (rime loop group-by)
  (export loop/core/group-by)
  (import (rnrs (6))
          (rime loop plugin)
          (rime loop keywords)
          (for (rime loop list-collector)))
  (define (make-group-by-plugin props)
    (let ([s-key (assq-id ':by props #f)]
          [s-value (assq-id ':group props #f)]
          [s-count (assq-id ':count props #f)]
          [s-var (assq-id ':into props #f)]
          [s-cond-expr (assq-id ':if props #t)]
          [s-make-hash-table (assq-id ':make-hash-table props #'(make-eq-hashtable))])

      (with-syntax ([key s-key]
                    [value s-value]
                    [var s-var]
                    [cond-expr s-cond-expr]
                    [make-hash-table s-make-hash-table]
                    [var-collector (new-sym s-var "group-by:collector")]
                    )
        (lambda (method . args)
          (case method
            [(debug)
             (object-to-string
              ":group "  (syntax->datum s-value)
              " :by "  (syntax->datum s-key)
              " :into " (syntax->datum s-var)
              " :if " (syntax->datum s-cond-expr))
             ]

            [(setup)
             (remove
              #f
              (list
               #'(var make-hash-table)
               (cond
                [s-count #f]
                [else #'(var-collector make-hash-table)])
               ))]

            [(iteration-body)
             (cons
              (cond
               [s-count
                #'(when cond-expr
                    (hashtable-update! var
                                       key
                                       (lambda (pre)
                                         (fx+ pre 1))
                                       0))
                ]
               [else
                #'(when cond-expr
                    (hashtable-update! var-collector key
                                       (lambda (tmp-collector)
                                         ((car tmp-collector) value)
                                         tmp-collector)
                                       (list-collector '())))])
              (car args))
             ]
            [(pre-finally)
             (cond
              [s-count (car args)]
              [else
               (cons
               #'(when var-collector
                   (let-values ([(tmp-keys tmp-values) (hashtable-entries var-collector)])
                     (vector-for-each
                      (lambda (tmp-key tmp-value)
                        (hashtable-set! var tmp-key ((cadr tmp-value))))
                      tmp-keys tmp-values))
                   (set! var-collector #f))
               (car args))])]
            [else (apply default-plugin #'make-collect-plugin method args)])))))

  (define (loop/core/group-by original-e)
    (let loop ([e original-e])
      (syntax-case e (:count :group :by :if :when :unless :into :make-hash-table)
        [(k :group expr :by key rest ...)
         (with-syntax ([return-value (loop-return-value #'k)])
           (loop #'(k (:group (:group . expr) (:by . key) (:into . return-value))
                      rest ...)))
         ]

        [(k (:group (prop . value) ...) :if cond-expr rest ...)
         (loop #'(k (:group (:if . cond-expr) (prop . value) ...) rest ...))]

        [(k (:group (prop . value) ...) :when cond-expr rest ...)
         (loop #'(k (:group (:if . cond-expr) (prop . value) ...) rest ...))]

        [(k (:group (prop . value) ...) :unless cond-expr rest ...)
         (loop #'(k (:group (:if . (not cond-expr)) (prop . value) ...) rest ...))]

        [(k (:group (prop . value) ...) :into var rest ...)
         (loop #'(k (:group (:into . var) (prop . value) ...) rest ...))]

        [(k (:group (prop . value) ...) :make-hash-table make-hash-table rest ...)
         (loop #'(k (:group (:make-hash-table . make-hash-table) (prop . value) ...) rest ...))]

        [(k (:group (prop . value) ...) rest ...)
         (values (make-group-by-plugin #'((prop . value) ...))
                 #'(k rest ...))]

        [(k rest ...)
         (values #f e)
         ]))))
