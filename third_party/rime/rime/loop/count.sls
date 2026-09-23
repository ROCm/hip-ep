(library (rime loop count)
  (export loop/core/count)
  (import (rnrs (6))
          (rime loop plugin)
          (rime loop keywords))

  ;; Plugin for :count with hashtable grouping (:by) and custom increment (:step)
  (define (make-count-plugin props)
    (let ([s-key (assq-id ':by props #f)]
          [s-step (assq-id ':step props #'1)]
          [s-var (assq-id ':into props #f)]
          [s-cond-expr (assq-id ':if props #t)]
          [s-make-hash-table (assq-id ':make-hash-table props #'(make-eq-hashtable))])

      (with-syntax ([key s-key]
                    [step-expr s-step]
                    [var s-var]
                    [cond-expr s-cond-expr]
                    [make-hash-table s-make-hash-table])
        (lambda (method . args)
          (case method
            [(debug)
             (object-to-string
              ":count"
              (if s-key (list " :by " (syntax->datum s-key)) "")
              (if s-step (list " :step " (syntax->datum s-step)) "")
              (if s-var (list " :into " (syntax->datum s-var)) "")
              " :if " (syntax->datum s-cond-expr))]

            [(setup)
             (remove
              #f
              (list
               (cond
                [s-key #'(var make-hash-table)]
                [else #'(var 0)])))]

            [(iteration-body)
             (cons
              (cond
               [s-key
                #'(when cond-expr
                    (hashtable-update! var
                                       key
                                       (lambda (pre)
                                         (fx+ pre 1))
                                       0))]
               [else
                #'(when cond-expr
                    (set! var (fx+ var step-expr)))])
              (car args))]

            [else (apply default-plugin #'make-count-plugin method args)])))))

  (define (loop/core/count original-e)
    (let loop ([e original-e])
      (syntax-case e (:count :if :when :unless :into :by :step :make-hash-table)
        ;; :count without :into defaults to :return-value
        [(k :count rest ...)
         (with-syntax ([return-value (loop-return-value #'k)])
           (loop #'(k (:count (:into . return-value)) rest ...)))]

        ;; :count with :into
        [(k (:count (prop . value) ...) :into var rest ...)
         (loop #'(k (:count (:into . var) (prop . value) ...) rest ...))]

        ;; :count with :by (grouping key for hashtable)
        [(k (:count (prop . value) ...) :by key rest ...)
         (loop #'(k (:count (:by . key) (prop . value) ...) rest ...))]

        ;; :count with :step (custom increment amount)
        [(k (:count (prop . value) ...) :step step-expr rest ...)
         (loop #'(k (:count (:step . step-expr) (prop . value) ...) rest ...))]

        ;; :count with :if
        [(k (:count (prop . value) ...) :if cond-expr rest ...)
         (loop #'(k (:count (:if . cond-expr) (prop . value) ...) rest ...))]

        ;; :count with :when (alias for :if)
        [(k (:count (prop . value) ...) :when cond-expr rest ...)
         (loop #'(k (:count (:if . cond-expr) (prop . value) ...) rest ...))]

        ;; :count with :unless
        [(k (:count (prop . value) ...) :unless cond-expr rest ...)
         (loop #'(k (:count (:if . (not cond-expr)) (prop . value) ...) rest ...))]

        ;; :count with :make-hash-table
        [(k (:count (prop . value) ...) :make-hash-table make-hash-table rest ...)
         (loop #'(k (:count (:make-hash-table . make-hash-table) (prop . value) ...) rest ...))]

        ;; Final: create the plugin
        [(k (:count (prop . value) ...) rest ...)
         (values (make-count-plugin #'((prop . value) ...))
                 #'(k rest ...))]

        [(k rest ...)
         (values #f e)]))))
