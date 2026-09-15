#!r6rs
(library (rime loop core)
  (export
   parse-loop-clauses
   loop-codegen
   )
  (import (rnrs (6))
          (rnrs mutable-pairs (6))
          (rime loop keywords)
          (rime pretty-print)
          (rime loop plugin)
          (rime loop arithmetic)
          (rime loop list)
          (rime loop vector)
          (rime loop string)
          (rime loop hashtable)
          (rime loop if)
          (rime loop do)
          (rime loop with)
          (rime loop recur)
          (rime loop collect)
          (rime loop break)
          (rime loop initially)
          (rime loop join-string)
          (rime loop group-by)
          (rime loop count)
          (rime loop directory)
          )

  (define (all-plugins)
    (list (cons 'loop/core/arithmetic loop/core/arithmetic)
          (cons 'loop/core/list loop/core/list)
          (cons 'loop/core/vector loop/core/vector)
          (cons 'loop/core/string loop/core/string)
          (cons 'loop/core/hashtable loop/core/hashtable)
          (cons 'loop/core/if loop/core/if)
          (cons 'loop/core/do loop/core/do)
          (cons 'loop/core/with loop/core/with)
          (cons 'loop/core/recur loop/core/recur)
          (cons 'loop/core/collect loop/core/collect)
          (cons 'loop/core/break loop/core/break)
          (cons 'loop/core/finally loop/core/finally)
          (cons 'loop/core/initially loop/core/initially)
          (cons 'loop/core/loop loop/core/loop)
          (cons 'loop/core/join-string loop/core/join-string)
          (cons 'loop/core/group-by loop/core/group-by)
          (cons 'loop/core/count loop/core/count)
          (cons 'loop/core/directory loop/core/directory)))
  (define (plugins-for-finally-clauses)
    (list (cons 'loop/core/do loop/core/do)
          (cons 'loop/core/with loop/core/with)
          (cons 'loop/core/collect loop/core/collect)
          ;; (cons 'loop/core/join-string loop/core/join-string)
          ;; (cons 'loop/core/group-by loop/core/group-by)
          ;; (cons 'loop/core/finally loop/core/finally)
          ))
  (define (parse-loop-clauses-with-plugins original-e all-plugins)
    (let ([clauses '()])
      (define (add-clause c)
        (set! clauses (cons c clauses)))
      (define (parse-clause e)
        (let loop-plugin ([plugins all-plugins])
          (if (null? plugins)
              (values #f e)
              (let ([plugin (car plugins)])
                (let ([plugin-name (car plugin)]
                      [plugin-func (cdr plugin)])
                  (let-values ([(clause rest) (plugin-func e)])
                    (if clause
                        (begin
                          (when (loop-trace-parser e)
                            (logger :info "[" plugin-name "]"
                                    " produces [" (clause 'debug) "]"
                                    " rest=" (syntax->datum rest) "\n"
                                    " e=" (syntax->datum e)))
                          (values clause rest))
                        (loop-plugin (cdr plugins)))))))))
      (let loop ([e original-e])
        (let-values ([(clause rest) (parse-clause e)])
          (cond
           [clause
            (cond
             [(procedure? clause)
              (add-clause clause)
              (loop rest)]
             [else (loop rest)])]
           [else
            (syntax-case e ()
              [((k props ...))
               (when (or #f (loop-trace-parser #'(k props ...)))
                 (logger :debug
                         " PARSING OK! "
                         (syntax->datum original-e) " => clauses: "
                         (loop-clauses-to-string "LOOP" (loop-level #'(k props ...))
                                                 (reverse clauses))
                         ))
               (values (reverse clauses) #'(k props ...))]
              [_else (syntax-violation
                      'parse-loop-clauses
                      (call-with-string-output-port
                       (lambda (port)
                         (put-string port "loop parse1 ")
                         (put-datum port (syntax->datum original-e))
                         (put-string port " error at")))
                      #'original-e
                      e)]
              )]
           )))))
  (define (parse-loop-clauses original-e)
    (parse-loop-clauses-with-plugins original-e (all-plugins)))

  (define (codegen-prologue-binding k clauses)
    (define (remove-weak bindings)
      (let ([vars '()]
            [values '()]
            [weaks '()])
        (define (push var value weak)
          (set! vars (cons var vars))
          (set! values (cons value values))
          (set! weaks (cons weak weaks)))
        (define (find-binding var)
          (let repeat ([vars vars]
                       [values values]
                       [weaks weaks])
            (if (not (null? vars))
                (if (bound-identifier=? (car vars) var)
                    (list vars values weaks)
                    (repeat (cdr vars) (cdr values) (cdr weaks)))
                #f)))
        (define (process-binding binding)
          (let repeat ([binding binding])
            (syntax-case binding ()
              [(name value)
               (repeat #'(name value #f))]
              [(name value weak)
               (let ([weak (syntax->datum #'weak)]
                     [var #'name]
                     [value #'value])
                 (let ([previous-binding (find-binding var)])
                   (cond
                    [(not previous-binding)
                     (push var value weak)]
                    [(not weak) ;; overwrite previous binding
                     (set-car! (list-ref previous-binding 1) value)
                     (set-car! (list-ref previous-binding 2) weak)
                     ]
                    [(car (list-ref previous-binding 2))
                     ;; previous also weak binding
                     (set-car! (list-ref previous-binding 1) value)
                     (set-car! (list-ref previous-binding 2) weak)
                     ]
                    ))
                 )])))
        (let repeat ([bindings  bindings])
          (cond
           [(null? bindings)
            (reverse (map list vars values))]
           [else (process-binding (car bindings))
                 (repeat (cdr bindings))]
           ))))
    (remove-weak
     (with-syntax ([loop-return-value (loop-return-value k)])
       (apply append
              (list #'(loop-return-value (if #f 0) #t))
              (map (lambda (loop-clause)
                     (loop-clause 'setup)) clauses)))))

  (define (loop-codegen-body-or-finally clauses tag)
    (let loop ([clauses clauses])
      (if (null? clauses)
          '()
          (begin
            ((car clauses) tag (loop (cdr clauses)))))))
  (define (loop-codegen-body clauses)
    (loop-codegen-body-or-finally clauses 'iteration-body))

  (define (loop-codegen-pre-finally clauses)
    (loop-codegen-body-or-finally clauses 'pre-finally))

  (define (loop-codegen-finally clauses)
    (loop-codegen-body-or-finally clauses 'finally))

  (define (loop-codegen k clauses original-e)
    (let ([code
           (with-syntax
               ([(prologue-binding ...) (codegen-prologue-binding k clauses)]
                [(inner-body ...) (loop-codegen-body clauses)])
             #'(let* (prologue-binding ...)
                 inner-body ...))])
      (when (loop-trace-codegen k)
        (logger :info " [" (syntax->datum original-e) "]"
                " expand to \n"
                (pretty-print-to-string (syntax->datum code))
                ))
      code))
  (define (loop-clauses-to-string PREFIX level clauses)
    (define space #\ )
    (define (indent level)
      (make-string (fx* 4 level) space))
    (apply string-append (map
                          (lambda (c)
                            (object-to-string
                             "\n" (indent (fx+ level 1)) PREFIX "[" level "] "
                             (c 'debug)))
                          clauses)))

  (define (display-loop-plugin level s-loop-expr clauses)
    (object-to-string
     ":loop@" level " "
     (syntax->datum s-loop-expr)
     (loop-clauses-to-string "LOOP" (fx+ level 1) clauses)))

  (define (make-loop-plugin s-k s-loop-expr s-recur-name)
    (let-values ([(clauses s-next-k) (parse-loop-clauses s-loop-expr)])
      (let ([loop-level (loop-level s-k)])
        (lambda (method . args)
          (with-syntax ()
            (case method
              [(debug)
               (display-loop-plugin loop-level s-loop-expr clauses)]
              [(setup)
               (apply append (map (lambda (c) (c 'setup)) clauses))]
              [(iteration-body)
               (cons
                (with-syntax
                    ([<<return-value>> (loop-return-value s-k)]
                     [<<recur-name>> s-recur-name]
                     [([<<recur-binding>> <<recur-value>>] ...)
                      (apply append (map (lambda (f) (f 'recur)) clauses))]
                     [([<<outer-binding>> <<outer-value>>] ...)
                      (apply append (map (lambda (f) (f 'outer-iteration)) clauses))]
                     [<<iter-name>> (new-sym s-k "LOOP-REPEAT")]
                     [([<<iteration-binding>> <<iteration-value>> <<step-expr>>] ...)
                      (apply append (map (lambda (f) (f 'iteration)) clauses))]
                     [([<<inner-binding>> <<inner-value>>] ...)
                      (apply append (map (lambda (f) (f 'inner-iteration)) clauses))]
                     [(<<continue-condition>> ...)
                      (remove #t (map (lambda (f) (f 'continue-condition)) clauses))]
                     [([<<inner-if-true-binding>> <<inner-if-true-value>>] ...)
                      (apply append (map (lambda (f) (f 'inner-if-true)) clauses))]
                     [(<<iteration-body>> ...) (loop-codegen-body clauses)]
                     [(<<pre-finally>> ...) (loop-codegen-pre-finally clauses)]
                     [(<<finally>> ...) (loop-codegen-finally clauses)])
                  #'(let <<recur-name>> ([<<recur-binding>> <<recur-value>>] ...)
                         (let* ([<<outer-binding>> <<outer-value>>] ...)
                           (let <<iter-name>> ([<<iteration-binding>> <<iteration-value>>] ...)
                                (let* ([<<inner-binding>> <<inner-value>>] ...)
                                  (if (and <<continue-condition>> ...)
                                      (let* ([<<inner-if-true-binding>> <<inner-if-true-value>>] ...)
                                        <<iteration-body>> ...
                                        (<<iter-name>> <<step-expr>> ...))
                                      (begin <<pre-finally>> ...
                                             'pre-finally-ended ;; only for debugging purpose
                                             <<return-value>>
                                             <<finally>> ...)))))))
                (car args))
               ]
              [else (apply default-plugin #'make-loop-plugin method args)]))))))

  (define (loop/core/loop original-e)
    (let loop ([e original-e])
      (syntax-case e (:name :loop)
        [(k (:loop :name <var> clauses ...) rest ...)
         (identifier? #'<var>)
         (with-syntax ([next-k (loop-level++ #'k)])
           (values (make-loop-plugin #'k #'(next-k clauses ...) #'<var>)
                   #'(k rest ...)))
         ]
        [(k (:loop clauses ...) rest ...)
         (with-syntax ([name (new-sym #'k "RECUR")])
           (loop #'(k (:loop :name name clauses ...) rest ...)))
         ]
        [(k rest ...)
         (values #f e)
         ])))
  (define (make-finally-plugin s-k s-expr s-return-value finally-clauses)
    (let ([loop-level (loop-level s-k)])
      (with-syntax ([expr s-expr]
                    [return-value s-return-value])
        (lambda (method . args)
          (with-syntax ([expr s-expr]
                        [:return-value s-return-value])
            (case method
              [(debug)
               (object-to-string
                ":finally " (syntax->datum #'expr)  " "
                (loop-clauses-to-string "FINALLY" loop-level finally-clauses)
                )]
              [(finally)
               (with-syntax ([(inner-body ...) (loop-codegen-body finally-clauses)])
                 (list #'(begin (begin (set! :return-value expr) :return-value)
                                inner-body ...)))]
              [else (apply default-plugin #'make-finally-plugin method args)]))))))

  (define (loop/core/finally original-e)
    (let loop ([e original-e])
      (syntax-case e (:finally)
        [(k :finally expr rest ...)
         (not (keyword? #'expr))
         (let-values ([(finally-clauses s-rest) (parse-loop-clauses-with-plugins #'(k rest ...) (plugins-for-finally-clauses))])
           (with-syntax ([ret (loop-return-value #'k)])
             (values (make-finally-plugin #'k #'expr (loop-return-value #'k) finally-clauses)
                     #'(k))))]
        [(k :finally other-clause-start-with-a-keyword rest ...)
         ;; go to pattern 0.
         (with-syntax ([ret (loop-return-value #'k)])
           (loop #'(k :finally ret other-clause-start-with-a-keyword rest ...)))
         ]
        [(k rest ...)
         (values #f e)
         ]))))
