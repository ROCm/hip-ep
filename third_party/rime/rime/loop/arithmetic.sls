#!r6rs
(library (rime loop arithmetic)
  (export loop/core/arithmetic)
  (import (rnrs (6))
          (rime loop keywords)
          (rime loop plugin))
  (define (build-for-as-arithmetic r-arithmetic)
    (lambda (method . args)
      (let ([s-end (for-as-arithmetic-s-end r-arithmetic)]
            [less (for-as-arithmetic-less r-arithmetic)]
            [inclusive (for-as-arithmetic-inclusive r-arithmetic)])
        (with-syntax ([var (for-as-arithmetic-s-var r-arithmetic)]
                      [init (for-as-arithmetic-s-init r-arithmetic)]
                      [end s-end]
                      [step (for-as-arithmetic-s-step r-arithmetic)])
          (case method
            [(debug)
             (object-to-string
              ":for " (syntax->datum #'var)
              " :from " (syntax->datum #'init)
              " :to " (syntax->datum #'end)
              " :by " (if less "<" ">")
              (if inclusive "=" ""))
             ]
            [(iteration)
             (list #'[var init (fx+ var step)])]
            [(continue-condition)
             (with-syntax ([compare (cond
                                     [(and less (not inclusive)) #'fx<?]
                                     [(and less inclusive) #'fx<=?]
                                     [(and (not less) inclusive) #'fx>=?]
                                     [(and (not less) (not inclusive)) #'fx>?]
                                     [else "oops"])])
               ;; todo evaluate end only once
               (if s-end #'(compare var end) #t))
             ]
            [else (apply default-plugin #'build-for-as-arithmetic method args)])))))

  (define-record-type for-as-arithmetic
    (fields s-var
            (mutable s-init)
            (mutable s-end)
            (mutable s-step)
            (mutable less)
            (mutable inclusive))
    (protocol
     (lambda (new)
       (lambda (var)
         (new var #'0 #'0 #'1 #t #t)))))

  (define (loop/core/arithmetic e)
    (let ([by 1]
          [from 0]
          [to #f]
          [less #t]
          [inclusive #f]
          [parse-ok #f])
      (let repeat ([e e])
        (syntax-case e (:for :repeat
                             :from :downfrom :upfrom :to :downto :upto :below :above :by
                             )
          [(k :for i :from expr1 rest ...)
           (begin
             (set! from #'expr1)
             (set! parse-ok #t)
             (repeat #'(k :for i rest ...)))
           ]
          [(k :for i :upfrom expr1 rest ...)
           (begin
             (set! from #'expr1)
             (set! by 1)
             (set! parse-ok #t)
             (repeat #'(k :for i rest ...)))
           ]          [(k :for i :downfrom expr1 rest ...)
           (begin
             (set! from #'expr1)
             (set! by -1)
             (set! parse-ok #t)
             (repeat #'(k :for i rest ...)))
           ]
          [(k :for i :to expr2 rest ...)
           (begin
             (set! to #'expr2)
             (set! inclusive #t)
             (set! parse-ok #t)
             (repeat #'(k :for i rest ...)))
           ]
          [(k :for i :upto expr2 rest ...)
           (begin
             (set! to #'expr2)
             (set! parse-ok #t)
             (set! inclusive #t)
             (repeat #'(k :for i rest ...)))
           ]
          [(k :for i :downto expr2 rest ...)
           (begin
             (set! to #'expr2)
             (set! parse-ok #t)
             (set! by -1)
             (repeat #'(k :for i rest ...)))
           ]
          [(k :for i :below expr2 rest ...)
           (begin
             (set! to #'expr2)
             (set! parse-ok #t)
             (repeat #'(k :for i rest ...)))
           ]
          [(k :for i :above expr2 rest ...)
           (begin
             (set! to #'expr2)
             (set! parse-ok #t)
             (repeat #'(k :for i rest ...)))
           ]
          [(k :for i :by expr3 rest ...)
           (begin
             (set! by #'expr3)
             (set! parse-ok #t)
             (repeat #'(k :for i rest ...)))
           ]

          [(k :for i rest ...)
           (let ([rest-list #'(rest ...)])
             (and parse-ok
                  (or (null? rest-list)
                      (not (one-of (car rest-list)
                                   (list #':by
                                         #':from #':upfrom #':downfrom
                                         #':to #':downto #':upto
                                         #':below #':above))))))
           (let ([sub-clause (make-for-as-arithmetic #'i)])
             (for-as-arithmetic-s-init-set! sub-clause from)
             (for-as-arithmetic-s-end-set! sub-clause to)
             (for-as-arithmetic-s-step-set! sub-clause by)
             (for-as-arithmetic-less-set! sub-clause less)
             (for-as-arithmetic-inclusive-set! sub-clause inclusive)
             (values (build-for-as-arithmetic sub-clause)  #'(k rest ...)))
           ]
          [(k :repeat expr rest ...)
           (let ([sub-clause (make-for-as-arithmetic (new-sym #'k "repeat-anonymous-var"))])
             (for-as-arithmetic-s-init-set! sub-clause 0)
             (for-as-arithmetic-s-end-set! sub-clause #'expr)
             (for-as-arithmetic-s-step-set! sub-clause #'1)
             (for-as-arithmetic-less-set! sub-clause #t)
             (for-as-arithmetic-inclusive-set! sub-clause #f)
             (values (build-for-as-arithmetic sub-clause)  #'(k rest ...)))
           ]
          [(k rest ...)
           (values #f e)
           ]
          )))))
