#!r6rs
(library (rime loop plugin)
  (export default-plugin
          LOOP-SKELETONE)
  (import (rnrs (6))
          (rime logging))

  ;; (loop ...) is expanded into something similar as below.
  ;;
  (define-syntax LOOP-SKELETONE
    (syntax-rules ()
      [(k
        <<recur-name>>
        ;; each plugin should return a list of bindings to build the
        ;; following environment
        ;; 1. setup
        ;; 2. recur
        ;; 3. outer-iteration
        ;; 4. iteration
        ;; 5. inner-interation
        ;; 6. inner-if-true-interation
        ([<<setup-binding>> <<setup-value>>] ...)
        ([<<recur-binding>> <<recur-value>>] ...)
        ([<<outer-binding>> <<outer-value>>] ...)
        ([<<iteration-binding>> <<iteration-value>> <<step-expr>>] ...)
        ([<<inner-binding>> <<inner-value>>] ...)
        (<<continue-condition>> ...)
        ([<<inner-if-true-binding>> <<inner-if-true-value>>] ...)
        (<<iteration-body>> ...)
        (<<finally>> ...)
        )
       (let* ( ;; setup-environment
              [<<setup-binding>> <<setup-value>>] ... )
         (let <<recur-name>> ( ;;  recur-environment
                              [<<recur-binding>> <<recur-value>>] ...)
              (let* ( ;; outer-iter-environment
                     [<<outer-binding>> <<outer-value>>] ...)
                (let <<iter-name>> ( ;; iter-environment
                                    [<<iteration-binding>> <<iteration-value>>] ...)
                     (let* ( ;;inner-iter-environment
                            [<<inner-binding>> <<inner-value>>] ...)
                       (if (and <<continue-condition>> ...)
                           (let* ( ;;inner-iter-yes-environment
                                  [<<inner-if-true-binding>> <<inner-if-true-value>>] ...
                                  )
                             <<iteration-body>> ...
                             (<<iter-name>> <<step-expr>> ...))
                           (begin <<finally>> ...)))))))
       ]))

  (define (default-plugin who method . args)
    (case method
      [(debug)
       "<NOT DEFINED"]
      [(setup)
       (list)]
      [(recur)
       (list)]
      [(outer-iteration)
       (list)]
      [(iteration)
       (list)]
      [(inner-iteration)
       (list)]
      [(continue-condition)
       #t]
      [(inner-if-true)
       (list)]
      [(iteration-body)
       (car args)]
      [(pre-finally)
       (car args)]
      [(finally)
       (car args)]
      [else (syntax-violation who (call-with-string-output-port
                                   (lambda (port)
                                     (display "never goes here" port)
                                     (display " method=")
                                     (display method))) method)])))
