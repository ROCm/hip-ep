#!r6rs
(library (rime logging __logger)
  (export logger
          object-to-string
          log-level
          :source-location
          :who
          :level
          :trace
          :info
          :debug
          :warning
          :error
          :critical
          )
  (import (rnrs (6))
          (rime syntax __syntax-location)
          (rime syntax __define-auxiliary-syntax))
  (define-auxiliary-syntax
    :source-location
    :who
    :level
    :trace
    :debug
    :info
    :warning
    :error
    :critical
    )
  (define log-level
    (let ([the-log-level 300])
      (case-lambda
        [() the-log-level]
        [(level)
         (let ([old-level the-log-level])
           (set! the-log-level
                 (cond
                  [(symbol? level)
                   (case level
                     [(trace) 500]
                     [(debug) 400]
                     [(info) 300]
                     [(warning) 200]
                     [(error) 100]
                     [(critical) 0]
                     )]
                  [(fixnum? level)
                   (* level 100)]))
           old-level)])))

  (define-syntax logger
    (lambda (e)

      (syntax-case e (:header
                      :level-label
                      :level-number
                      :source-location
                      :who
                      :trace
                      :info
                      :debug
                      :warning
                      :error
                      :critical
                      )
        [(logger expr exprs ...)
         (not (free-identifier=? #'expr #':header))
         #'(logger :header ((:source-location . logger)
                            (:who . logger)
                            (:level-number . 400)
                            (:level-label . "I")
                            ) expr exprs ...)]

        [(logger :header ((k . v) ...) :trace exprs ...)
         #'(logger :header ((:level-number . 500)
                            (:level-label . "TRCE") (k . v) ...) exprs ...)]

        [(logger :header ((k . v) ...) :debug exprs ...)
         #'(logger :header ((:level-number . 400)
                            (:level-label . "DEBUG") (k . v) ...) exprs ...)]

        [(logger :header ((k . v) ...) :info exprs ...)
         #'(logger :header ((:level-number . 300)
                            (:level-label . "INFO") (k . v) ...) exprs ...)]

        [(logger :header ((k . v) ...) :warning exprs ...)
         #'(logger :header ((:level-number . 200)
                            (:level-label . "WARN") (k . v) ...) exprs ...)]

        [(logger :header ((k . v) ...) :error exprs ...)
         #'(logger :header ((:level-number . 100)
                            (:level-label . "ERR") (k . v) ...) exprs ...)]

        [(logger :header ((k . v) ...) :critical exprs ...)
         #'(logger :header ((:level-number . 0)
                            (:level-label . "FATAL") (k . v) ...) exprs ...)]

        [(logger :header ((k . v) ...) :who who exprs ...)
         #'(logger :header ((:who . who) (k . v) ...) exprs ...)]

        [(logger :header ((k . v) ...) :source-location loc exprs ...)
         #'(logger :header ((:source-location . loc) (k . v) ...) exprs ...)]

        [(logger :header ((k . v) ...)  exprs ...)
         (let ([header #'((k . v) ...)])
           (define (assq key)
             (cdr (assp (lambda (k) (free-identifier=? k key)) header)))
           (define (source-location)
             (let ([loc (assq #':source-location)])
               (cond
                [(string? (syntax->datum loc))
                 (syntax->datum loc)]
                [else (syntax-location-as-string loc)])))

           (with-syntax ([level-number (assq #':level-number)]
                         [level-label (assq #':level-label)]
                         [file:line:column (source-location)])
             #'(if (fx<=? level-number (log-level))
                   (let ([port (current-output-port)])
                     (display file:line:column port)
                     (display " " port)
                     (display level-label port)
                     (display " " port)
                     (display exprs port) ...
                     (newline port)))))]
        )))
  (define (object-to-string . args)
    (call-with-string-output-port
     (lambda (port)
       (let loop ([args args])
         (unless (null? args)
           (let ([arg (car args)])
             (cond
              [(and (pair? arg)
                    (eq? '~s (car arg)))
               (put-datum port (cdr arg))]
              [(string? arg) (put-string port arg)]
              [(char? arg) (put-char port arg)]
              [else (put-datum port arg)])
             (loop (cdr args)))))))))
