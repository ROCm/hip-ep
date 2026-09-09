(library (rime loop join-string)
  (export loop/core/join-string)
  (import (rnrs (6))
          (rime loop plugin)
          (rime loop keywords))
  (define (make-join-string-plugin s-expr s-seperator s-var s-cond-expr)
    (let* ([s-port-and-extractor (new-sym s-var "join-string:port-and-extractor")]
           [s-port (new-sym s-var "join-string:port")]
           [s-extractor (new-sym s-var "join-string:extractor")]
           [s-iter-sep (new-sym s-var "join-string:seperator")])
      (with-syntax ([expr s-expr]
                    [port-and-extractor s-port-and-extractor]
                    [port s-port]
                    [extractor s-extractor]
                    [iter-sep s-iter-sep]
                    [seperator s-seperator]
                    [var s-var]
                    [cond-expr s-cond-expr])
        (lambda (method . args)
          (case method
            [(debug)
             (object-to-string
              ":join-string " (syntax->datum #'expr)
              " :seperator " (syntax->datum #'seperator)
              " :into " (syntax->datum #'var)
              " :if " (syntax->datum #'cond-expr)
              )]

            [(setup)
             (list #'(port-and-extractor (call-with-values open-string-output-port list))
                              #'(port (car port-and-extractor))
                              #'(extractor (cadr port-and-extractor))
                              #'(var "")
                              #'(iter-sep ""))
             ]

            [(iteration-body)
             (with-syntax ([nport s-port]
                           [(rest-body ...) (car args)])
               (list #'(begin
                         (when cond-expr
                           (display iter-sep nport)
                           (display expr port)
                           (set! iter-sep seperator))
                         rest-body ...)))]

            [(pre-finally)
             (cons
              #'(when extractor
                  (set! var (extractor))
                  (set! extractor #f))
              (car args))]

            [else (apply default-plugin #'make-join-string-plugin method args)])))))

  (define (loop/core/join-string original-e)
    (let loop ([e original-e])
      (syntax-case e (:join-string :if :when :unless :into :seperator)
        [(k (:join-string) expr rest ...)
         (loop #'(k (:join-string expr) rest ...))
         ]

        [(k (:join-string expr) :seperator seperator rest ...)
         (loop #'(k (:join-string expr :seperator seperator) rest ...))]

        [(k (:join-string expr) rest ...)
         (loop #'(k (:join-string expr :seperator "") rest ...))]

        [(k (:join-string expr :seperator separator) :into var rest ...)
         (loop #'(k (:join-string expr :seperator separator :into var) rest ...))]

        [(k (:join-string expr :seperator separator) rest ...)
         (with-syntax ([ret (loop-return-value #'k)])
           (loop #'(k (:join-string expr :seperator separator :into ret)  rest ...)))]

        [(k (:join-string expr :seperator separator :into var) :if cond-expr rest ...)
         (loop #'(k (:join-string expr :seperator separator :into var :if cond-expr) rest ...))]

        [(k (:join-string expr :seperator separator :into var) :when cond-expr rest ...)
         (loop #'(k (:join-string expr :seperator separator :into var :if cond-expr) rest ...))]

        [(k (:join-string expr :seperator separator :into var) :unless cond-expr rest ...)
         (loop #'(k (:join-string expr :seperator separator :into var :if (not cond-expr)) rest ...))]

        [(k (:join-string expr :seperator separator :into var) rest ...)
         (loop #'(k (:join-string expr :seperator separator :into var :if #t) rest ...))]

        [(k (:join-string expr :seperator separator :into var :if cond-expr) rest ...)
         (values (make-join-string-plugin #'expr #'separator #'var #'cond-expr)
                 #'(k rest ...))]

        [(k :join-string expr rest ...)
         (loop #'(k (:join-string) expr rest ...))]

        [(k rest ...)
         (values #f e)
         ]))))
