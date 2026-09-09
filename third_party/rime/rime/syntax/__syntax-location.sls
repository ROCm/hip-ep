#!r6rs
(library (rime syntax __syntax-location)
  (export syntax-location syntax-location-as-string)
  (import (rnrs (6))
          (rnrs eval (6))
          (rime rime-0)
          )
  (define (unknown-loc)
    (list "<UNKNOWN>" 0 0))
  (cond-expand
   [chezscheme
    (define (is-record? obj type)
      (and (record? obj) (eq? (record-type-name (record-rtd obj)) type) obj))
    (define record-get-field
      (case-lambda
        [(obj) obj]
        [(obj type) (is-record? obj type)]
        [(obj type index)
         (and (is-record? obj type) ((record-accessor (record-rtd obj) index) obj))]
        [(obj type index . args)
         (let ([field (record-get-field obj type index)])
           (if field
               (apply record-get-field field args)
               #f))]))

    (define (get-column-line-number file-name position)
      (call-with-port (open-file-input-port file-name
                                            (file-options)
                                            (buffer-mode block)
                                            (native-transcoder))
        (lambda (port)
          (let loop ([pos 0]
                     [line-number 1]
                     [column-number 0])
            (let ([ch (get-char port)])
              (cond
               [(eof-object? ch) #f]
               [(fx>=? pos position) (cons line-number column-number)]
               [(char=? ch #\newline) (loop (fx+ pos 1) (fx+ 1 line-number) 1)]
               [else (loop (fx+ pos 1) line-number (fx+ 1 column-number))]
               ))))))

    (define (syntax-location syn)
      (cond
       [(record-get-field syn 'syntax-object 0 'annotation 1 'source)
        =>
        (lambda (source)
          (let* ([source-name (record-get-field source 'source 0 'source-file-descriptor 0)]
                 [bfp (record-get-field source 'source 1)]
                 [efp (record-get-field source 'source 2)]
                 [line:column (get-column-line-number source-name bfp)])
            (list source-name (car line:column) (cdr line:column))))]
       [else (unknown-loc)]))]
   [guile
    (define (syntax-location syn)
      (let ([syntax-source (eval 'syntax-source (environment '(guile)))])
        (let ([src (map cdr (syntax-source syn))])
          (if (fx=? 3 (length src))
              (begin
                (list (list-ref src 0)
                      (fx+ (list-ref src 1) 1)
                      (fx+ (list-ref src 2) 1)))
              (unknown-loc)))))]
   [else
    (define (syntax-location syn)
      (list "not" 0 0))])

  (define (syntax-location-as-string syn)
    (apply
     (lambda (file line column)
       (call-with-string-output-port
        (lambda (port)
          (display file port)
          (display ":" port)
          (display line port)
          (display ":" port)
          (display column port)
          (display ":" port))))
     (syntax-location syn)))
  ;;
  )
