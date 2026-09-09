#!r6rs
(library (rime pretty-print)
  (export pretty-print pretty-print-to-string)
  (import (rnrs (6))
          (rnrs eval (6))
          (rime rime-0))
  (cond-expand
   [(library (ice-9 pretty-print))
    (define pretty-print (eval 'pretty-print (environment '(ice-9 pretty-print))))]
   [(library (chezscheme))
    (define pretty-print (eval 'pretty-print (environment '(chezscheme))))]
   [else (define pretty-print display)])
  (define (pretty-print-to-string obj)
     (call-with-string-output-port
      (lambda (port)
        (pretty-print obj port)))))
