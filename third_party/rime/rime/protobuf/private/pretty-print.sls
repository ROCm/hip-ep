(library (rime protobuf private pretty-print)
  (export pretty-print)
  (import (rnrs (6))
          (rime logging)
          (rime protobuf private platform))
  (cond-expand
   [(library (ice-9 pretty-print))
    (use (ice-9 pretty-print)
         (rename pretty-print $pretty-print))]
   [(library (chezscheme))
    (use (chezscheme)
         (rename pretty-print $pretty-print))]
   [else (define $pretty-print #f)])

  (define pretty-print
    (if (procedure? $pretty-print)
        (lambda  (expr)
          (call-with-string-output-port
           (lambda (port)
             ($pretty-print expr port))))
        object-to-string)))
