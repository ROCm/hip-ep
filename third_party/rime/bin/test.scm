#!r6rs
(import (rnrs (6))
        (rnrs eval (6))
        (rime loop)
        (rime logging))
(logger :info " TEST MODULE: " (cdr (command-line)))
(let* ([import-spec
        (loop :for arg :in (cdr (command-line))
              :collect (string->symbol arg))]
       [expr `(import ',import-spec)]
       [exception #t])
  (let [(loaded (guard (exn [else (set! exception exn)#f])
                  (eval #t (environment import-spec))))]
    (unless loaded
      (logger :error "cannot load module " import-spec)
      (raise exception)
      (exit 1))
    (let ([main (guard (exn [else (set! exception exn)#f])
                  (eval 'main (environment import-spec)))])
      (if (procedure? main)
          (main)
          (logger :info "IGNORE " (cdr (command-line)) " (main) is not found")))))
