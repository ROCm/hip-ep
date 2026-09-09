#!r6rs
(library (rime loop list-collector)
  (export list-collector)
  (import (rnrs (6))
          (rnrs mutable-pairs (6)))
  (define (list-collector init-value)
    (let [(tmp-var init-value)
          (tmp-tail '())]
      (list
       (lambda (x)
         (if (null? tmp-var)
             (set! tmp-var (cons x '()))
             (begin
               (when (null? tmp-tail)
                 (let find-tail ([tmp-var tmp-var])
                   (if (not (null? (cdr tmp-var)))
                       (find-tail (cdr tmp-var))
                       (set! tmp-tail tmp-var))))
               (set-cdr! tmp-tail (cons x '()))
               (set! tmp-tail (cdr tmp-tail)))))
       (lambda () tmp-var)))))
