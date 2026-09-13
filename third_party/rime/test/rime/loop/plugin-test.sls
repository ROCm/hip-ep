#!r6rs
(library (test rime loop plugin-test)
  (export main)
  (import (rnrs (6))
          (rime unit-test)
          (rime loop)
          (rime loop plugin)
          (rime logging)
          )
  (define-test test:for-i-in-123
    (LOOP-SKELETONE
     a-name
     ()                        ;; setup
     ()                        ;; recur
     ()                        ;; outer
     ([xs '(1 2 3) (cdr xs)])  ;; iteration
     ()                        ;; inner
     ((not (null? xs)))        ;; condition
     ([x (car xs)])            ;; inner if true
     ((logger :info "x = " x)) ;; body
     (1))) ; finnay

  (define (main)
    (run-all-tests)))
