#!r6rs
(library (test rime rime-0 cond-expand-test)
  (export main)
  (import (rnrs (6))
          (rime unit-test)
          (rime rime-0)
          (rime logging)
          )
  (define (main)
    (run-all-tests))
  (define-test
    test-print-darwin-windows-linux
    (cond-expand
     [darwin
      (logger :info "hello OS=darwin")]
     [windows
      (logger :info "hello OS=windows")]
     [linux
      (logger :info "hello OS=linux")]
     [else
      (logger :info "hello OS=UNKNOWN")]
     ))

  (define-test
    test-print-imp
    (cond-expand
     [guile
      (logger :info "hello SCHEME=guile")]
     [chezscheme
      (logger :info "hello SCHEME=chezscheme")]
     [else
      (logger :info "hello SCHEME=UNKNOWN")]
     )))
