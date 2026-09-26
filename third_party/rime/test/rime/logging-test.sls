#!r6rs
(library (test rime logging-test)
  (export main)
  (import (rnrs (6))
          (rime logging)
          (rime unit-test))
  (define-test
    show-logs
    (logger :critical "log level= " (log-level))
    (logger :trace "hello trace log message")
    (logger :info "hello info log message")
    (logger :debug "hello debug log message")
    (logger :warning "hello warning log message")
    (logger :error "hello error log message")
    (logger :critical "hello critical log message"))

  (define (main)
    (run-all-tests)))
