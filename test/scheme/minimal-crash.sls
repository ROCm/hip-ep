#!r6rs
(library (minimal-crash)
  (export test-crash test-working)
  (import (chezscheme))

  ;; This declaration causes crash during library loading
  (define crash-func
    (foreign-procedure "crash_func" (unsigned-64) void))

  ;; This works fine
  (define working-func
    (foreign-procedure "working_func" (unsigned-64) void))

  (define (test-crash)
    (display "Calling crash-func...\n")
    (crash-func 12345)
    (display "crash-func succeeded!\n"))

  (define (test-working)
    (display "Calling working-func...\n")
    (working-func 67890)
    (display "working-func succeeded!\n"))
)
