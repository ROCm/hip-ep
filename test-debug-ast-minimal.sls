#!r6rs
;; Minimal reproduction of "encountered raw symbol" error

(library (test-debug-ast-minimal)
  (export test-macro make-dummy-record dummy-record?)
  (import (except (rnrs) =))

  ;; Simple record type
  (define-record-type dummy-record
    (fields name value))

  ;; Keyword
  (define-syntax :debug (lambda (x) (syntax-violation 'keyword "misplaced" x)))

  ;; Macro that generates a definition
  (define-syntax test-macro
    (lambda (stx)
      (syntax-case stx (:debug)
        ;; Pattern with :debug flag before function name
        [(_ :debug fname rest ...)
         #`(define fname
             (make-dummy-record 'fname "dummy"))]

        ;; Pattern without :debug
        [(_ fname rest ...)
         #`(define fname "no-debug")]))))
