#!r6rs
(library (rime unit-test __define-test)
  (export define-test run-all-tests)
  (import (rnrs (6))
          (rime logging)
          (rime syntax __syntax-location)
          (rime rime-98)
          )
  (define _store '())
  (define (all-tests)
    (reverse _store))
  (define (add-test name value)
    (set! _store (cons (cons name value) _store)))

  (define-syntax define-test
    (lambda (e)
      (syntax-case e ()
        [(k name x xs ...)
         (with-syntax [(LOC (syntax-location-as-string #'name))
                       (test-name (string-append
                                   (symbol->string (syntax->datum #'name))
                                   "@"
                                   (syntax-location-as-string #'name)
                                   ))]
           #'(define name
               (let ([value (lambda (method)
                              (case method
                                [(run)
                                 x xs ... #t]
                                [(loc)
                                 LOC]
                                )
                              )])
                 (add-test test-name value)
                 (logger :info :source-location LOC "ADD TEST [" 'name "]")
                 value)))])))
  (define (run-all-tests)
    (log-level (read (open-string-input-port (or (get-environment-variable "VERBOSE") "3"))))
    (let ((tests (all-tests)))
      (logger :trace "START TO RUN ALL TEST:")
      (for-all run-test tests)))

  (define (run-test test)
    (let ([name (car test)]
          [func (cdr test)]
          [succeeded? #f]
          [exception #f])
      (logger :trace "TEST [" name "] START!")
      (set! succeeded? (func 'run))
      (logger :info "TEST [" name "]" (if succeeded? " OK" " FAIL"))
      succeeded?)))
