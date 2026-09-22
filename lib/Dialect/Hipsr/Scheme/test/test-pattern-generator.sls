#!r6rs
;;===----------------------------------------------------------------------===;;
;; Test Pattern Generator - Generates patterns from test-pattern-bodies.scm
;;===----------------------------------------------------------------------===;;

(library (test test-pattern-generator)
  (export define-test-patterns)
  (import (except (rnrs) =)
          (for (only (chezscheme) include call-with-input-file read) expand)
          (for (mlir pattern-macro) expand))

  ;; Macro: define-test-patterns
  ;; Reads test-pattern-bodies.scm and generates patterns with specified flags
  (define-syntax define-test-patterns
    (lambda (x)
      (define (read-pattern-bodies)
        (call-with-input-file "test-pattern-bodies.scm" read))

      (define (make-pattern-name base-name suffix)
        (string->symbol (string-append (symbol->string base-name) suffix)))

      (syntax-case x ()
        ;; No debug flags: (define-test-patterns suffix)
        [(_ suffix)
         (string? (syntax->datum #'suffix))
         (let* ([patterns (read-pattern-bodies)]
                [suffix-str (syntax->datum #'suffix)])
           (with-syntax ([((pattern-name pattern-body ...) ...)
                          (datum->syntax x patterns)])
             (with-syntax ([(full-name ...)
                            (map (lambda (name)
                                   (datum->syntax x (make-pattern-name (syntax->datum name) suffix-str)))
                                 (syntax->list #'(pattern-name ...)))])
               #'(begin
                   (define-conversion-pattern full-name
                     pattern-body ...)
                   ...))))]

        ;; With debug flag: (define-test-patterns :debug-parse suffix)
        [(_ debug-flag suffix)
         (and (identifier? #'debug-flag)
              (string? (syntax->datum #'suffix)))
         (let* ([patterns (read-pattern-bodies)]
                [suffix-str (syntax->datum #'suffix)])
           (with-syntax ([((pattern-name pattern-body ...) ...)
                          (datum->syntax x patterns)])
             (with-syntax ([(full-name ...)
                            (map (lambda (name)
                                   (datum->syntax x (make-pattern-name (syntax->datum name) suffix-str)))
                                 (syntax->list #'(pattern-name ...)))])
               #'(begin
                   (define-conversion-pattern debug-flag full-name
                     pattern-body ...)
                   ...))))]))))
