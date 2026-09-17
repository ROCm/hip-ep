#!r6rs
;;===----------------------------------------------------------------------===;;
;; Simple Test Framework for Scheme Code
;;===----------------------------------------------------------------------===;;

(library (test test-framework)
  (export
    test-begin
    test-end
    test-equal
    test-assert
    test-error)

  (import (rnrs (6)))

  ;;===--------------------------------------------------------------------===;;
  ;; Test State
  ;;===--------------------------------------------------------------------===;;

  (define *test-suite-name* #f)
  (define *test-pass-count* 0)
  (define *test-fail-count* 0)

  ;;===--------------------------------------------------------------------===;;
  ;; Test Lifecycle
  ;;===--------------------------------------------------------------------===;;

  (define (test-begin suite-name)
    (set! *test-suite-name* suite-name)
    (set! *test-pass-count* 0)
    (set! *test-fail-count* 0)
    (display (string-append "\n=== Running: " suite-name " ===\n")))

  (define (test-end)
    (let ([total (+ *test-pass-count* *test-fail-count*)])
      (display (string-append "\n=== Results: " *test-suite-name* " ===\n"))
      (display (string-append "  Passed: " (number->string *test-pass-count*) 
                             "/" (number->string total) "\n"))
      (when (> *test-fail-count* 0)
        (display (string-append "  FAILED: " (number->string *test-fail-count*) "\n"))
        (exit 1))
      (display "\n")))

  ;;===--------------------------------------------------------------------===;;
  ;; Test Assertions
  ;;===--------------------------------------------------------------------===;;

  (define (test-equal description actual expected)
    (if (equal? actual expected)
        (begin
          (set! *test-pass-count* (+ *test-pass-count* 1))
          (display (string-append "  ✓ " description "\n")))
        (begin
          (set! *test-fail-count* (+ *test-fail-count* 1))
          (display (string-append "  ✗ " description "\n"))
          (display (string-append "    Expected: " (format-value expected) "\n"))
          (display (string-append "    Got:      " (format-value actual) "\n")))))

  (define (test-assert description condition)
    (if condition
        (begin
          (set! *test-pass-count* (+ *test-pass-count* 1))
          (display (string-append "  ✓ " description "\n")))
        (begin
          (set! *test-fail-count* (+ *test-fail-count* 1))
          (display (string-append "  ✗ " description " (expected true)\n")))))

  (define (test-error description thunk)
    (guard (ex [else
                (set! *test-pass-count* (+ *test-pass-count* 1))
                (display (string-append "  ✓ " description " (error caught)\n"))])
      (thunk)
      (set! *test-fail-count* (+ *test-fail-count* 1))
      (display (string-append "  ✗ " description " (expected error)\n"))))

  ;;===--------------------------------------------------------------------===;;
  ;; Helpers
  ;;===--------------------------------------------------------------------===;;

  (define (format-value v)
    (call-with-string-output-port
      (lambda (p) (write v p))))

) ;; end library
