#!r6rs
;;===----------------------------------------------------------------------===;;
;; Test Helpers - Load and eval patterns from test-pattern-bodies.scm
;;===----------------------------------------------------------------------===;;

(library (test test-helpers)
  (export load-test-cases
          eval-pattern
          get-field
          run-one-phase
          run-phase-tests
          show-pattern-output)
  (import (chezscheme)
          (except (mlir pattern-macro) =)  ; Exclude = to avoid conflict
          (rename (rime loop) (:with :rime-with)))

  ;;=======================================================================
  ;; Load test data from cases/ directory
  ;;=======================================================================

  (define (load-test-cases)
    "Load all test cases from test/cases/*.scm directory.
     Returns list of (name . test-body) pairs."
    (let* ([files (directory-list "test/cases")]
           [scm-files (filter (lambda (f) (string-suffix? ".scm" f)) files)])
      (map (lambda (filename)
             (let* ([name (substring filename 0 (- (string-length filename) 4))]
                    [path (string-append "test/cases/" filename)]
                    [body (call-with-input-file path read)])
               (cons (string->symbol name) body)))
           scm-files)))

  (define (string-suffix? suffix str)
    (let ([slen (string-length suffix)]
          [len (string-length str)])
      (and (>= len slen)
           (string=? (substring str (- len slen) len) suffix))))

  ;;=======================================================================
  ;; Extract field from test case
  ;;=======================================================================

  (define (get-field key test-body)
    ;; Simple plist traversal - advance by 2
    ;; test-body is already the plist: (:pattern ... :expect-parse ...)
    (let loop ([rest test-body])
      (cond
        [(null? rest) #f]
        [(eq? (car rest) key) (cadr rest)]
        [else (loop (cddr rest))])))

  ;;=======================================================================
  ;; Check expectations against actual result - EXACT MATCH
  ;;=======================================================================

  (define (check-all-expectations result expectations)
    ;; Returns (passed? . failing-reason-or-#f)
    ;; Expects exact match: result must equal expectations
    (if (equal? result expectations)
        (cons #t #f)
        (cons #f "Output does not match expectations exactly")))

  ;;=======================================================================
  ;; Eval pattern with debug flag and check expectations
  ;;=======================================================================

  (define (eval-pattern name debug-flag pattern-body expectations)
    (guard (e [else
               (display (format "ERROR: Pattern ~a failed:\n" name))
               (display-condition e)
               (newline)
               #f])
      (let* ([pattern-name (string->symbol (string-append "pattern-" (symbol->string name)))]
             [full-expr (if debug-flag
                           `(define-conversion-pattern ,debug-flag ,pattern-name ,@pattern-body)
                           `(define-conversion-pattern ,pattern-name ,@pattern-body))])
        ;; Eval the pattern definition
        (eval full-expr (interaction-environment))

        ;; Get the defined pattern
        (let ([pattern-fn (eval pattern-name (interaction-environment))])
          (cond
            [(not (procedure? pattern-fn))
             (display (format "ERROR: Pattern ~a is not a procedure\n" name))
             #f]

            ;; Debug mode: call function, check expectations
            [debug-flag
             (let ([result (pattern-fn)])
               (cond
                 [(not (list? result))
                  (display (format "ERROR: Debug pattern ~a did not return a list\n" name))
                  #f]
                 ;; Check expectations if provided
                 [expectations
                  (let ([check-result (check-all-expectations result expectations)])
                    (if (car check-result)
                        #t
                        (begin
                          (display (format "  Expectation failed: ~s\n" (cdr check-result)))
                          (display (format "  Actual result:\n  "))
                          (pretty-print result)
                          #f)))]
                 ;; No expectations
                 [else #t]))]

            ;; Normal mode
            [else #t])))))

  ;;=======================================================================
  ;; Show pattern output for troubleshooting (doesn't check expectations)
  ;;=======================================================================

  (define (show-pattern-output name debug-flag pattern-body)
    (guard (e [else
               (display (format "ERROR: Pattern ~a failed:\n" name))
               (display-condition e)
               (newline)])
      (let* ([pattern-name (string->symbol (string-append "pattern-" (symbol->string name)))]
             [full-expr (if debug-flag
                           `(define-conversion-pattern ,debug-flag ,pattern-name ,@pattern-body)
                           `(define-conversion-pattern ,pattern-name ,@pattern-body))])
        ;; Eval the pattern definition
        (eval full-expr (interaction-environment))

        ;; Get and show the result
        (let ([pattern-fn (eval pattern-name (interaction-environment))])
          (if (procedure? pattern-fn)
              (if debug-flag
                  (let ([result (pattern-fn)])
                    (display (format "Pattern ~a output:\n" name))
                    (pretty-print result))
                  (display "Normal mode - no output to show\n"))
              (display (format "ERROR: ~a is not a procedure\n" pattern-name)))))))

  ;;=======================================================================
  ;; Run one phase for one test case
  ;;=======================================================================

  (define (run-one-phase phase-name debug-flag expect-key test-name test-body)
    (let ([pattern (get-field ':pattern test-body)]
          [expectations (get-field expect-key test-body)])
      (if expectations
          (begin
            (display (format "~a... " phase-name))
            (let ([result (eval-pattern test-name debug-flag pattern expectations)])
              (display (if result "✓\n" "✗\n"))
              result))
          (begin
            (display (format "~a... skipped (no expectations)\n" phase-name))
            'skipped))))

  ;;=======================================================================
  ;; Run tests for a phase (OLD - kept for compatibility)
  ;;=======================================================================

  (define (run-phase-tests phase-name debug-flag expect-key test-bodies)
    (display (format "\n=== Phase: ~a ===\n" phase-name))
    (loop :for test-case :in test-bodies
          :rime-with name := (car test-case)
          :rime-with pattern := (get-field ':pattern test-case)
          :rime-with expectations := (get-field expect-key test-case)

          :rime-with result := (if expectations
                                   (begin
                                     (display (format "  Testing ~a... " name))
                                     (let ([r (eval-pattern name debug-flag pattern expectations)])
                                       (display (if r "✓\n" "✗\n"))
                                       r))
                                   (begin
                                     (display (format "  Skipping ~a (no expectations)\n" name))
                                     'skipped))

          :count :into passed :if (eq? result #t)
          :count :into failed :if (eq? result #f)

          :finally
            (begin
              (display (format "\nResults: ~a passed, ~a failed\n" passed failed))
              (list passed failed))))

) ;; end library
