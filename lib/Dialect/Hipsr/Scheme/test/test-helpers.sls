#!r6rs
;;===----------------------------------------------------------------------===;;
;; Test Helpers - Load and eval patterns from test-pattern-bodies.scm
;;===----------------------------------------------------------------------===;;

(library (test test-helpers)
  (export load-test-bodies
          eval-pattern
          get-field
          run-one-phase
          run-phase-tests)
  (import (chezscheme)
          (except (mlir pattern-macro) =)  ; Exclude = to avoid conflict
          (rename (rime loop) (:with :rime-with)))

  ;;=======================================================================
  ;; Load test data
  ;;=======================================================================

  (define (load-test-bodies)
    (call-with-input-file "test/test-pattern-bodies.scm" read))

  ;;=======================================================================
  ;; Extract field from test case
  ;;=======================================================================

  (define (get-field key test-case)
    ;; Simple plist traversal - advance by 2
    (let loop ([rest (cdr test-case)])
      (cond
        [(null? rest) #f]
        [(eq? (car rest) key) (cadr rest)]
        [else (loop (cddr rest))])))

  ;;=======================================================================
  ;; Check expectations against actual plist result
  ;;=======================================================================

  (define (plist-ref plist key)
    ;; Get value from plist by key
    (let loop ([rest plist])
      (cond
        [(null? rest) #f]
        [(null? (cdr rest)) #f]
        [(eq? (car rest) key) (cadr rest)]
        [else (loop (cddr rest))])))

  (define (check-expectation result expectation)
    ;; Check one expectation (key . expected-value) against result plist
    (let ([key (car expectation)]
          [expected (cdr expectation)])
      (cond
        ;; has-function-name - check key exists
        [(eq? key 'has-function-name)
         (if expected
             (and (plist-ref result 'function-name) #t)
             (not (plist-ref result 'function-name)))]

        ;; match-count - count elements in match list
        [(eq? key 'match-count)
         (let ([match-list (plist-ref result 'match)])
           (and match-list (fx= (length match-list) expected)))]

        ;; rewrite-count - count elements in rewrite list
        [(eq? key 'rewrite-count)
         (let ([rewrite-list (plist-ref result 'rewrite)])
           (and rewrite-list (fx= (length rewrite-list) expected)))]

        ;; Default: check value equality
        [else
         (let ([actual (plist-ref result key)])
           (equal? actual expected))])))

  (define (check-all-expectations result expectations)
    ;; Returns (passed? . failing-expectation-or-#f)
    (loop :for exp :in expectations
          :rime-with passed := (check-expectation result exp)
          :unless passed
          :break (cons #f exp)
          :finally (cons #t #f)))

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
  ;; Run one phase for one test case
  ;;=======================================================================

  (define (run-one-phase phase-name debug-flag expect-key test-case)
    (let ([name (car test-case)]
          [pattern (get-field ':pattern test-case)]
          [expectations (get-field expect-key test-case)])
      (if expectations
          (begin
            (display (format "~a... " phase-name))
            (let ([result (eval-pattern name debug-flag pattern expectations)])
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
