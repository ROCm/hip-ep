#!r6rs
;;===----------------------------------------------------------------------===;;
;; Test Helpers - Load and eval patterns from test-pattern-bodies.scm
;;===----------------------------------------------------------------------===;;

(library (test test-helpers)
  (export load-test-bodies
          eval-pattern
          get-field
          run-phase-tests)
  (import (except (chezscheme) =)
          (except (mlir pattern-macro) :with)  ; Exclude :with to avoid conflict
          (rime loop))  ; Use rime loop's :with

  ;;=======================================================================
  ;; Load test data
  ;;=======================================================================

  (define (load-test-bodies)
    (call-with-input-file "test/test-pattern-bodies.scm" read))

  ;;=======================================================================
  ;; Extract field from test case
  ;;=======================================================================

  (define (get-field key test-case)
    ;; Manual recursion is clearest for plist traversal (advance by 2)
    (let loop ([rest (cdr test-case)])  ;; Skip name
      (cond
        [(null? rest) #f]
        [(eq? (car rest) key) (cadr rest)]
        [else (loop (cddr rest))])))

  ;;=======================================================================
  ;; Eval pattern with debug flag
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

        ;; Get the result (AST for debug modes, function otherwise)
        (let ([result (eval pattern-name (interaction-environment))])
          (cond
            ;; Debug mode: result is an AST list - check expectations
            [(and debug-flag (list? result))
             ;; TODO: Check expectations against AST
             ;; For now, just verify it's a list
             #t]

            ;; Normal mode: result should be a function
            [(procedure? result)
             #t]

            [else
             (display (format "ERROR: Pattern ~a returned unexpected type\n" name))
             #f])))))

  ;;=======================================================================
  ;; Run tests for a phase
  ;;=======================================================================

  (define (run-phase-tests phase-name debug-flag expect-key test-bodies)
    ;; Simplified with rime loop - collect results then count
    (display (format "\n=== Phase: ~a ===\n" phase-name))
    (let ([results
           (loop :for test-case :in test-bodies
                 :collect
                 (let ([name (car test-case)]
                       [pattern (get-field ':pattern test-case)]
                       [expectations (get-field expect-key test-case)])
                   (if expectations
                       (begin
                         (display (format "  Testing ~a... " name))
                         (let ([result (eval-pattern name debug-flag pattern expectations)])
                           (display (if result "✓\n" "✗\n"))
                           result))
                       (begin
                         (display (format "  Skipping ~a (no expectations)\n" name))
                         'skipped))))])
      (let ([passed (length (filter (lambda (x) (eq? x #t)) results))]
            [failed (length (filter (lambda (x) (eq? x #f)) results))])
        (display (format "\nResults: ~a passed, ~a failed\n" passed failed))
        (list passed failed))))

) ;; end library (test test-helpers)
