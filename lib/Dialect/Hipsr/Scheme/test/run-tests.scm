#!r6rs
;;===----------------------------------------------------------------------===;;
;; Data-Driven Test Runner
;;===----------------------------------------------------------------------===;;
;;
;; Usage:
;;   scheme --script test/run-tests.scm              # Run all tests
;;   scheme --script test/run-tests.scm NAME PHASE   # Show output for NAME/PHASE
;;===----------------------------------------------------------------------===;;

;; Set library search paths
(library-directories '("." "libraries" "../../../../third_party/rime"))

(import (except (chezscheme) =)
        (mlir pattern-macro)
        (rename (rime loop) (:with :rime-with))
        (test test-helpers))

;; Import pattern-macro into interaction-environment so eval can use it
(eval '(import (mlir pattern-macro)) (interaction-environment))

(define (find-test test-name test-cases)
  (let ([pair (assq test-name test-cases)])
    (if pair (cdr pair) #f)))

(define (show-output test-name phase-name debug-flag)
  (let* ([test-cases (load-test-cases)]
         [test-body (find-test test-name test-cases)])
    (if test-body
        (let ([pattern (get-field ':pattern test-body)])
          (display (format "Test: ~a, Phase: ~a\n\n" test-name phase-name))
          (show-pattern-output test-name debug-flag pattern))
        (display (format "Test ~a not found\n" test-name)))))

(define (run-all-tests test-cases)
  (display "==================================================\n")
  (display "Pattern DSL Test Suite (Data-Driven)\n")
  (display "==================================================\n\n")

  ;; For each test case, run all 4 phases and count results
  (let ([total-passed 0]
        [total-failed 0])
    (for-each
      (lambda (test-pair)
        (let* ([test-name (car test-pair)]
               [test-body (cdr test-pair)])
          (display (format "Testing: ~a\n" test-name))

          ;; Run all phases
          (let* ([parse-result (run-one-phase "  Parse" ':debug-parse ':expect-parse test-name test-body)]
                 [validate-result (run-one-phase "  Validate" ':debug-validate ':expect-validate test-name test-body)]
                 [analyze-result (run-one-phase "  Analyze" ':debug-analyze ':expect-analyze test-name test-body)]
                 [codegen-result (run-one-phase "  Codegen" ':debug-codegen ':expect-codegen test-name test-body)]
                 [passed (+ (if (eq? parse-result #t) 1 0)
                           (if (eq? validate-result #t) 1 0)
                           (if (eq? analyze-result #t) 1 0)
                           (if (eq? codegen-result #t) 1 0))]
                 [failed (+ (if (eq? parse-result #f) 1 0)
                           (if (eq? validate-result #f) 1 0)
                           (if (eq? analyze-result #f) 1 0)
                           (if (eq? codegen-result #f) 1 0))])

            ;; Update totals
            (set! total-passed (+ total-passed passed))
            (set! total-failed (+ total-failed failed))

            ;; Display per-test result
            (display (format "  Result: ~a passed, ~a failed\n\n" passed failed)))))
      test-cases)

    ;; Display totals
    (display "==================================================\n")
    (display (format "Total: ~a passed, ~a failed\n" total-passed total-failed))
    (display "==================================================\n")))

(define (run-single-test test-name test-cases)
  (let* ([test-pair (find (lambda (p) (eq? (car p) test-name)) test-cases)])
    (if test-pair
        (let ([test-body (cdr test-pair)])
          (display "==================================================\n")
          (display (format "Testing: ~a\n" test-name))
          (display "==================================================\n\n")

          (let* ([parse-result (run-one-phase "  Parse" ':debug-parse ':expect-parse test-name test-body)]
                 [validate-result (run-one-phase "  Validate" ':debug-validate ':expect-validate test-name test-body)]
                 [analyze-result (run-one-phase "  Analyze" ':debug-analyze ':expect-analyze test-name test-body)]
                 [codegen-result (run-one-phase "  Codegen" ':debug-codegen ':expect-codegen test-name test-body)]
                 [passed (+ (if (eq? parse-result #t) 1 0)
                           (if (eq? validate-result #t) 1 0)
                           (if (eq? analyze-result #t) 1 0)
                           (if (eq? codegen-result #t) 1 0))]
                 [failed (+ (if (eq? parse-result #f) 1 0)
                           (if (eq? validate-result #f) 1 0)
                           (if (eq? analyze-result #f) 1 0)
                           (if (eq? codegen-result #f) 1 0))])

            (display "\n==================================================\n")
            (display (format "Result: ~a passed, ~a failed\n" passed failed))
            (display "==================================================\n")))
        (display (format "Test ~a not found\n" test-name)))))

(define (main args)
  (let ([test-cases (load-test-cases)])
    (cond
      ;; No arguments: run all tests
      [(null? args)
       (run-all-tests test-cases)]

      ;; One argument: run all phases for one test
      [(fx= (length args) 1)
       (let ([test-name (string->symbol (car args))])
         (run-single-test test-name test-cases))]

      ;; Two arguments: show test-name phase output
      [(fx= (length args) 2)
       (let ([test-name (string->symbol (car args))]
             [phase (string->symbol (cadr args))])
         (show-output test-name phase
                      (case phase
                        [(parse) ':debug-parse]
                        [(validate) ':debug-validate]
                        [(analyze) ':debug-analyze]
                        [(codegen) ':debug-codegen]
                        [else (error 'main "Unknown phase" phase)])))]

      ;; Invalid usage
      [else
       (display "Usage:\n")
       (display "  scheme --script test/run-tests.scm              # Run all tests\n")
       (display "  scheme --script test/run-tests.scm NAME         # Run all phases for one test\n")
       (display "  scheme --script test/run-tests.scm NAME PHASE   # Show output for one phase\n")
       (display "    Phases: parse, validate, analyze, codegen\n")
       (display "    Example: scheme --script test/run-tests.scm basic\n")
       (display "    Example: scheme --script test/run-tests.scm basic parse\n")])))

;; Get command line arguments (skip program name)
(main (cdr (command-line)))
