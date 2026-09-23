#!r6rs
;;===----------------------------------------------------------------------===;;
;; Data-Driven Test Runner
;;===----------------------------------------------------------------------===;;
;;
;; Usage:
;;   scheme --script test/run-tests.scm              # Run all tests
;;   scheme --script test/run-tests.scm NAME         # Run all phases for one test
;;   scheme --script test/run-tests.scm NAME PHASE   # Show output for NAME/PHASE
;;===----------------------------------------------------------------------===;;

(library-directories '("." "libraries" "../../../../third_party/rime"))

(import (except (chezscheme) =)
        (mlir pattern-macro)
        (rename (rime loop) (:with :rime-with))
        (test test-helpers))

;; Import pattern-macro into interaction-environment so eval can use it
(eval '(import (mlir pattern-macro)) (interaction-environment))

;;===----------------------------------------------------------------------===;;
;; Phase Configuration
;;===----------------------------------------------------------------------===;;

(define *phases*
  '((parse    "  Parse"    :debug-parse    :expect-parse)
    (validate "  Validate" :debug-validate :expect-validate)
    (analyze  "  Analyze"  :debug-analyze  :expect-analyze)
    (codegen  "  Codegen"  :debug-codegen  :expect-codegen)))

(define (phase-name->debug-flag phase-name)
  (let ([entry (assq phase-name *phases*)])
    (if entry (caddr entry) #f)))

(define (get-phase-info phase-name)
  (assq phase-name *phases*))

;;===----------------------------------------------------------------------===;;
;; Result Counting
;;===----------------------------------------------------------------------===;;

(define (count-results results)
  (loop :for result :in results
        :count :into passed :if (eq? result #t)
        :count :into failed :if (eq? result #f)
        :finally (cons passed failed)))

;;===----------------------------------------------------------------------===;;
;; Single Test Execution
;;===----------------------------------------------------------------------===;;

(define (run-test-phases test-name test-body)
  "Run all phases for a single test, return list of results"
  (loop :for phase-info :in *phases*
        :rime-with label := (cadr phase-info)
        :rime-with debug-flag := (caddr phase-info)
        :rime-with expect-key := (cadddr phase-info)
        :collect (run-one-phase label debug-flag expect-key test-name test-body)))

(define (find-test test-name test-cases)
  (let ([pair (assq test-name test-cases)])
    (if pair (cdr pair) #f)))

;;===----------------------------------------------------------------------===;;
;; Show Output (for troubleshooting individual phase)
;;===----------------------------------------------------------------------===;;

(define (show-output test-name phase-name debug-flag)
  (let* ([test-cases (load-test-cases)]
         [test-body (find-test test-name test-cases)])
    (if test-body
        (let ([pattern (get-field ':pattern test-body)])
          (display (format "Test: ~a, Phase: ~a\n\n" test-name phase-name))
          (show-pattern-output test-name debug-flag pattern))
        (display (format "Test ~a not found\n" test-name)))))

;;===----------------------------------------------------------------------===;;
;; Run Single Test
;;===----------------------------------------------------------------------===;;

(define (run-single-test test-name test-cases)
  (let* ([test-pair (find (lambda (p) (eq? (car p) test-name)) test-cases)])
    (if test-pair
        (let* ([test-body (cdr test-pair)]
               [results (run-test-phases test-name test-body)]
               [counts (count-results results)])
          (display "==================================================\n")
          (display (format "Testing: ~a\n" test-name))
          (display "==================================================\n\n")

          ;; Results already displayed by run-one-phase

          (display "\n==================================================\n")
          (display (format "Result: ~a passed, ~a failed\n" (car counts) (cdr counts)))
          (display "==================================================\n"))
        (display (format "Test ~a not found\n" test-name)))))

;;===----------------------------------------------------------------------===;;
;; Run All Tests
;;===----------------------------------------------------------------------===;;

(define (run-all-tests test-cases)
  (display "==================================================\n")
  (display "Pattern DSL Test Suite (Data-Driven)\n")
  (display "==================================================\n\n")

  (let* ([all-counts (loop :for test-pair :in test-cases
                           :rime-with test-name := (car test-pair)
                           :rime-with test-body := (cdr test-pair)
                           :rime-with results := (run-test-phases test-name test-body)
                           :rime-with counts := (count-results results)
                           :do (display (format "Testing: ~a\n" test-name))
                           :do (display (format "  Result: ~a passed, ~a failed\n\n"
                                              (car counts) (cdr counts)))
                           :collect counts)]
         [total-passed (apply + (map car all-counts))]
         [total-failed (apply + (map cdr all-counts))])

    ;; Display totals
    (display "==================================================\n")
    (display (format "Total: ~a passed, ~a failed\n" total-passed total-failed))
    (display "==================================================\n")))

;;===----------------------------------------------------------------------===;;
;; Main Entry Point
;;===----------------------------------------------------------------------===;;

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
       (let* ([test-name (string->symbol (car args))]
              [phase (string->symbol (cadr args))]
              [debug-flag (phase-name->debug-flag phase)])
         (if debug-flag
             (show-output test-name phase debug-flag)
             (begin
               (display (format "Unknown phase: ~a\n" phase))
               (display "Valid phases: parse, validate, analyze, codegen\n"))))]

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
