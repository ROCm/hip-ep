#!r6rs
;;===----------------------------------------------------------------------===;;
;; Data-Driven Test Runner
;;===----------------------------------------------------------------------===;;
;;
;; Reads test-pattern-bodies.scm and runs all phases for each pattern
;; Each pattern declares which phases to test via :expect-parse, :expect-validate, etc.
;;===----------------------------------------------------------------------===;;

;; Set library search paths (relative to CWD where script is run from)
;; Searches: current dir (for test/), libraries/ (for mlir/), and rime (for rime/)
(library-directories '("." "libraries" "../../../../third_party/rime"))

(import (except (chezscheme) =)
        (except (mlir pattern-macro) :with)  ; Import pattern macro
        (test test-helpers))

;; Import pattern-macro into interaction-environment so eval can use it
(eval '(import (mlir pattern-macro)) (interaction-environment))

(define (main)
  (let ([test-bodies (load-test-bodies)])
    (display "==================================================\n")
    (display "Pattern DSL Test Suite (Data-Driven)\n")
    (display "==================================================\n\n")

    ;; For each test case, run all 4 phases
    (let ([total-passed 0]
          [total-failed 0])
      (for-each
        (lambda (test-case)
          (let ([name (car test-case)])
            (display (format "Testing: ~a\n" name))

            ;; Run all phases for this test case
            (let ([parse-result (run-one-phase "  Parse" ':debug-parse ':expect-parse test-case)]
                  [validate-result (run-one-phase "  Validate" ':debug-validate ':expect-validate test-case)]
                  [analyze-result (run-one-phase "  Analyze" ':debug-analyze ':expect-analyze test-case)]
                  [codegen-result (run-one-phase "  Codegen" #f ':expect-codegen test-case)])

              ;; Count results
              (let ([passed (+ (if (eq? parse-result #t) 1 0)
                               (if (eq? validate-result #t) 1 0)
                               (if (eq? analyze-result #t) 1 0)
                               (if (eq? codegen-result #t) 1 0))]
                    [failed (+ (if (eq? parse-result #f) 1 0)
                               (if (eq? validate-result #f) 1 0)
                               (if (eq? analyze-result #f) 1 0)
                               (if (eq? codegen-result #f) 1 0))])
                (set! total-passed (+ total-passed passed))
                (set! total-failed (+ total-failed failed))
                (display (format "  Result: ~a passed, ~a failed\n\n" passed failed))))))
        test-bodies)

      (display "==================================================\n")
      (display (format "Total: ~a passed, ~a failed\n" total-passed total-failed))
      (display "==================================================\n"))))

(main)
