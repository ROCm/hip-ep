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
        (test test-helpers))

(define (main)
  (let ([test-bodies (load-test-bodies)])
    (display "==================================================\n")
    (display "Pattern DSL Test Suite (Data-Driven)\n")
    (display "==================================================\n\n")

    ;; Run all 4 phases
    (run-phase-tests "Parse" ':debug-parse ':expect-parse test-bodies)
    (run-phase-tests "Validate" ':debug-validate ':expect-validate test-bodies)
    (run-phase-tests "Analyze" ':debug-analyze ':expect-analyze test-bodies)
    (run-phase-tests "Codegen" #f ':expect-codegen test-bodies)

    (display "\n==================================================\n")
    (display "All tests complete\n")
    (display "==================================================\n")))

(main)
