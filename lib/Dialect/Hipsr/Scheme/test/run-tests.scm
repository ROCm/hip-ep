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
        (mlir pattern-macro)
        (rename (rime loop) (:with :rime-with))  ; Rename :with to avoid conflict
        (test test-helpers))

;; Import pattern-macro into interaction-environment so eval can use it
(eval '(import (mlir pattern-macro)) (interaction-environment))

(define (main)
  (let ([test-bodies (load-test-bodies)])
    (display "==================================================\n")
    (display "Pattern DSL Test Suite (Data-Driven)\n")
    (display "==================================================\n\n")

    ;; For each test case, run all 4 phases and count results
    (loop :for test-case :in test-bodies
          :do (display (format "Testing: ~a\n" (car test-case)))

          ;; Run all phases and bind results using :rime-with
          :rime-with parse-result := (run-one-phase "  Parse" ':debug-parse ':expect-parse test-case)
          :rime-with validate-result := (run-one-phase "  Validate" ':debug-validate ':expect-validate test-case)
          :rime-with analyze-result := (run-one-phase "  Analyze" ':debug-analyze ':expect-analyze test-case)
          :rime-with codegen-result := (run-one-phase "  Codegen" #f ':expect-codegen test-case)

          ;; Count passed tests into total-passed
          :count :into total-passed :if (eq? parse-result #t)
          :count :into total-passed :if (eq? validate-result #t)
          :count :into total-passed :if (eq? analyze-result #t)
          :count :into total-passed :if (eq? codegen-result #t)

          ;; Count failed tests into total-failed
          :count :into total-failed :if (eq? parse-result #f)
          :count :into total-failed :if (eq? validate-result #f)
          :count :into total-failed :if (eq? analyze-result #f)
          :count :into total-failed :if (eq? codegen-result #f)

          ;; Display per-test result
          :do (let ([passed (+ (if (eq? parse-result #t) 1 0)
                               (if (eq? validate-result #t) 1 0)
                               (if (eq? analyze-result #t) 1 0)
                               (if (eq? codegen-result #t) 1 0))]
                    [failed (+ (if (eq? parse-result #f) 1 0)
                               (if (eq? validate-result #f) 1 0)
                               (if (eq? analyze-result #f) 1 0)
                               (if (eq? codegen-result #f) 1 0))])
                (display (format "  Result: ~a passed, ~a failed\n\n" passed failed)))

          :finally
            (begin
              (display "==================================================\n")
              (display (format "Total: ~a passed, ~a failed\n" total-passed total-failed))
              (display "==================================================\n")))))

(main)
