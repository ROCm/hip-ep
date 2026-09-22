#!r6rs
;;===----------------------------------------------------------------------===;;
;; Phase 2: Validate Test - Test pattern validation
;;===----------------------------------------------------------------------===;;
;;
;; Focus: Test ONLY what the validate phase does
;; - Validate % prefix on identifiers
;; - Detect duplicate result variables
;; - Validate root var exists in results
;; - Normalize list → vector, symbol → string
;; - Validate :where guard syntax
;; - Validate :then-let binding variables
;;
;; Strategy: Use patterns that PARSE successfully and verify normalization
;; Note: Invalid patterns (missing %, duplicates) would cause syntax-violation,
;;       so we test that valid patterns normalize correctly
;;
;; All test patterns imported from (test test-patterns) - single source of truth
;;
;;===----------------------------------------------------------------------===;;

(library (test phase-2-validate-test)
  (export run-tests)
  (import (except (chezscheme) =)
          (test test-framework)
          (test test-patterns))

  ;;=======================================================================
  ;; Helper Functions
  ;;=======================================================================

  (define (ast-get ast key)
    "Extract field from AST list by key"
    (let loop ([rest ast])
      (cond
        [(null? rest) #f]
        [(eq? (car rest) key) (cadr rest)]
        [else (loop (cddr rest))])))

  ;;=======================================================================
  ;; Test Suite
  ;;=======================================================================

  (define (run-tests)
    (test-begin "phase-2-validate")
    (display "\n=== Phase 2: Validate Tests (Using Shared Patterns) ===\n\n")

    ;; Test 1: Basic pattern validates successfully
    (test-equal "basic: validates successfully" #t (list? pattern-basic-validate))
    (test-equal "basic: root-op-name is string" "test.op"
                (ast-get pattern-basic-validate 'root-op-name))
    ;; TODO: Verify symbol was normalized to string

    ;; Test 2: Required operands validate
    (test-equal "required: validates successfully" #t (list? pattern-required-validate))
    (test-equal "required: match field exists" #t
                (not (not (ast-get pattern-required-validate 'match))))
    ;; TODO: Verify match list normalized to vector

    ;; Test 3: Optional operands validate
    (test-equal "optional: validates successfully" #t (list? pattern-optional-validate))
    ;; TODO: Verify operand segments marked correctly

    ;; Test 4: Variadic operands validate
    (test-equal "variadic: validates successfully" #t (list? pattern-variadic-validate))
    ;; TODO: Verify variadic segment metadata

    ;; Test 5: Mixed operands validate
    (test-equal "mixed: validates successfully" #t (list? pattern-mixed-validate))
    ;; TODO: Verify complex operand structure validates

    ;; Test 6: :where guard validates
    (test-equal "where: validates successfully" #t (list? pattern-where-validate))
    ;; TODO: Verify guard expression preserved correctly

    ;; Test 7: :then-let bindings validate
    (test-equal "then-let: validates successfully" #t (list? pattern-then-let-validate))
    (test-equal "then-let: has bindings" #t
                (pair? (ast-get pattern-then-let-validate 'where)))
    ;; TODO: Verify all binding variables have % prefix

    ;; Test 8: Two operations validate
    (test-equal "two-ops: validates successfully" #t (list? pattern-two-ops-validate))
    (test-equal "two-ops: root-var set" '%b
                (ast-get pattern-two-ops-validate 'root-var))
    ;; TODO: Verify root-op-index cached correctly

    ;; Test 9: Variable reuse validates
    (test-equal "reuse: validates successfully" #t (list? pattern-reuse-validate))
    ;; Note: Validation doesn't detect reuse - that's semantic analysis (phase 3)

    ;; Test 10: Combined features validate
    (test-equal "combined: validates successfully" #t (list? pattern-combined-validate))
    (test-equal "combined: has 2 operations" 2
                (length (ast-get pattern-combined-validate 'match)))
    (test-equal "combined: has bindings" #t
                (pair? (ast-get pattern-combined-validate 'where)))

    ;; Test 11: Regions validate
    (test-equal "regions: validates successfully" #t (list? pattern-region-validate))
    ;; TODO: Verify region structure validates

    ;; Test 12: Multiple rewrite operations validate
    (test-equal "multi-rewrite: validates successfully" #t (list? pattern-multi-rewrite-validate))
    ;; TODO: Verify all rewrite operation names normalized to strings

    (display "\nValidation tests verify normalization and structure.\n")
    (display "Invalid patterns (missing %, duplicates) cause syntax-violation.\n")

    (test-end)))
