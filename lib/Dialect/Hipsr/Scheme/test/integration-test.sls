#!r6rs
;;===----------------------------------------------------------------------===;;
;; Integration Test - End-to-End: Test all phases together
;;===----------------------------------------------------------------------===;;
;;
;; Focus: Test complete pattern DSL pipeline
;; - All 4 phases working together (parse → validate → analyze → codegen)
;; - Generated pattern functions are callable
;; - Pattern matching works correctly (if mock FFI available)
;; - Rewrite operations execute properly
;; - Edge cases and error handling
;;
;; No debug flags - test normal lambda generation mode
;;
;; All test patterns imported from (test test-patterns) - single source of truth
;;
;;===----------------------------------------------------------------------===;;

(library (test integration-test)
  (export run-tests)
  (import (except (chezscheme) =)
          (test test-framework)
          (test test-patterns))

  ;;=======================================================================
  ;; Helper Functions
  ;;=======================================================================

  (define (is-procedure? x)
    "Check if value is a procedure"
    (procedure? x))

  ;; Mock FFI functions for testing (if real FFI not available)
  (define (mlir-operation-get-attribute op name)
    "Mock: return truthy value for testing"
    #t)

  (define (mlir-operation-has-one-use op)
    "Mock: return true for testing"
    #t)

  (define (is-1x1-conv? op)
    "Mock: return true for testing"
    #t)

  (define (mlir-get-context op)
    "Mock: return dummy context"
    'mock-context)

  (define (get-context)
    "Mock: return dummy context"
    'mock-context)

  (define (compute-value)
    "Mock: return dummy value"
    'mock-value)

  (define (get-device-type val)
    "Mock: return dummy device type"
    'mock-device)

  ;;=======================================================================
  ;; Test Suite
  ;;=======================================================================

  (define (run-tests)
    (test-begin "integration")
    (display "\n=== Integration Tests: All Phases Together (Using Shared Patterns) ===\n\n")

    ;; Test 1: Basic pattern generates lambda
    (test-equal "basic: generates lambda" #t (is-procedure? pattern-basic-lambda))
    ;; TODO: Call pattern with mock operation and verify behavior
    ;; (test-equal "basic: callable" #t
    ;;   (procedure? (lambda () (pattern-basic-lambda mock-op mock-ref mock-rewriter mock-converter))))

    ;; Test 2: Required operands
    (test-equal "required: generates lambda" #t (is-procedure? pattern-required-lambda))

    ;; Test 3: Optional operands
    (test-equal "optional: generates lambda" #t (is-procedure? pattern-optional-lambda))
    ;; TODO: Test with operation that has optional operands
    ;; TODO: Test with operation missing optional operands

    ;; Test 4: Variadic operands
    (test-equal "variadic: generates lambda" #t (is-procedure? pattern-variadic-lambda))
    ;; TODO: Test with varying numbers of operands
    ;; TODO: Verify %rest captures all remaining operands

    ;; Test 5: Mixed operands
    (test-equal "mixed: generates lambda" #t (is-procedure? pattern-mixed-lambda))

    ;; Test 6: :where guard
    (test-equal "where: generates lambda" #t (is-procedure? pattern-where-lambda))
    ;; TODO: Test that guard function is called during matching
    ;; TODO: Test that pattern fails when guard returns #f

    ;; Test 7: :then-let
    (test-equal "then-let: generates lambda" #t (is-procedure? pattern-then-let-lambda))
    ;; TODO: Verify bindings are computed after match succeeds
    ;; TODO: Verify bindings are NOT computed if match fails

    ;; Test 8: Two operations - DAG matching
    (test-equal "two-ops: generates lambda" #t (is-procedure? pattern-two-ops-lambda))

    ;; Test 9: Variable reuse
    (test-equal "reuse: generates lambda" #t (is-procedure? pattern-reuse-lambda))

    ;; Test 10: Combined features
    (test-equal "combined: generates lambda" #t (is-procedure? pattern-combined-lambda))
    ;; TODO: Test complex pattern with guards, bindings, multiple ops

    ;; Test 11: Regions
    (test-equal "regions: generates lambda" #t (is-procedure? pattern-region-lambda))
    ;; TODO: Verify region structure in rewrite
    ;; TODO: Test block arguments and operations

    ;; Test 12: Multiple rewrite operations
    (test-equal "multi-rewrite: generates lambda" #t (is-procedure? pattern-multi-rewrite-lambda))
    ;; TODO: Verify operations created in sequence
    ;; TODO: Verify each operation uses previous results

    (display "\nIntegration tests verify lambda generation for complex patterns.\n")
    (display "All patterns imported from test-patterns.sls - single source of truth.\n")
    (display "TODO: Add actual pattern matching tests with mock MLIR operations.\n")
    (display "      Requires mock FFI or test harness for operation creation.\n")

    (test-end)))
