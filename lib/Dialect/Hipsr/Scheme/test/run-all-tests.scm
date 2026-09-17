#!/usr/bin/env scheme-script
;;===----------------------------------------------------------------------===;;
;; Test Runner - Run all Scheme unit tests
;;===----------------------------------------------------------------------===;;

(import (rnrs (6))
        (test basic-test))

(display "\n╔════════════════════════════════════════╗\n")
(display "║   Scheme Unit Test Suite               ║\n")
(display "╚════════════════════════════════════════╝\n")

;; Run test suites
;; Note: pattern-dsl-test requires FFI bindings (run via hip-mlir-opt)
(run-tests)

(display "✓ All test suites completed\n\n")
