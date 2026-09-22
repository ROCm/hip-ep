#!/usr/bin/env scheme-script
#!r6rs

(import (rnrs (6))
        (prefix (test test-framework) framework:)
        (prefix (test phase-1-parse-test) phase1:)
        (prefix (test phase-2-validate-test) phase2:)
        (prefix (test phase-3-analyze-test) phase3:)
        (prefix (test phase-4-codegen-test) phase4:)
        (prefix (test integration-test) integration:))

(display "\n")
(display "==============================================================================\n")
(display "  Pattern DSL Test Suite - All Phases (Refactored with Shared Patterns)\n")
(display "==============================================================================\n")

;; Phase-specific tests using shared patterns from test-patterns.sls
(phase1:run-tests)
(phase2:run-tests)
(phase3:run-tests)
(phase4:run-tests)

;; Integration tests (all phases together)
(integration:run-tests)

(display "\n")
(display "==============================================================================\n")
(display "  All tests complete\n")
(display "==============================================================================\n")
(display "\n")
(display "Test Architecture:\n")
(display "  - test-patterns.sls:      Shared test pattern definitions (single source)\n")
(display "  - phase-1-parse-test.sls: Tests parsing phase only\n")
(display "  - phase-2-validate-test.sls: Tests validation phase only\n")
(display "  - phase-3-analyze-test.sls: Tests analyze phase only\n")
(display "  - phase-4-codegen-test.sls: Tests codegen phase only\n")
(display "  - integration-test.sls:   Tests all phases together\n")
(display "\n")
