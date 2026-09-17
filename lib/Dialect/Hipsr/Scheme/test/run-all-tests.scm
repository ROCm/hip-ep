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
(run-tests)

(display "✓ All tests completed\n\n")
