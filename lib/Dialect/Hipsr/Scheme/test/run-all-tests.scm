#!/usr/bin/env scheme-script
(import (rnrs (6))
        (test basic-test))

(display "\n╔════════════════════════════════════════╗\n")
(display "║   Scheme Unit Test Suite               ║\n")
(display "╚════════════════════════════════════════╝\n")

(run-tests)

(display "✓ All test suites completed\n\n")
