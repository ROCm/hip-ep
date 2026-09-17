#!/usr/bin/env scheme-script
(import (rnrs (6))
        (prefix (test basic-test) basic:)
        (prefix (test pattern-macro-test) macro:))

(display "\n╔════════════════════════════════════════╗\n")
(display "║   Scheme Unit Test Suite               ║\n")
(display "╚════════════════════════════════════════╝\n")

(basic:run-tests)
(macro:run-tests)

(display "✓ All test suites completed\n\n")
