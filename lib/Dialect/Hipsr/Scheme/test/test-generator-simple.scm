#!/usr/bin/env scheme-script
#!r6rs
(import (rnrs)
        (test test-pattern-generator))

(generate-test-patterns :debug-parse "-parse")

(display "Pattern generation test complete\n")
