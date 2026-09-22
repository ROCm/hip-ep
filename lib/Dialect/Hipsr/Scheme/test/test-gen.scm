#!/usr/bin/env scheme-script
#!r6rs
(import (rnrs)
        (test test-pattern-generator))

(define-test-patterns :debug-parse "-parse")

(display "basic-parse defined: ")
(display (procedure? basic-parse))
(newline)
