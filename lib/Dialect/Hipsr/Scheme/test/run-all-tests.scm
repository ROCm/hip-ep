#!/usr/bin/env scheme-script
#!r6rs

(import (rnrs (6))
        (prefix (test basic-test) basic:)
        (prefix (test pattern-macro-incremental-test) incremental:))

(basic:run-tests)
(incremental:run-tests)
