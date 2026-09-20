#!/usr/bin/env scheme-script
#!r6rs

(import (rnrs (6))
        (prefix (test framework-test) framework:)
        (prefix (test parse-validate-test) parse-validate:))

(framework:run-tests)
(parse-validate:run-tests)
