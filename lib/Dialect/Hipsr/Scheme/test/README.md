# Scheme Unit Tests

Simple test framework for Scheme libraries.

## Running Tests

```bash
cd /workspace/hip-ep/hip-ep-1/lib/Dialect/Hipsr/Scheme
scheme --libdirs .:libraries --program test/run-all-tests.scm
```

## Writing Tests

1. Create test file: `test/my-test.sls`

```scheme
#!r6rs
(library (test my-test)
  (export run-tests)
  (import (rnrs (6))
          (test test-framework)
          (my library))

  (define (run-tests)
    (test-begin "my-test")
    
    (test-equal "description" actual expected)
    (test-assert "description" condition)
    (test-error "description" (lambda () (error "boom")))
    
    (test-end)))
```

2. Add to `test/run-all-tests.scm`:

```scheme
(import (test my-test))
(run-tests)
```

## Test Framework API

- `(test-begin "suite-name")` - Start test suite
- `(test-end)` - End suite, exit(1) if failures
- `(test-equal desc actual expected)` - Assert equality
- `(test-assert desc condition)` - Assert truth
- `(test-error desc thunk)` - Assert error raised

## Current Tests

- `basic-test.sls` - Framework self-test (no FFI)
- `pattern-dsl-test.sls` - Pattern DSL (requires FFI, TODO)
