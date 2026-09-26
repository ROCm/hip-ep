# Pattern DSL Tests - Data-Driven Approach

Simple, scalable test infrastructure using eval-based pattern generation.

## Quick Start

```bash
cd /workspace/hip-ep/hip-ep-1/lib/Dialect/Hipsr/Scheme
scheme --script test/run-tests.scm
```

## Architecture

```
test/
├── run-tests.scm              # Test runner - runs all 4 phases
├── test-helpers.sls           # Infrastructure - eval patterns from data
├── test-pattern-bodies.scm    # Test data - single source of truth
└── README.md                  # This file
```

## How It Works

1. **Load test data**: Read S-expressions from `test-pattern-bodies.scm`
2. **Generate pattern**: Build `define-conversion-pattern` expression with debug flag
3. **Eval pattern**: Use `eval` + `interaction-environment` to define it
4. **Check result**: Verify pattern compiled successfully

## Test Data Format

Each test case in `test-pattern-bodies.scm`:

```scheme
(pattern-name
  :pattern (
    :match %out = "test.op" (%in) : (f32) -> f32
    :rewrite %out :with
      (%new = "new.op" (%in) : (f32) -> f32))
  :expect-parse ((match-count . 1) (rewrite-count . 1))
  :expect-validate ((validation-passes . #t))
  :expect-analyze ((where-actions . 0))
  :expect-codegen ((compiles . #t)))
```

**Key features:**
- `:pattern` contains S-expression (not string!)
- `:expect-*` fields are optional per phase
- Missing expectation = skip that phase for this test

## Adding a New Test

Just add one S-expression to `test-pattern-bodies.scm`:

```scheme
(my-new-test
  :pattern (
    :match %a = "my.op" (%x) : (i32) -> i32
    :rewrite %a :with
      (%b = "new.op" (%x) : (i32) -> i32))
  :expect-parse ((has-function-name . #t))
  :expect-codegen ((compiles . #t)))
```

No code, no macros - just data!

## Implementation

**test-helpers.sls** provides:
- `load-test-bodies` - read data file
- `eval-pattern` - generate and eval pattern with debug flag
- `get-field` - extract `:expect-*` from test case
- `run-phase-tests` - run all tests for one phase

**How eval works:**
```scheme
(define (eval-pattern name debug-flag pattern-body)
  (let ([full-expr `(define-conversion-pattern 
                      ,debug-flag 
                      ,pattern-name 
                      ,@pattern-body)])
    (eval full-expr (interaction-environment))))
```

This works because:
1. `test-helpers.sls` imports `(mlir pattern-macro)` → loads the library
2. `interaction-environment` includes all imported libraries
3. `eval` expands macros in that environment

## Why This Approach?

**Old approach (deleted):**
- 4 phase-specific test files
- Complex macro system to generate patterns from data
- Lots of duplication and indirection

**New approach:**
- Single data file with all tests
- Simple eval-based infrastructure
- Easy to add/modify tests

**Benefits:**
- No temp files, no string conversion
- No macro complexity
- Clear separation: data vs infrastructure
- Scalable: 100 tests = 100 S-expressions in data file
- Each test declares which phases matter

## Example Output

```
==================================================
Pattern DSL Test Suite (Data-Driven)
==================================================

=== Phase: Parse ===
  Testing basic... ✓
  Testing optional... ✓
  Testing where-guard... ✓
  Skipping reuse (no expectations)

Results: 3 passed, 0 failed

=== Phase: Validate ===
  Testing basic... ✓
  Testing optional... ✓
  ...
```
