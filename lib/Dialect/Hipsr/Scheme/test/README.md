# Scheme Pattern DSL Test Suite

Clean, organized test architecture with **zero duplication** - all test patterns defined once in a single source of truth.

## Quick Start

```bash
cd /workspace/hip-ep/hip-ep-1/lib/Dialect/Hipsr/Scheme
scheme --libdirs .:libraries --program test/run-all-tests.scm
```

## Architecture: Single Source of Truth

```
test/
├── test-patterns.sls          # ⭐ SINGLE SOURCE: All test patterns defined here
├── phase-1-parse-test.sls     # Tests parsing (imports test-patterns)
├── phase-2-validate-test.sls  # Tests validation (imports test-patterns)
├── phase-3-analyze-test.sls   # Tests analysis (imports test-patterns)
├── phase-4-codegen-test.sls   # Tests codegen (imports test-patterns)
├── integration-test.sls       # Tests all phases (imports test-patterns)
├── test-framework.sls         # Test harness (assertions, lifecycle)
├── mock-ffi.sls               # Mock FFI functions
└── run-all-tests.scm          # Test runner
```

### Key Design: No Duplication

**Problem (Old):** Test patterns duplicated across parse-test.sls, validate-test.sls, analyze-test.sls, codegen-test.sls → maintenance nightmare

**Solution (New):** Define each test pattern ONCE in `test-patterns.sls` with 5 versions:
- `pattern-*-parse`: Uses `:debug-parse` flag (returns AST after parse)
- `pattern-*-validate`: No debug flags (validates successfully)
- `pattern-*-analyze`: Uses `:debug-analyze` flag (returns AST with actions)
- `pattern-*-codegen`: Uses `:debug-codegen` flag (returns quoted code)
- `pattern-*-lambda`: No debug flags (normal lambda generation)

**Example:**
```scheme
;; test-patterns.sls - define once
(define-conversion-pattern :debug-parse pattern-basic-parse
  :match %out = "test.op" (%in)
  :rewrite %out :with (%new = "new.op" (%in)))

(define-conversion-pattern pattern-basic-validate
  :match %out = "test.op" (%in)
  :rewrite %out :with (%new = "new.op" (%in)))

(define-conversion-pattern :debug-analyze pattern-basic-analyze
  :match %out = "test.op" (%in)
  :rewrite %out :with (%new = "new.op" (%in)))
;; ... and so on
```

Phase test files just **import and assert**:
```scheme
;; phase-1-parse-test.sls
(import (test test-patterns))
(test-equal "basic: has root-op-name" "test.op"
  (ast-get pattern-basic-parse 'root-op-name))
```

## Test Patterns (12 Canonical Patterns)

All defined in `test-patterns.sls`:

1. **pattern-basic**: Single operation, simple match/rewrite
2. **pattern-required**: Multiple required operands
3. **pattern-optional**: `(&optional ...)` operand group
4. **pattern-variadic**: `(&variadic ...)` operand group
5. **pattern-mixed**: Required + optional + variadic
6. **pattern-where**: `:where` guard (per-operation)
7. **pattern-then-let**: `:then-let` bindings (global)
8. **pattern-two-ops**: Two operations, DAG traversal
9. **pattern-reuse**: Same operand twice (check-eq)
10. **pattern-combined**: :where + :then-let + multiple ops
11. **pattern-region**: Regions with blocks
12. **pattern-multi-rewrite**: Multiple rewrite operations

Each pattern has 5 versions: `-parse`, `-validate`, `-analyze`, `-codegen`, `-lambda`

## Phase-Specific Tests

### Phase 1: Parse (`phase-1-parse-test.sls`)

Tests ONLY parsing:
- Operand parsing and flattening
- :where guard parsing
- :then-let binding parsing
- Region/block structure parsing
- Debug flags

Uses `:debug-parse` patterns to inspect AST structure.

### Phase 2: Validate (`phase-2-validate-test.sls`)

Tests ONLY validation:
- % prefix validation
- Duplicate result variable detection
- Root var existence validation
- Normalization (list → vector, symbol → string)

Uses non-debug patterns to verify successful validation.

### Phase 3: Analyze (`phase-3-analyze-test.sls`)

Tests ONLY analysis:
- Action generation from match operations
- Binding manager creation
- Operand segment handling (optional/variadic)
- DAG traversal order

Uses `:debug-analyze` patterns to inspect actions list.

### Phase 4: Codegen (`phase-4-codegen-test.sls`)

Tests ONLY code generation:
- Lambda signature generation
- Variable initialization code
- :where guard code generation
- :then-let binding code generation
- Optional/variadic operand access code

Uses `:debug-codegen` patterns to inspect quoted code.

### Integration (`integration-test.sls`)

Tests all phases together:
- Complete pipeline (parse → validate → analyze → codegen)
- Generated lambdas are callable
- Pattern matching behavior (with mock FFI)

Uses normal patterns (no debug flags) to verify lambda generation.

## Test Framework API

```scheme
(test-begin "suite-name")     ; Start test suite
(test-end)                    ; End suite, exit(1) if failures
(test-equal desc actual expected)  ; Assert equality
(test-assert desc condition)  ; Assert truth
(test-error desc thunk)       ; Assert error raised
```

## Running Specific Tests

```bash
cd /workspace/hip-ep/hip-ep-1/lib/Dialect/Hipsr/Scheme

# Run only parse tests
scheme --libdirs .:libraries --program test/phase-1-parse-test.sls

# Run only codegen tests
scheme --libdirs .:libraries --program test/phase-4-codegen-test.sls

# Run all tests
scheme --libdirs .:libraries --program test/run-all-tests.scm
```

## Adding New Tests

To add a new test pattern:

1. **Define pattern in test-patterns.sls** (5 versions):
```scheme
(define-conversion-pattern :debug-parse pattern-new-feature-parse
  :match ...)

(define-conversion-pattern pattern-new-feature-validate
  :match ...)

(define-conversion-pattern :debug-analyze pattern-new-feature-analyze
  :match ...)

(define-conversion-pattern :debug-codegen pattern-new-feature-codegen
  :match ...)

(define-conversion-pattern pattern-new-feature-lambda
  :match ...)
```

2. **Export pattern names** in test-patterns.sls export list

3. **Add assertions** in relevant phase test files:
```scheme
;; phase-1-parse-test.sls
(test-equal "new-feature: parses as AST" #t 
  (list? pattern-new-feature-parse))
```

**Benefits:**
- ✅ Pattern defined ONCE, not 4+ times
- ✅ Easy to see what each pattern tests
- ✅ Changes to pattern automatically propagate to all tests
- ✅ Adding new test: define pattern once, add assertions

## Debug Flags

- `:debug-parse` - Returns AST list instead of lambda (parse phase only)
- `:debug-analyze` - Returns AST with actions list (parse + validate + analyze)
- `:debug-codegen` - Returns quoted generated code (all phases, code as data)
- `:debug-matching` - Runtime debugging (prints match progress when pattern runs)

## File Organization Principles

**Keep:**
- `test-patterns.sls` - Single source of truth for patterns
- `phase-*-test.sls` - Phase-specific test assertions
- `integration-test.sls` - End-to-end tests
- `test-framework.sls` - Test harness
- `mock-ffi.sls` - Mock FFI functions
- `run-all-tests.scm` - Test runner
- `README.md` - This file

**Clean:**
- ❌ No duplicate pattern definitions
- ❌ No obsolete/experimental files (case4-*.scm, show-*.scm)
- ❌ No disabled tests (*.disabled)
- ❌ No redundant test runners (run-simple-test.scm, etc.)

## Design Goals Achieved

1. ✅ **DRY (Don't Repeat Yourself)**: Patterns defined once, used everywhere
2. ✅ **Single Source of Truth**: test-patterns.sls is the canonical source
3. ✅ **Clean Directory**: No clutter, no obsolete files
4. ✅ **Easy to Extend**: Add pattern once, works across all phases
5. ✅ **Clear Separation**: Each phase tests only its concern
6. ✅ **Maintainable**: Change pattern once, all tests updated

## Notes

- TODOs in test files mark places for deeper inspection (requires analyzing AST/action records)
- Invalid patterns (missing %, duplicates) cause syntax-violation at macro expansion time
- Mock FFI functions defined in integration-test.sls for testing without real MLIR operations
