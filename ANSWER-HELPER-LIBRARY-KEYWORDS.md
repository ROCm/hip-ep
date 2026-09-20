# Answer: Cross-Library Keyword Matching for Helper Libraries

**Date:** 2026-09-20  
**Status:** ✅ Verified with Chez Scheme 9.5.8  
**Question:** How to make `syntax-case` literal matching work when parser is in a separate library?

---

## TL;DR - The Solution

**Use `(for <library> expand)` to import keywords at expansion time.**

```scheme
(library (my-parser)
  (export parse-it)
  (import (rnrs)
          (for (keyword-definitions) expand))  ;; <-- KEY: Import at expand time!
  
  (define (parse-it stx)
    (syntax-case stx (:foo :bar)  ;; Now :foo and :bar are in scope
      [(_ :foo x) #'(quote foo-matched)]
      [(_ :bar x) #'(quote bar-matched)])))
```

---

## The Problem

When you move a parser function into a separate library, keyword literals are not in scope:

```scheme
;; ❌ WRONG: Keywords not in scope
(library (my-parser-wrong)
  (export parse-it)
  (import (rnrs))  ;; Missing keyword imports!
  
  (define (parse-it stx)
    (syntax-case stx (:foo :bar)  ;; ERROR: :foo and :bar unbound
      [(_ :foo x) #'(quote foo)]
      [(_ :bar x) #'(quote bar)])))
```

**Error:** `Exception: identifier :foo out of context`

---

## The Solution: Three-Library Architecture

### Architecture Overview

```
keyword-definitions.sls    ← Defines :foo, :bar
         ↓
    parser.sls             ← Imports keywords with (for ... expand)
         ↓
    macro.sls              ← Imports parser, re-exports keywords
         ↓
    user-code.scm          ← Uses macro
```

### Library 1: Keyword Definitions

```scheme
(library (solution-lib-keywords)
  (export :foo :bar)
  (import (rnrs))

  (define-syntax :foo (lambda (x) (syntax-violation 'keyword "misplaced" x)))
  (define-syntax :bar (lambda (x) (syntax-violation 'keyword "misplaced" x))))
```

**Purpose:** Define keywords in one place, no dependencies.

### Library 2: Parser (The Key Part!)

```scheme
(library (solution-lib-parser)
  (export parse-it)
  (import (rnrs)
          (for (solution-lib-keywords) expand))  ;; ← SOLUTION!

  (define (parse-it stx)
    ;; Keywords :foo and :bar are now in scope at expand-time
    (syntax-case stx (:foo :bar)
      [(_ :foo x) #'(quote foo-matched)]
      [(_ :bar x) #'(quote bar-matched)]
      [_ #'(quote no-match)])))
```

**Key insight:** `(for (solution-lib-keywords) expand)` makes `:foo` and `:bar` available **at expansion time** when `syntax-case` runs.

### Library 3: Main Macro

```scheme
(library (solution-lib-macro)
  (export my-macro :foo :bar)  ;; Re-export keywords for users
  (import (rnrs)
          (for (solution-lib-keywords) expand)  ;; Keywords
          (for (solution-lib-parser) expand))   ;; Parser

  (define-syntax my-macro
    (lambda (stx)
      (parse-it stx))))
```

**Purpose:** Tie everything together, re-export keywords so users can write `(my-macro :foo 42)`.

### User Code

```scheme
(import (rnrs) (solution-lib-macro))

(my-macro :foo 42)  ;; → foo-matched
(my-macro :bar 99)  ;; → bar-matched
(my-macro :baz 10)  ;; → no-match
```

---

## Verified Test Results

```
Test 1: (my-macro :foo 42) => foo-matched
Test 2: (my-macro :bar 99) => bar-matched
Test 3: (my-macro :baz 10) => no-match

✅ SUCCESS: Cross-library keyword matching works!
```

**Test files:**
- `/tmp/solution-lib-keywords.sls`
- `/tmp/solution-lib-parser.sls`
- `/tmp/solution-lib-macro.sls`
- `/tmp/solution-final-test.scm`

---

## Why `(for ... expand)` Works

### R6RS Import Phases

From R6RS Section 7.1:

> The `for` form specifies the meta-level at which imported bindings are available:
> - `(for <import-set> run)` - available at run time (default)
> - `(for <import-set> expand)` - available at expansion time
> - `(for <import-set> (meta n))` - available at meta-level n

### When `syntax-case` Runs

`syntax-case` **executes at expansion time**, not run time:

```scheme
(define-syntax my-macro
  (lambda (stx)
    ;; This lambda runs at EXPANSION TIME
    (syntax-case stx (:foo)  ;; :foo must be bound at EXPANSION TIME
      ...)))
```

### Import Phase Matching

For literal matching to work:
1. `syntax-case` needs `:foo` at expansion time
2. Parser runs at expansion time (called from macro transformer)
3. Parser imports `:foo` with `(for ... expand)`
4. `:foo` is available when `syntax-case` executes
5. Pattern matches via `free-identifier=?`

**Without `(for ... expand)`:**
- `:foo` would be imported for run time only
- Not available at expansion time
- `syntax-case` can't find the binding
- Error: `identifier :foo out of context`

---

## Alternative Approaches (and Why They Don't Work)

### ❌ Approach 1: Pass Keywords as Parameters

```scheme
;; Doesn't work - loses binding identity
(define (parse-it stx foo-kw bar-kw)
  (syntax-case stx (foo-kw bar-kw)  ;; ERROR: these are variables, not literals
    ...))
```

**Problem:** `syntax-case` literals must be **literal identifiers**, not computed values.

### ❌ Approach 2: Use `free-identifier=?` Manually

```scheme
(define (parse-it stx)
  (syntax-case stx ()
    [(_ kw x)
     (if (free-identifier=? #'kw #':foo)  ;; ERROR: #':foo not in scope
         #'(quote foo)
         ...)]))
```

**Problem:** Still need `:foo` in scope to compare against it!

### ❌ Approach 3: Keep Everything in One Library

```scheme
(library (monolithic)
  (export my-macro :foo :bar)
  (import (rnrs))
  
  (define-syntax :foo ...)
  (define-syntax :bar ...)
  
  (define (parse-it stx)
    (syntax-case stx (:foo :bar)  ;; Works - same library
      ...))
  
  (define-syntax my-macro
    (lambda (stx) (parse-it stx))))
```

**Problem:** Works, but defeats the goal of separating concerns into multiple files.

---

## Best Practices

### 1. Separate Keyword Definitions

Create a dedicated library for keywords with **no dependencies**:

```scheme
(library (my-keywords)
  (export :match = : -> when)
  (import (rnrs))  ;; Minimal imports
  
  (define-syntax :match (lambda (x) (syntax-violation 'keyword "misplaced" x)))
  (define-syntax = (lambda (x) (syntax-violation 'keyword "misplaced" x)))
  ...)
```

**Benefits:**
- No circular dependencies
- Easy to import from multiple libraries
- Clear separation of concerns

### 2. Use `(for ... expand)` in Parser

Any library that uses keywords in `syntax-case` must import them with `(for ... expand)`:

```scheme
(library (my-parser)
  (export parse-it)
  (import (rnrs)
          (for (my-keywords) expand))  ;; Always use (for ... expand)
  
  (define (parse-it stx)
    (syntax-case stx (:match = : ->)
      ...)))
```

### 3. Re-export Keywords from Main Macro

Users shouldn't need to know about internal keyword library:

```scheme
(library (my-macro-api)
  (export my-macro :match = : ->)  ;; Re-export keywords
  (import (rnrs)
          (for (my-keywords) expand)
          (for (my-parser) expand))
  
  (define-syntax my-macro ...))
```

**User code:**
```scheme
(import (rnrs) (my-macro-api))  ;; One import gets everything
(my-macro :match x = 42)
```

### 4. Avoid Naming Conflicts

Use non-conflicting keyword names:
- ✅ `:match`, `:foo`, `:when` - Unlikely conflicts
- ⚠️ `=`, `:`, `->` - May conflict with `(rnrs)`

If using conflicting names, document that users must:
```scheme
(import (except (rnrs) =)  ;; Exclude rnrs's =
        (my-macro-api))    ;; Import library's =
```

---

## Comparison with Single-Library Approach

### Single Library (From Previous Answer)

```scheme
(library (single-lib)
  (export my-macro :foo)
  (import (rnrs))
  
  (define-syntax :foo ...)
  
  (define-syntax my-macro
    (lambda (x)
      (syntax-case x (:foo)  ;; Works - same lexical scope
        ...))))

(import (rnrs) (single-lib))
(my-macro :foo 42)  ;; Client must import :foo from single-lib
```

**Difference:**
- Single-library: Client imports keywords from **same library as macro**
- Multi-library: Parser imports keywords with **`(for ... expand)`**

Both rely on `free-identifier=?` for literal matching, but the **import phase** is critical for multi-library.

---

## Application to Your Refactoring

For your pattern-macro refactoring:

```scheme
;; pattern-keywords.sls
(library (pattern-keywords)
  (export :match = : -> when)
  (import (rnrs))
  (define-syntax :match ...)
  (define-syntax = ...)
  ...)

;; pattern-parse.sls
(library (pattern-parse)
  (export parse-pattern)
  (import (rnrs)
          (pattern-ast)
          (for (pattern-keywords) expand))  ;; KEY!
  
  (define (parse-pattern stx)
    (syntax-case stx (:match = : ->)
      ...)))

;; pattern-macro.sls
(library (pattern-macro)
  (export define-conversion-pattern :match = : ->)
  (import (rnrs)
          (for (pattern-keywords) expand)
          (for (pattern-parse) expand)
          (for (pattern-validate) expand)
          (for (pattern-codegen) expand))
  
  (define-syntax define-conversion-pattern
    (lambda (stx)
      (let* ([ast (parse-pattern stx)]
             [validated (validate-pattern ast)])
        (generate-code validated)))))
```

---

## Summary

✅ **Solution:** Use `(for <keyword-library> expand)` in parser library

✅ **Why it works:** Makes keywords available at expansion time when `syntax-case` runs

✅ **Verified:** Test shows `:foo` and `:bar` correctly matched across three libraries

❌ **Won't work:** Regular imports, parameter passing, or manual `free-identifier=?` without proper imports

**Key insight:** R6RS `syntax-case` literal matching requires **binding identity via `free-identifier=?`**, which requires the literal identifier to be **in scope at expansion time** in the library where `syntax-case` is called.
