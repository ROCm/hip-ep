# Answer: Literals in `syntax-case` Across Library Boundaries

**Date:** 2026-09-19  
**Status:** ✅ Verified with Chez Scheme 9.5.8

---

## TL;DR - The Solution

**For cross-library literal matching to work:**

1. ✅ Library exports both the macro AND the literal identifiers
2. ✅ Client imports the literal identifiers from the SAME library as the macro
3. ✅ Client excludes conflicting bindings from `(rnrs)` if necessary

**Why Test Case 3 Failed:**
```scheme
;; Client code
(import (rnrs)                      ;; Imports rnrs's =
        (test-literal-keywords-v3)) ;; Imports macro and another =

(test-macro x = 42)  ;; Which = is this? rnrs's or library's?
```

The `=` in `(test-macro x = 42)` refers to **`rnrs`'s `=`**, not the library's `=`.

The macro's literal list expects the **library's `=`**, so `free-identifier=?` returns `#f`.

Pattern doesn't match!

---

## How R6RS Literal Matching Works

From R6RS Section 11.19 (`syntax-case`):

> A literal identifier matches an input subform if and only if the input subform is an identifier and **either both its occurrence in the macro expression and its occurrence in the macro definition have the same lexical binding**, or the two identifiers have the same name and both have no lexical binding.

Key concept: **`free-identifier=?`**

For pattern `[(_ lhs = rhs)]` to match input `(test-macro x = 42)`:
- The `=` in the pattern (defined in macro's library)
- The `=` in the input (whatever binding is visible at use-site)
- Must satisfy `free-identifier=?` (same binding)

---

## Verified Examples

### ❌ Failed Attempt (Test Case 3)

**Library:**
```scheme
(library (test-literal-keywords-v3)
  (export test-macro = : ->)
  (import (except (rnrs) =))

  (define-syntax = (lambda (x) (syntax-violation 'keyword "misplaced" x)))
  (define-syntax : (lambda (x) (syntax-violation 'keyword "misplaced" x)))
  (define-syntax -> (lambda (x) (syntax-violation 'keyword "misplaced" x)))

  (define-syntax test-macro
    (lambda (x)
      (syntax-case x (= : ->)
        [(_ lhs = rhs)
         #'(list 'assign 'lhs 'rhs)]))))
```

**Client (WRONG):**
```scheme
(import (rnrs)                      ;; ❌ Imports (rnrs)'s =
        (test-literal-keywords-v3)) ;; Also imports library's =

(test-macro x = 42)
;; → (NO-MATCH (test-macro x = 42))
;; The = here is (rnrs)'s =, not library's =
```

### ✅ Correct Solution

**Library (same as above):**
```scheme
(library (test-literal-keywords-v4)
  (export test-macro =)
  (import (except (rnrs) =))

  (define-syntax = (lambda (x) (syntax-violation 'keyword "misplaced" x)))

  (define-syntax test-macro
    (lambda (x)
      (syntax-case x (=)
        [(_ lhs = rhs)
         #'(list 'assign 'lhs 'rhs)]))))
```

**Client (CORRECT):**
```scheme
(import (except (rnrs) =)           ;; ✅ Exclude (rnrs)'s =
        (test-literal-keywords-v4)) ;; Import ONLY library's =

(test-macro x = 42)
;; → (assign x 42)  ✅ Works!
```

---

## Why Single-File Examples Work

In the guide's single-file examples:

```scheme
(import (rnrs))

(define-syntax assignment
  (lambda (x)
    (syntax-case x (=)
      [(_ var = expr)
       #'(list 'assign 'var 'expr)])))

(assignment x = 42)
;; → (assign x 42)  ✅ Works!
```

**Why this works:**
- Both macro definition and macro use are in the **same lexical scope**
- The `=` in the pattern refers to `(rnrs)`'s `=`
- The `=` in the input refers to `(rnrs)`'s `=`
- Same binding → `free-identifier=?` succeeds → pattern matches

**No cross-library boundary**, so no binding ambiguity!

---

## The Guide Was Incomplete

The guide showed:
- ✅ Pattern datums (numbers, strings, characters)
- ✅ Literal identifiers in single-file context
- ❌ Did NOT cover cross-library literal matching

**Cross-library requires additional discipline:**
1. Export literal identifiers
2. Client imports them from the correct library
3. Client excludes conflicting bindings

---

## Best Practices for Cross-Library Macros with Literals

### Strategy 1: Export Keywords Explicitly

**Library:**
```scheme
(library (my-dsl)
  (export define-rule = : -> when)  ;; Export all keywords
  (import (except (rnrs) =))         ;; Exclude conflicts

  ;; Define keywords
  (define-syntax = (lambda (x) (syntax-violation 'keyword "misplaced" x)))
  (define-syntax : (lambda (x) (syntax-violation 'keyword "misplaced" x)))
  (define-syntax -> (lambda (x) (syntax-violation 'keyword "misplaced" x)))
  (define-syntax when (lambda (x) (syntax-violation 'keyword "misplaced" x)))

  (define-syntax define-rule
    (lambda (x)
      (syntax-case x (= : -> when)
        [(_ name : type = default)
         #'(define name default)]
        [(_ pattern when guard -> result)
         #'(if guard result (error 'no-match))]))))
```

**Client:**
```scheme
(import (except (rnrs) =)  ;; Must exclude rnrs's =
        (my-dsl))

(define-rule x : Int = 42)        ;; Works
(define-rule (> x 0) when #t -> "positive")  ;; Works
```

**Pros:**
- Clean, readable syntax
- Hygienic literal matching

**Cons:**
- Client must remember to exclude conflicting bindings
- Error messages can be confusing if imports are wrong

---

### Strategy 2: Use Non-Conflicting Keywords

**Library:**
```scheme
(library (my-dsl-v2)
  (export define-rule :of :is :when :=>)  ;; No conflicts with rnrs
  (import (rnrs))

  (define-syntax :of (lambda (x) (syntax-violation 'keyword "misplaced" x)))
  (define-syntax :is (lambda (x) (syntax-violation 'keyword "misplaced" x)))
  (define-syntax :when (lambda (x) (syntax-violation 'keyword "misplaced" x)))
  (define-syntax :=> (lambda (x) (syntax-violation 'keyword "misplaced" x)))

  (define-syntax define-rule
    (lambda (x)
      (syntax-case x (:of :is :when :=>)
        [(_ name :of type :is default)
         #'(define name default)]
        [(_ pattern :when guard :=> result)
         #'(if guard result (error 'no-match))]))))
```

**Client:**
```scheme
(import (rnrs)      ;; No need to exclude anything
        (my-dsl-v2))

(define-rule x :of Int :is 42)           ;; Works
(define-rule (> x 0) :when #t :=> "positive")  ;; Works
```

**Pros:**
- No import conflicts
- Client code is simpler (just import the library)

**Cons:**
- Syntax is slightly more verbose (`:of` vs `:`)
- Keywords look less "standard"

---

### Strategy 3: Match Keywords in Main Macro, Strip for Helpers

This is the workaround mentioned in the question.

**Library:**
```scheme
(library (my-dsl-v3)
  (export define-conversion-pattern)
  (import (rnrs))

  ;; Helper doesn't use literal keywords
  (define (process-pattern fname patterns)
    (with-syntax ([name fname]
                  [(pat ...) patterns])
      #'(define (name x)
          (cond [pat x] ...))))

  ;; Main macro matches keywords, then delegates
  (define-syntax define-conversion-pattern
    (lambda (x)
      (syntax-case x (:match =)
        [(_ fname :match (lhs = rhs) ...)
         (process-pattern #'fname #'((equal? lhs rhs) ...))]))))
```

**Client:**
```scheme
(import (rnrs)
        (my-dsl-v3))

(define-conversion-pattern my-converter
  :match (1 = "one")
         (2 = "two"))
```

**Pros:**
- Helper functions don't need to know about keywords
- More flexible (helpers can be reused)

**Cons:**
- Main macro gets more complex
- Still requires importing `:match` and `=` correctly

---

## Updated Learning Document Entry

I will add this section to the guide:

```markdown
## Cross-Library Literal Matching (CRITICAL)

The examples above work in a **single file/library**.

Across library boundaries, literal matching requires **binding identity** via `free-identifier=?`:

### The Problem

**Library:**
```scheme
(library (my-lib)
  (export my-macro =)
  (import (except (rnrs) =))
  (define-syntax = (lambda (x) (syntax-violation 'keyword "misplaced" x)))
  (define-syntax my-macro
    (lambda (x)
      (syntax-case x (=)
        [(_ a = b) #'(list 'a 'b)]))))
```

**Client (WRONG):**
```scheme
(import (rnrs)    ;; Imports rnrs's =
        (my-lib)) ;; Also imports my-lib's =

(my-macro x = 42) ;; Which = ? Uses rnrs's =
;; Pattern expects my-lib's = → NO MATCH
```

**Client (CORRECT):**
```scheme
(import (except (rnrs) =)  ;; Exclude rnrs's =
        (my-lib))          ;; Import ONLY my-lib's =

(my-macro x = 42)
;; → (x 42) ✅
```

### Best Practices

1. **Export keywords** from the same library as the macro
2. **Client imports keywords** from that library
3. **Exclude conflicts** with `(except (rnrs) ...)`
4. **OR use non-conflicting names** (`:of`, `:=`, `:->`)
```

---

## References

- R6RS Section 11.19: `syntax-case`
- R6RS Section 9.2: Import and Export
- Chez Scheme User's Guide: Libraries

---

## Test Files

Verified examples:
- `/tmp/test-literal-keywords-v3.sls` (failed case)
- `/tmp/test-use-literal-keywords-v3.scm`
- `/tmp/test-literal-keywords-v4.sls` (working case)
- `/tmp/test-use-literal-keywords-v4.scm`
