# Question About Literal Identifiers in `syntax-case` Across Libraries

**Date:** 2026-09-19  
**Context:** R6RS Scheme, Chez Scheme 9.5.8  
**Reference:** `/home/chunywan/w/work-log-2026/main/tech/guides/2026-09-19-learn-scheme.md`

---

## The Guide Says This Works

From section "Can We Use Symbol Operators as Literals?":

```scheme
(define-syntax assignment
  (lambda (x)
    (syntax-case x (=)  ; = as literal
      [(_ var = expr)
       #'(list 'assign 'var 'expr)])))

(assignment x = 42)
;; → (assign x 42)
```

**This works perfectly** ✅

---

## But Does It Work Across Library Boundaries?

### Test Case 1: Export Keywords Without Defining Them

**Library:**
```scheme
(library (test-literal-keywords)
  (export test-macro = : ->)  ;; <- Export symbols
  (import (rnrs))

  ;; DON'T define them, just use as literals
  (define-syntax test-macro
    (lambda (x)
      (syntax-case x (= : ->)
        [(_ lhs = rhs)
         #'(list 'assign 'lhs 'rhs)]))))
```

**Error:**
```
Exception: missing definition for multiple exports, including -> 
at line 6, char 26 of test-literal-keywords.sls
```

**Conclusion:** Cannot export undefined identifiers ❌

---

### Test Case 2: Define Keywords, But R6RS Already Defines `=`

**Library:**
```scheme
(library (test-literal-keywords-v2)
  (export test-macro = : ->)
  (import (rnrs))

  ;; Define keywords
  (define-syntax = (lambda (x) (syntax-violation 'keyword "misplaced" x)))
  (define-syntax : (lambda (x) (syntax-violation 'keyword "misplaced" x)))
  (define-syntax -> (lambda (x) (syntax-violation 'keyword "misplaced" x)))

  (define-syntax test-macro
    (lambda (x)
      (syntax-case x (= : ->)
        [(_ lhs = rhs)
         #'(list 'assign 'lhs 'rhs)]))))
```

**Error:**
```
Exception: multiple definitions for = in body
```

**Conclusion:** R6RS `(rnrs)` already exports `=` ❌

---

### Test Case 3: Exclude `=`, Define All Keywords

**Library:**
```scheme
(library (test-literal-keywords-v3)
  (export test-macro = : ->)
  (import (except (rnrs) =))  ;; <- Exclude R6RS =

  ;; Define our keywords
  (define-syntax = (lambda (x) (syntax-violation 'keyword "misplaced" x)))
  (define-syntax : (lambda (x) (syntax-violation 'keyword "misplaced" x)))
  (define-syntax -> (lambda (x) (syntax-violation 'keyword "misplaced" x)))

  (define-syntax test-macro
    (lambda (x)
      (syntax-case x (= : ->)
        [(_ lhs = rhs)
         #'(list 'assign 'lhs 'rhs)]))))
```

**Client:**
```scheme
(import (rnrs)
        (test-literal-keywords-v3))

(test-macro x = 42)  ;; <- Use the macro
```

**Error:**
```
Exception: invalid syntax (test-macro x = 42) 
at line 8, char 10 of test-use-literal-keywords-v3.scm
```

**Conclusion:** Pattern doesn't match! ❌

---

## The Question

**Why doesn't the pattern match in Test Case 3?**

We:
1. ✅ Defined `=`, `:`, `->` as syntax in the library
2. ✅ Exported them
3. ✅ Listed them in `syntax-case` literals clause `(= : ->)`
4. ✅ Imported them in the client file
5. ✅ The pattern structure is correct: `[(_ lhs = rhs) ...]`

But the pattern **fails to match** when:
- Macro is defined in library A
- Macro is used in library/script B
- Keywords are imported from library A

**The guide shows examples working within a single file/library.**  
**Does literal matching work differently across library boundaries?**

---

## What We Expected

Based on the guide and R6RS hygiene:
- `free-identifier=?` should match imported `=` with literal `=`
- Both refer to the same binding (exported from library)
- Pattern should match

---

## What Actually Happens

Pattern doesn't match, macro falls through to default case or syntax error.

---

## Workaround We Found

Research shows we need to either:

**Option A:** Match keywords in main transformer, pass keyword-free data to helpers
```scheme
(define-syntax define-conversion-pattern
  (lambda (x)
    (syntax-case x (:match = : ->)  ;; Match here
      [(_ fname :match (%x eq op colon type arrow type2) ...)
       (and (free-identifier=? #'eq #'=)  ;; Verify
            (free-identifier=? #'colon #':)
            (free-identifier=? #'arrow #'->))
       (helper #'fname #'%x #'op ...)])))  ;; No keywords passed
```

**Option B:** Use datum matching in helpers (loses hygiene)
```scheme
(define (helper stx)
  (syntax-case stx ()
    [(lhs eq rhs)
     (eq? (syntax->datum #'eq) '=)  ;; Check name, not binding
     ...]))
```

---

## Request

Can you verify whether:
1. The guide examples assume single-file/single-library usage?
2. Cross-library literal matching requires additional techniques?
3. Is there a correct way to make Test Case 3 work?

**Test files:** `/workspace/hip-ep/hip-ep-1/test-literal-keywords-v*.sls`

Thank you!
