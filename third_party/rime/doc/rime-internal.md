**Date:** 2026-09-21
**Document Type:** Implementation
**Status:** Draft
**Related:** [loop.md](loop.md) - User-facing loop API

---

## Overview

The `(rime loop)` macro uses a plugin-based architecture for code generation. Each loop clause (`:for`, `:collect`, etc.) creates a plugin function that responds to method calls. The code generator calls these methods to build nested let expressions. This document explains the expansion structure, plugin pattern, method dispatch, and common bugs.

---

## Loop Expansion Structure

The loop macro expands into a nested structure of let expressions. Understanding this structure is essential for writing plugins correctly.

### The LOOP-SKELETONE Template

From `rime/loop/plugin.sls`, the template shows what every loop expands to:

```scheme
(let* SETUP-BINDINGS              ; from 'setup method
  (let RECUR-NAME RECUR-BINDINGS  ; from 'recur method
    (let* OUTER-BINDINGS          ; from 'outer-iteration method
      (let ITER-NAME ITERATION-BINDINGS  ; from 'iteration method
        (let* INNER-BINDINGS      ; from 'inner-iteration method
          (if CONTINUE-CONDITION  ; from 'continue-condition method
              (let* INNER-IF-TRUE-BINDINGS  ; from 'inner-if-true method
                ITERATION-BODY    ; from 'iteration-body method
                (ITER-NAME STEP-EXPR))  ; step from 'iteration method
              FINALLY))))))       ; from 'finally method
```

**Key insight:** Each level serves a specific purpose:

**Level 1: SETUP-BINDINGS** (anonymous `let*`)

One-time setup before loop starts. Rarely used.

**Level 2: RECUR-NAME with RECUR-BINDINGS** (named `let`)

Stores **initial/constant** values. If you call `(RECUR-NAME ...)`, loop restarts from these values.

For `:for i :in '(1 2 3)`: stores the **original list** `'(1 2 3)`

For `:recur x := 1 :then (+ x 1)`: stores **initial value** `x = 1`

**Level 3: OUTER-BINDINGS** (anonymous `let*`)

Additional outer bindings. Rarely used.

**Level 4: ITER-NAME with ITERATION-BINDINGS** (named `let`)

Stores **current values** that change each iteration. Calling `(ITER-NAME STEP-EXPR)` advances to next iteration.

For `:for i :in '(1 2 3)`: stores **current list** that steps: `'(1 2 3)` → `'(2 3)` → `'(3)` → `'()`

For `:recur x := 1 :then (+ x 1)`: stores **current value** that steps: `1` → `2` → `3` → ...

**Level 5: INNER-BINDINGS** (anonymous `let*`)

Bindings created each iteration before condition test.

**Level 6: CONTINUE-CONDITION** (if test)

Boolean expression: should we continue looping?

**Level 7: INNER-IF-TRUE-BINDINGS** (anonymous `let*`)

Bindings created when continuing (condition is true).

For `:for i :in '(1 2 3)`: extracts **current element** `(car current-list)` into variable `i`

**Level 8: ITERATION-BODY**

The actual loop body code.

**Level 9: FINALLY**

Code executed after loop ends.

### Concrete Example 1: Vector Iteration

User writes:

```scheme
(loop :for i :in-vector (vector 10 20 30) :collect i)
```

This expands to:

```scheme
(let* ()  ; setup - empty
  (let recur ((i-vector-expr (vector 10 20 30)))  ; recur - store ORIGINAL vector
    (let* ()  ; outer-iteration - empty
      (let loop ((i-vector-index 0))  ; iteration - CURRENT index: 0 → 1 → 2 → 3
        (let* ()  ; inner-iteration - empty
          (if (< i-vector-index (vector-length i-vector-expr))  ; continue-condition
              (let* ((i (vector-ref i-vector-expr i-vector-index)))  ; inner-if-true
                ...collect i...  ; iteration-body
                (loop (+ 1 i-vector-index)))  ; step to next index
              result))))))  ; finally
```

**Bindings from plugin methods:**

- `(plugin 'recur)` → `i-vector-expr` = original vector (constant, can restart from here)
- `(plugin 'iteration)` → `i-vector-index` = current index (steps: 0 → 1 → 2 → 3)
- `(plugin 'continue-condition)` → `(< i-vector-index ...)`
- `(plugin 'inner-if-true)` → `i` = current element

### Concrete Example 2: List Iteration

User writes:

```scheme
(loop :for i :in '(a b c) :collect i)
```

This expands to:

```scheme
(let* ()
  (let recur ((i-list-recur '(a b c)))  ; recur - store ORIGINAL list
    (let* ()
      (let loop ((i-list-current i-list-recur))  ; iteration - CURRENT list
        (let* ()
          (if (not (null? i-list-current))  ; continue-condition
              (let* ((i (car i-list-current)))  ; inner-if-true - extract first element
                ...collect i...
                (loop (cdr i-list-current)))  ; step - move to rest of list
              result))))))
```

**Key difference from vector:**

- Vector: `recur` stores vector, `iteration` stores index (integer counter)
- List: `recur` stores original list, `iteration` stores current sublist (list pointer)

Both follow the same pattern:
- **`recur`**: initial/constant value (restart point)
- **`iteration`**: current value that steps forward

---

## Plugin Architecture

Now that you understand what the loop expands to, let's see how plugins generate these sections.

### Plugin Pattern

A plugin is a closure that responds to method names:

```scheme
(define (make-vector-plugin s-var s-expr)
  (lambda (method . args)
    (case method
      [(recur)
       (list #'(expr-var expr))]
      [(iteration)
       (list #'[var-index 0 (+ 1 var-index)])]
      [(continue-condition)
       #'(< var-index (vector-length expr-var))]
      [else
       (apply default-plugin method args)])))
```

Code generator calls methods to get pieces:

```scheme
;; Build the expansion by calling plugin methods
(plugin 'recur)              → '((i-vector-expr (vector ...)))
(plugin 'iteration)          → '((i-vector-index 0 (+ 1 ...)))
(plugin 'continue-condition) → '(< i-vector-index ...)
```

### Method Dispatch

When method name matches: returns syntax objects for that section.

When method name doesn't match: falls to `[else ...]` clause.

The `[else ...]` clause typically delegates to `default-plugin`, which returns empty list `'()` for unknown methods.

**Critical:** If a method returns `'()`, no bindings are created for that section. The expansion will still reference variables that were never bound, causing "unbound identifier" errors.

---

## Plugin Methods Reference

### Required Methods

**`'recur`**

Variables that stay constant across all iterations.

For `:in-vector`: stores vector expression once

```scheme
[(recur)
 (list #'(i-vector-expr (vector 1 2 3)))]
```

Expansion:

```scheme
(let recur ((i-vector-expr (vector 1 2 3)))
  ...)
```

**`'iteration`**

Loop counter variables that change each iteration.

Format: `(var-name initial-value step-expression)`

For `:in-vector`: index counter

```scheme
[(iteration)
 (list #'[i-vector-index 0 (+ 1 i-vector-index)])]
```

Expansion:

```scheme
(let loop ((i-vector-index 0))
  ...
  (loop (+ 1 i-vector-index)))
```

**`'continue-condition`**

Test whether to continue looping.

For `:in-vector`: check index < length

```scheme
[(continue-condition)
 #'(< i-vector-index (vector-length i-vector-expr))]
```

Expansion:

```scheme
(if (< i-vector-index (vector-length i-vector-expr))
    ...)
```

**`'inner-if-true`**

Bindings created when continue-condition is true.

For `:in-vector`: extract current element

```scheme
[(inner-if-true)
 (list #'[i (vector-ref i-vector-expr i-vector-index)])]
```

Expansion:

```scheme
(if ...
    (let* ((i (vector-ref i-vector-expr i-vector-index)))
      ...))
```

### Optional Methods

**`'setup`** - One-time setup before loop starts

**`'outer-iteration`** - Outer loop bindings (for nested loops)

**`'inner-iteration`** - Bindings inside each iteration before condition test

**`'iteration-body`** - The loop body code

**`'finally`** - Code after loop ends

**`'debug`** - Debug string representation

---

## Code Generator Flow

From `rime/loop/core.sls`:

**Step 1: Parse**

```scheme
(loop/core/vector e)
  → creates plugin function
  → stores in clauses list
```

**Step 2: Code Generation**

Line 234 calls `'iteration` on all plugins:

```scheme
[([<<iteration-binding>> <<iteration-value>> <<step-expr>>] ...)
 (apply append (map (lambda (f) (f 'iteration)) clauses))]
```

**Step 3: Template Expansion**

Builds nested let forms using bindings from each method.

---

## Plugin Implementation Checklist

When implementing a new loop clause plugin:

**Method names must match what core.sls calls:**

- Use `'iteration` for loop counter variables
- Use `'recur` for constant bindings
- Use `'continue-condition` for loop test
- Use `'inner-if-true` for element extraction

**Check core.sls for exact method names:**

```bash
grep "(f '" rime/loop/core.sls
```

**Test with unbound identifier check:**

Enable strict checking in Chez Scheme 10.4.1+ to catch missing bindings.

**Compare with working plugins:**

Reference `arithmetic.sls`, `string.sls`, or `list.sls` for correct patterns.

---

## Common Issues

### Unbound Identifier Error

**Symptom:** Error during macro expansion: "attempt to reference unbound identifier"

**Cause:** Method name mismatch - plugin defines one name, code generator calls another.

**Fix:** Verify `case method` clause names match what `core.sls` calls. Check existing plugins (`arithmetic.sls`, `string.sls`, `list.sls`) for correct method names.

### Empty Bindings

**Symptom:** Loop variables not created, or loop doesn't iterate.

**Cause:** Method returns `'()` when it should return bindings.

**Fix:** Verify method implementation returns non-empty list with proper syntax structure.

### Wrong Expansion

**Symptom:** Loop expands but behaves incorrectly.

**Cause:** Method returns incorrect syntax structure or binding format.

**Fix:** Compare with `LOOP-SKELETONE` template. For `'iteration`, format must be `(name initial-value step-expression)`.

---

## Related Documents

- [loop.md](loop.md) - User-facing loop API documentation
- `rime/loop/plugin.sls` - LOOP-SKELETONE template definition
- `rime/loop/core.sls` - Code generator implementation
- `rime/loop/vector.sls` - Vector iteration plugin (reference implementation)
