<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Report structure

`generate_final_reports.py` renders both files from `report_input.json`.
This describes what it produces, so you know what you are reading and what
is missing if a section is absent.

Do not re-render or reformat the output. Read it and quote it.

## `model_compatibility_report.md`

| # | Section | Present when |
|---|---|---|
| 1 | `# Model compatibility report` | always |
| 2 | metadata, including evidence level | always |
| 3 | evidence badge | level is not A |
| 4 | `## Summary` | always |
| 5 | `## What needs doing` | always |
| 6 | `## Original ONNX vs EP input` | the comparison ran |
| 7 | `## Operator distribution` | always |
| 8 | `## Capability gaps` | an attribute is ignored |
| 9 | `## Documentation drift` | an operator compiles but is undocumented |

### Summary

Total instances, then the five statuses. The percentage counts `supported`
only; the other four get their own lines. The denominator excludes weight
constants, and says so.

### What needs doing

The report's point, placed before the tables. Four groups, one per kind of
work, each sorted by instance count:

```
### Implement the operator          (unsupported)
### Extend an existing operator     (blocked)
### Complete the lowering chain     (lowering-broken)
### Handle an ignored attribute     (partial)
```

Per entry: signature, data types, shapes, attributes, and where it applies
the existing implementation, the ignored attributes, the lowering error, or
which operand blocks conversion.

Two fields read `_to be filled in_`: **converter source** and **root
cause**. They are yours to complete -- finding them means reading the
converter, which no script does. An entry left with both blanks is an
unfinished report.

### Operator distribution

Nine columns:

```
Op Type | Domain | Count | Data Types | Shapes | Status | Target |
Backend (runtime) | Description
```

`Target` is what the operator became, observed: a `hip.*` operation, a
`tensor.*` one, or `(folded at compile time)` when nothing in the output
carries its source line. `Backend` is `Compile-time` when the lowering
reaches no runtime symbol, otherwise the documented implementation with the
symbol.

`Shapes` matters because dynamic shapes are a common reason an existing
converter rejects a model.

### Capability gaps

Attributes no converter reads, proven by perturbation. `Set by the model`
distinguishes a defect here from a gap another model would hit. A gap with
`no` does not affect this model but would silently miscompile one that sets
the attribute.

### Documentation drift

Operators that compile but are missing from
`docs/supported-operations.md`. Informational; the probe outranks the doc.

## `model_compatibility_details.md`

Every operator in one table, including the ones the report summarizes:
status, target, runtime symbol, data types, ignored attributes.

## Rules

1. Every number comes from `report_input.json`. Do not recompute or round.
2. A missing field renders as `—`.
3. State the evidence level when it is not A, before anything else.
4. At level D, lead with the caveat: no compilation happened, so operators
   whose implementation rejects this model read as supported.
