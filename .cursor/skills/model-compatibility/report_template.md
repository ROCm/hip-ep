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
| 5 | `## Original ONNX vs EP input` | the comparison ran |
| 6 | `## Operator distribution` | always |
| 7 | `### Compatibility summary` | always |
| 8 | `## What needs doing` | always |
| 9 | `## Capability gaps` | an attribute is ignored |
| 10 | `## Documentation drift` | an operator compiles but is undocumented |

Ordered widest to narrowest: the counts, then every operator, then the same
operators bucketed by status, then only those needing work.

### Summary

Total instances, then the five statuses. The percentage counts `supported`
only; the other four get their own lines. The denominator excludes weight
constants, and says so. When some supported operators had no lowering
check, the supported line says how many, since they are inside that figure
rather than beside it.

A line above the percentage says whether the model compiles as a whole.
That is a different question from how many operators are supported, and at
level C the answer is no while the percentage can still be high.

### Compatibility summary

The same operators as the table above, bucketed by status, one line each:
where a supported operator lowers to and which runtime symbol it reaches,
or for the rest, what stands in the way. Easier to scan than the nine-column
table when the question is "what is in each bucket".

### What needs doing

One row per operator needing attention, with the kind of work and a one-line
finding: which operand blocks conversion, which attribute the converter
ignores, or the error.

| Work | Comes from |
|---|---|
| Implement the operator | `unsupported` |
| Extend an existing operator | `blocked` |
| Finish the lowering pipeline | `lowering-broken` |
| Handle an ignored attribute | `partial` |
| Extend the MorphiZen ONNX importer | the importer refused the operator |
| Check the lowering by hand | supported, but no slice could be built |

The last two are not statuses; see SKILL.md. Supporting evidence is in the
details file.

Neither this section nor the report names the converter source or the exact
check that rejects the model. Both require reading the converter, so they
are yours to add -- see SKILL.md.

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

The evidence behind each worklist row -- signature, data types, shapes,
attributes, documented implementation, blocking operand, ignored
attributes, lowering error, and the line in the EP input MLIR. Nothing
else: every operator is already listed in the report.

## Rules

1. Every number comes from `report_input.json`. Do not recompute or round.
2. A missing field renders as `—`.
3. State the evidence level when it is not A, before anything else.
4. At level D, lead with the caveat: no compilation happened, so operators
   whose implementation rejects this model read as supported.
