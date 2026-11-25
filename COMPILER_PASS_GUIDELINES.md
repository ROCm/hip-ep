<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->

# Guidelines for generic compiler passes

Keep compiler passes generic wherever possible. Use explicit IR information
and shared analyses. Keep frontend-specific assumptions and backend-specific
policy out of generic pass logic.

## 1. Use IR semantics, not frontend or backend assumptions

A generic pass bases its decisions on the semantics and analyses of the IR it
operates on, not on frontend-specific assumptions or backend-specific policy.

Use operation semantics, types, attributes, interfaces, and analysis results.

Do not assume that:

- `memref.load` or `memref.store` always runs on the host.
- A buffer is host-accessible.
- A compute operation always runs on a GPU.
- CPU execution is always synchronous or GPU execution is always asynchronous.
- An earlier operation has finished using a buffer.
- A pattern observed in one model applies to all models.

Check the required facts through an explicit contract or analysis.

## 2. Separate generic algorithms from frontend and backend logic

Keep each pass focused on one job. Reuse existing passes, analyses, and
interfaces. Prefer standard MLIR support before adding project-specific logic.

Generic passes should reason from IR semantics and contracts expressed through
types, attributes, interfaces, and analyses. Do not introduce frontend- or
backend-specific decisions into generic passes merely by putting them behind
an interface.

Frontend conversion must express the required source semantics in the IR.
Generic passes should not need to know which framework or exporter produced it.

A backend pass makes decisions about target execution, memory placement, or
runtime behavior and may use backend contracts. Any constraints needed by
later generic passes must be represented in the IR. A pass's name or source
directory does not determine whether it is generic.

For example, a buffer-reuse pass should reason about aliasing, lifetimes, and
execution dependencies represented in the IR. Backend passes must preserve
those guarantees when choosing memory placement and implementing
synchronization.

## 3. Keep operation rules with the operation

- Keep shape rules in the operation's inference or reification implementation.
- Describe aliasing and memory effects through interfaces.
- Let generic passes query these interfaces instead of repeating operation rules.

Operation-specific canonicalization and lowering patterns are valid. Their
rules should not be copied into generic passes.

## 4. Fix the issue at the layer that owns it

Fix missing information or incorrect behavior in the layer that owns it. Use
consistent IR contracts across affected passes.

Later passes must preserve correctness guarantees. For example, pooling must
preserve a buffer-reuse restriction.

Document required pass ordering in the
[pipeline pass menu](docs/pipeline_pass_menu.md) and test it when correctness
depends on it. Do not rely on accidental ordering or optional canonicalization
for correctness.

Apply a rewrite only when operation semantics, interface contracts, or
analysis results establish that it is safe. If the required facts are missing,
skip an optional rewrite. Report a diagnostic if a required transformation
cannot proceed safely.

A rewrite pattern or reifier that reports failure must leave the input IR
unchanged.

## 5. Test the general rule

Include tests for:

- Cases where the transformation applies.
- Cases where it must not apply.
- Relevant aliases and views.
- Interactions with later passes.

Use small IR examples to test the rule. For non-obvious rewrites, include a
minimal Before/After IR example in the implementation and keep it updated.
Keep examples and comments focused on IR patterns and invariants, not the
model that exposed the issue. A successful run of one model does not prove
that a transformation is correct in general.

## Questions for code review

- Does the decision follow IR semantics, or depend on frontend assumptions or
  backend policy?
- Can an existing IR interface or analysis provide the required facts?
- Would another frontend or target require changes to the generic algorithm?
- Do later passes preserve the same guarantees?

## References

- [LLVM Coding Standards](https://llvm.org/docs/CodingStandards.html)
- [MLIR Interfaces](https://mlir.llvm.org/docs/Interfaces/)
- [MLIR Canonicalization](https://mlir.llvm.org/docs/Canonicalization/)
