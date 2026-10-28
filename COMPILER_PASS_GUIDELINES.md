<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->

# Guidelines for generic compiler passes

Keep compiler passes generic wherever possible. Use explicit IR information
and shared analyses. Put backend-specific decisions in the backend layer.

## 1. Use IR semantics, not target assumptions

Use operation semantics, types, attributes, interfaces, and analysis results.

Do not assume that:

- `memref.load` or `memref.store` always runs on the host.
- A compute operation always runs on a GPU.
- CPU execution is always synchronous or GPU execution is always asynchronous.
- An earlier operation has finished using a buffer.
- A pattern observed in one model applies to all models.

Check the required facts through an explicit contract or analysis.

## 2. Separate the algorithm from backend policy

Keep each pass focused on one job. Reuse existing passes, analyses, and
interfaces before adding new logic.

A generic pass may get target information through an interface. Keep the
backend logic behind that interface.

For example, a buffer-reuse pass checks whether buffers can safely share
storage. The backend supplies target-specific execution, memory, and runtime
restrictions.

Put target-specific transformations in backend passes.

## 3. Keep operation rules with the operation

- Keep shape rules in the operation's inference or reification implementation.
- Describe aliasing and memory effects through interfaces.
- Let generic passes query these interfaces instead of repeating operation rules.

Operation-specific canonicalization and lowering patterns are valid. Their
rules should not be copied into generic passes.

## 4. Fix the issue at the layer that owns it

Fix missing information or incorrect behavior in the layer that owns it. Use
the same contract in all affected passes.

Later passes must preserve correctness guarantees. For example, pooling must
preserve a buffer-reuse restriction.

Document required pass ordering. Do not rely on accidental ordering or optional
canonicalization for correctness.

Apply a rewrite only when you can prove it is safe.

## 5. Test the general rule

Include tests for:

- Cases where the transformation applies.
- Cases where it must not apply.
- Relevant aliases and views.
- Interactions with later passes.

Use small IR examples to test the rule. A successful run of one model does not
prove that a transformation is correct in general.

## Questions for code review

- Does the decision use IR facts or an unstated target assumption?
- Can an existing interface or analysis provide the information?
- Would another target require changes to the generic algorithm?
- Do later passes preserve the same guarantees?

## References

- [LLVM Coding Standards](https://llvm.org/docs/CodingStandards.html)
- [MLIR Interfaces](https://mlir.llvm.org/docs/Interfaces/)
- [MLIR Canonicalization](https://mlir.llvm.org/docs/Canonicalization/)
