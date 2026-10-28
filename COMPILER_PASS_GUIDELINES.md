<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->

# Guidelines for keeping compiler passes generic

Keep compiler passes generic wherever possible. Generic passes should use
explicit IR information and shared analyses. Backend-specific decisions belong
in the backend layer.

These guidelines apply to all contributors and reviewers.

## 1. Use IR semantics, not target assumptions

Make decisions from operation semantics, types, attributes, interfaces, and
analysis results.

Do not assume that:

- `memref.load` or `memref.store` always runs on the host.
- A compute operation always runs on a GPU.
- CPU execution is always synchronous or GPU execution is always asynchronous.
- An earlier operation has finished using a buffer.
- A pattern observed in one model applies to all models.

If a transformation requires this information, obtain it from an explicit
contract or analysis.

## 2. Separate the algorithm from backend policy

Keep the general algorithm independent of the target.

For example, a buffer-reuse pass should determine whether buffers can safely
share storage. Backend-specific analysis should provide restrictions caused by
execution order, memory placement, or runtime requirements.

A generic pass may consume target information through a defined interface. It
should not duplicate the backend logic that determines that information.

Put inherently target-specific transformations in clearly identified backend
passes.

## 3. Keep operation rules with the operation

Use existing helpers and MLIR interfaces where they fit.

- Keep shape rules in the operation's inference or reification implementation.
- Describe aliasing and memory effects through the appropriate interfaces.
- Let generic passes query those interfaces.
- Avoid maintaining separate operation-specific rules in multiple passes.

Operation-specific canonicalization and lowering patterns are valid. Their
semantics should not be duplicated inside otherwise generic passes.

Reuse existing mechanisms before introducing a new interface or framework.

## 4. Fix the issue at the layer that owns it

Identify where information is missing or behavior becomes incorrect. Correct
that layer and establish one contract for all affected passes.

A correctness guarantee must survive the rest of the pipeline. For example, a
buffer-reuse restriction must remain effective through later pooling.

Document required pass ordering. Do not depend on incidental ordering or
optional canonicalization for correctness.

If an optimization cannot prove that a rewrite is safe, it must not apply the
rewrite.

## 5. Test the general rule

Include tests for:

- Cases where the transformation applies.
- Cases where it must not apply.
- Relevant aliases and views.
- Interactions with later passes.

Use small IR examples that demonstrate the rule. A successful run of one model
does not establish that a transformation is generally correct.

## Questions for code review

- Does the decision follow from the IR or from an unstated target assumption?
- Can an existing interface or shared analysis provide the information?
- Would supporting another target require changes to this generic algorithm?
- Do later passes preserve the same correctness guarantees?

## References

- [LLVM Coding Standards](https://llvm.org/docs/CodingStandards.html)
- [MLIR Interfaces](https://mlir.llvm.org/docs/Interfaces/)
- [MLIR Canonicalization](https://mlir.llvm.org/docs/Canonicalization/)
