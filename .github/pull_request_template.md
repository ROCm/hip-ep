<!--
Thanks for contributing! Please skim the two policies this repo
follows before requesting review (1-2 minutes each):

  Incremental development:
    https://llvm.org/docs/DeveloperPolicy.html#incremental-development
  AI tool use:
    https://llvm.org/docs/AIToolPolicy.html

A short summary lives in CONTRIBUTING.md at the repo root. Each
section below has an HTML comment explaining what to put in it; you
can delete the comment after filling the section in.
-->

## Summary

<!--
1-3 sentences. What changed at a high level and what gap or bug it
closes.
-->

## Why

<!--
Design rationale. What alternatives did you consider and why did you
pick this one? This section is what reviewers use to skip ahead and
agree with the design before reading the code. Even one paragraph is
fine if the choice is obvious; for non-trivial design decisions, be
explicit about the alternatives you rejected.
-->

## What

<!--
Numbered list of concrete changes, with file references. Useful
shapes:

1. **New pass / op / runtime helper** at `path/to/file.cpp`. <one
   sentence on the contract; one sentence on the predicate or scope>.
2. **Pipeline placement / call-site update**: <where it slots in and
   why that placement>.
3. **Helper rename / API change**: <one sentence>.
4. **Intentional non-changes**: <things the reader will expect to see
   touched that aren't, with the reason>.
-->

## Test plan

<!--
- Lit / pytest / runner commands you ran and the result (pass count,
  numerics delta vs reference, etc).
- Manual reproduction steps for behavioural changes (model name,
  hardware, command).
- For perf-sensitive changes, include a before/after table — see the
  optional Performance section below.
-->

## Notes for reviewers

<!--
- Known limitations and what they would take to fix.
- Follow-ups already planned (link the design issue / next PR).
- Anything load-bearing that isn't obvious from the diff (invariants,
  ABI assumptions, ordering constraints).
- "None" is a valid answer if the PR is small and self-contained.
-->

<!--
============================================================
Optional sections — include when applicable. Delete this whole
block if you don't need any of them.
============================================================

## Compiler IR walkthrough

<details>
<summary><b>Before</b> — short description</summary>

```mlir
// IR before this PR's pass / lowering
```

</details>

<details>
<summary><b>After</b> — short description</summary>

```mlir
// IR after this PR's pass / lowering
```

</details>

## Performance

<details>
<summary>Short headline result (e.g. "22x over DML EP on …")</summary>

| Metric | This PR | Baseline | Ratio |
|---|---:|---:|---:|
| Inferences/sec | … | … | … |
| Mean latency | … | … | … |

Hardware: gfxNNNN, model: <name>, build: <flags>.

</details>

============================================================
-->

## Checklist

- [ ] **Incremental.** This PR is either standalone, or a planned step
      in a documented series. If it exceeds ~500 LOC or ~10 files, the
      Summary / Why explains why it cannot be split, OR links the
      precursor PRs and the design issue. See:
      https://llvm.org/docs/DeveloperPolicy.html#incremental-development
- [ ] **AI policy.** If AI tools (Cursor, Claude, Copilot, etc.)
      generated substantial code or text, the Summary discloses it,
      the commit messages carry the trailers documented in
      [`CONTRIBUTING.md`](../blob/main/CONTRIBUTING.md#commit-message-trailers),
      and you can defend each design decision in review without
      re-prompting an LLM. See: https://llvm.org/docs/AIToolPolicy.html
- [ ] **No `good first issue` automation.** This PR is not an AI-tool
      fix to an issue tagged `good first issue` (those are reserved as
      learning opportunities for new contributors).
- [ ] **Reviews resolved.** All review-comment threads from prior
      rounds are resolved (enforced by branch protection on `main`).
- [ ] **CODEOWNERS routing.** Reviewers were assigned automatically by
      [`CODEOWNERS`](../blob/main/.github/CODEOWNERS). If the change
      touches a path outside the current ownership map, the PR
      description names the operational owner.
