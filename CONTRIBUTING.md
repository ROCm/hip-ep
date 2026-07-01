<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->

# Contributing to onnx-hipdnn-ep

This repo follows two contribution policies adopted from the LLVM
project, with small adaptations noted below. Every PR is expected to
honour both. The list will grow over time; for now this is the minimum
bar for opening a PR on `main`.

- [Incremental Development](https://llvm.org/docs/DeveloperPolicy.html#incremental-development)
- [AI Tool Use Policy](https://llvm.org/docs/AIToolPolicy.html)

If you have not contributed here before, please skim the two LLVM pages
linked above before opening a PR. They are short. The rest of this
document explains how those policies translate into concrete
expectations for this repo.

---

## TL;DR

1. **Keep PRs small.** Large feature work should be decomposed into a
   series of independent PRs (precursors first, then the main change).
   ~1000 LOC / ~20 files is the soft threshold; bigger PRs need the
   Summary section to explain why they cannot be split.
2. **Human in the loop for AI-generated content.** You can use Cursor,
   Claude, Copilot, or whatever helps you write code, but you are the
   author and are accountable for the change. You must be able to
   answer review questions about the work without delegating back to
   the tool.
3. **Disclose substantial AI usage.** Note it in the PR description
   and use the commit message trailers documented in
   [Commit message trailers](#commit-message-trailers) below.
4. **Don't let AI tools fix `good first issue`s.** Those are reserved
   as learning opportunities for new contributors.
5. **Address every review comment.** Branch protection on `main`
   requires every review-comment thread to be resolved before merge.

---

## Incremental development

The full LLVM rationale is at
<https://llvm.org/docs/DeveloperPolicy.html#incremental-development>.
The points that bite hardest in this repo:

> - Each change should be kept as small as possible. This simplifies
>   your work (into a logical progression), simplifies code review and
>   reduces the chance that you will get negative feedback on the
>   change. ...
> - Large/invasive changes usually have a number of secondary changes
>   that are required before the big change can be made (e.g. API
>   cleanup, etc). These sorts of changes can often be done before the
>   major change is done, independently of that work.
> - Often, an independent precursor to a big change is to add a new API
>   and slowly migrate clients to use the new API. ...

### What "small" means here

- **~1000 changed LOC** or **~20 changed files** (whichever is hit
  first) is the soft threshold. PRs above this get the `large-pr`
  label automatically as a reviewer signal — not a hard block.
- A `large-pr` PR is acceptable when the work genuinely cannot be
  split. **Listing a category below is not a free pass** — the
  Summary / Why section must explain *why this specific change*
  couldn't be sliced further, naming the parts you considered
  extracting (precursor refactors, build/CI bumps, isolated test
  fixtures, separable increments) and why each one would not be
  reviewable on its own. Categories that have carried justified
  `large-pr` size on this repo:
  - a feature with irreducible cross-layer coupling (compiler +
    runtime + tests that are only meaningful together);
  - a coordinated rename or signature change across many call sites
    where splitting would leave the tree uncompilable between PRs;
  - a single test that exercises many paths.
  Reviewers retain the right to apply the `extractive` label even
  when a category applies, if the slicing argument doesn't hold up.
- **Hard ceiling.** PRs above 2,000 LOC or 30 files require a
  pre-PR design discussion linked from the description — a GitHub
  issue, a sync-meeting agenda item, or a Teams thread with the
  affected reviewers. At that size, "explain why in the PR
  description" stops being a useful review tool; the design needs
  to be agreed on before code is written.
- A `large-pr` PR that *could* have been split but wasn't may be
  marked `extractive` and asked to land as a series. See the
  [Extractive contributions](#extractive-contributions) section
  below.

### How to split

A typical decomposition for a feature that touches the compiler and
the runtime (the most common shape in this repo):

1. **Precursor PRs** — refactors that don't change behaviour but make
   the main change reviewable: factor a helper, rename a symbol, add
   a TableGen option, introduce a runtime export with an old-symbol
   fallback. Each lands on its own. Tag them with the
   `incremental-precursor` label.
2. **First increment** — the smallest end-to-end change that does
   something useful. Often "add the new pass / op behind a flag,
   default off". Lands once tests cover it.
3. **Subsequent increments** — flip the flag, migrate callers, remove
   the old path. Each is a separate PR.

If you are unsure how to slice a change, open a design issue first
and tag a reviewer for input before writing code. This is much
cheaper than discovering during PR review that the structure is
wrong.

---

## AI tool use

The full LLVM policy is at <https://llvm.org/docs/AIToolPolicy.html>.
Five points are load-bearing for this repo:

### Human in the loop

> Contributors must read and review all LLM-generated code or text
> before they ask other project members to review it. The contributor
> is always the author and is fully accountable for their
> contributions.

You should be sufficiently confident in the change that asking a
maintainer to review it is a good use of their time. Concretely:
during review, you must be able to answer questions about each
design decision without delegating back to the tool. If you cannot,
the PR is not ready.

### Disclose substantial AI assistance

PR-level disclosure goes in the Summary section: a short note on which
parts of the change were AI-assisted and how the output was validated.
Commit-level disclosure goes via the trailers in
[Commit message trailers](#commit-message-trailers) below.

The threshold for "substantial" is judgment, but a useful rule of
thumb: if a reviewer would reasonably want to know that an LLM
produced this code in order to ask sharper questions about it,
disclose it.

### Write your own PR description

Write the PR description yourself. Tools for translation,
spell-check, or copy-editing are fine, but letting an LLM generate
the entire description usually produces a longer, less-grounded
text that doesn't match the actual code.

### No automated reviews

Following LLVM's stance, **automated review tools that publish
comments without human approval are not allowed** in this repo. An
opt-in tool that routes through a human reviewer is fine. Posting an
LLM's review verdict directly on a PR is not.

### `good first issue` is off-limits for AI

Issues tagged `good first issue` are explicitly reserved as learning
opportunities for new contributors. Per LLVM's policy, fully
automating those PRs squanders the learning opportunity and adds
little value to the project. **Do not use AI tools to fix issues
labelled `good first issue`.**

---

## Extractive contributions

LLVM defines an *extractive contribution* as one where the cost of
reviewing it is greater than the project's benefit from merging it.
In practice, the patterns we see most often:

- A 5,000+ LOC PR touching the compiler, runtime, tests, and docs,
  where the reviewer must hold the whole change in their head.
- A "fixes everything" PR that bundles unrelated cleanups with a
  feature change.
- A PR that lands the output of an AI tool without the contributor
  understanding the design enough to defend it in review.

If a maintainer judges that a PR fits this pattern, they will:

1. Apply the `extractive` label.
2. Comment with the LLVM-standard request:

   > This PR doesn't appear to comply with our policy on
   > tool-generated content, and requires additional justification
   > for why it is valuable enough to the project for us to review
   > it. Please see our developer policy on AI-generated
   > contributions: http://llvm.org/docs/AIToolPolicy.html

3. Ask the contributor to either reduce the scope, split it into a
   series, or document why review cost is justified.

This is not a personal judgement on the contributor — it is a
statement that *the shape of the patch* is not a good fit for the
project's review capacity right now. The fix is almost always to
split.

---

## PR description template

Every PR is auto-populated with
[`.github/pull_request_template.md`](.github/pull_request_template.md).
The expected sections are:

| Section | Always required? | What goes here |
|---|---|---|
| **Summary** | Yes | 1-3 sentences: what changed and what gap it closes. |
| **Why** | Yes | Design rationale. Alternatives considered and why this one wins. |
| **What** | Yes for non-trivial PRs | Numbered list of concrete changes with file references. |
| **Test plan** | Yes | What you ran and the result. Numerics / TPS deltas where relevant. |
| **Notes for reviewers** | Yes (use "None" if truly nothing) | Known limits, follow-ups, load-bearing invariants. |
| **Compiler IR walkthrough** | Optional | For compiler-side changes — wrap before/after IR in a `<details>` block to keep the description scannable. |
| **Performance** | Optional | For perf-sensitive changes — table comparing this PR vs baseline; mention hardware (gfx target, model, build flags). |

You don't need every section on every PR. Small, single-purpose fixes
can keep the description tight (Summary / Why / What / Test plan /
Notes) and skip the optional sections entirely.

---

## Commit message trailers

Every commit you author should carry these trailers in the body,
separated from the message body by a blank line:

```
Co-Authored-By: <Tool or Person> <email>
Made-with: <tool name>
```

Examples used today on this repo:

```
Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
Made-with: Cursor
```

```
Co-Authored-By: Cursor <cursoragent@cursor.com>
```

Rules:

- Capitalisation is exactly `Co-Authored-By` (not `Co-authored-by`)
  for AI tool attribution. Both forms are accepted by GitHub for
  display purposes, but match the existing convention for grep-ability.
- The model name in the trailer should match the model that actually
  did the work (e.g. `Claude Opus 4.7`, `Claude Sonnet 4.5`,
  `GPT-5`, `Cursor`).
- Add the trailer on **every** commit you author, including test,
  perf-attribution, and revert commits.
- For multi-author work between humans, use the standard
  `Co-Authored-By:` form for each collaborator.

If a commit you authored does not carry the trailer, you may amend
locally before push; if it has already been pushed and reviewed, do
not force-push to amend — leave it and add the trailer on the next
commit.

---

## Plugin extension API

This repo ships a plugin extension API so a down-stream team can add ONNX
ops, custom kernels, and MLIR passes on top of `onnx-hipdnn-ep` without
forking the public repo.

The design and downstream-usage guide lives at
[`docs/design/plugin-interface.md`](docs/design/plugin-interface.md); the
practical authoring guide is
[`docs/plugin_authoring.md`](docs/plugin_authoring.md). Three points worth
knowing as a contributor:

1. **Plugins are linked STATICALLY into the host**, selected at configure time
   with `-DHIPDNN_EP_COMPILER_PLUGINS=<id>` (and, for an out-of-tree plugin repo,
   `-DHIPDNN_EP_COMPILER_PLUGIN_PATHS=<dir>`). The machinery is in
   [`cmake/HipEpPlugins.cmake`](cmake/HipEpPlugins.cmake); the public surface sits
   at [`include/hip/Compiler/PluginAPI.h`](include/hip/Compiler/PluginAPI.h)
   and [`include/hip/Compiler/PluginRegistry.h`](include/hip/Compiler/PluginRegistry.h).
   A working sample plugin lives at `test/plugin/sample_plugin/` and is exercised
   by [`test/plugin/test_static_plugins.cpp`](test/plugin/test_static_plugins.cpp).
   The sample exercises `registerPass<>`, `requestPipelineSlot`,
   `addRuntimeBitcode`, `addLibraryPath`, `addLibrary`, and
   `addDialectRegistration` (a custom dialect + op), documented in the design
   doc's "Custom ops" section.
2. **Improvements that need not live out of tree go upstream, not into a
   plugin.** A down-stream team may use the plugin for any reason, but a
   generic op, bug fix, or non-target-specific pass is still best contributed
   upstream through the normal flow. See the design doc for the
   plugin-vs-upstream guidance.
3. **Editing `HipEpPluginRegistry` is editing a public surface.** Adding a new
   capability requires (a) appending an entry to the `VTable` struct in
   `PluginRegistry.h`, (b) implementing it in
   `lib/Compiler/PluginRegistry.cpp`, (c) wiring at most one host site to read
   the recorded state, and (d) a unit-test assertion that exercises the
   round-trip. Do **not** remove or reorder existing `VTable` entries; that is a
   breaking change that bumps `HIP_EP_PLUGIN_API_VERSION`.

Because plugins are statically linked, a contributed MLIR pass shares the host's
one MLIR instance by construction (no symbol export, no dynamic loading) and an
ABI mismatch is a build error, not a load-time surprise -- see the "Linkage
model" in the design doc.

---

## Other repo conventions

- **Branch protection on `main`:** PR + 1 CODEOWNER approval, all
  threads resolved, no force-push, no delete. Admins can bypass for
  emergencies.
- **CODEOWNERS:** see [`.github/CODEOWNERS`](.github/CODEOWNERS).
  Every path is co-owned by at least three reviewers, so a single
  person being unreachable does not block routing.
- **Code style:** clang-format / black / lit-test conventions are
  enforced by pre-commit. Run `pre-commit install` once per clone.
- **Pre-existing pre-commit hooks must not be skipped** (no
  `--no-verify`) unless explicitly approved by a reviewer.

---

## Labels

Reviewers may apply these labels to give shared vocabulary:

| Label | Meaning |
|---|---|
| `extractive` | Review cost > project benefit. Needs to be split or made smaller. |
| `large-pr` | Auto-applied when a PR exceeds the soft size thresholds. Reviewer signal, not a block. |
| `incremental-precursor` | Standalone refactor that unblocks a larger upcoming change. |
| `ai-assisted` | Substantial AI-generated content disclosed in the PR. |
| `breaking-change` | Potentially-breaking change to a public API or runtime ABI. Surface in release notes. |

This list will grow as additional policies (release-notes discipline,
breaking-change handling, RFC process) are added on top of the two
starting policies above.
