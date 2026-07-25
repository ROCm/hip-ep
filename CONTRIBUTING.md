<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Contributing to hip-ep

Thank you for contributing to hip-ep. We welcome improvements to the compiler, runtime, tests, documentation, and developer tooling. This guide explains how to prepare a change that is easy to review and safe to merge.

hip-ep follows the LLVM project's guidance on:

- [Incremental development](https://llvm.org/docs/DeveloperPolicy.html#incremental-development)
- [AI tool use](https://llvm.org/docs/AIToolPolicy.html)

The sections below describe the project-specific workflow.

## Before you start

- Search [existing issues](https://github.com/ROCm/hip-ep/issues) before opening a new one.
- For a bug, include a minimal reproduction, relevant logs, platform details, and expected versus actual behavior.
- Coordinate substantial work in a GitHub issue or discussion before implementation. This helps contributors avoid duplicate work and gives maintainers a chance to provide early design feedback.
- Changes to public compiler/runtime ABIs, dialect semantics, artifact formats, or pipeline architecture should have a documented design discussion before implementation begins.
- Use the [Windows quick start](docs/quick_start.md) or [Linux quick start](docs/quick_start_linux.md) to prepare a development environment.

## Develop and validate

Keep each change focused on one purpose. Add or update tests at the layer where the behavior is implemented:

- LIT tests for MLIR transformations and lowerings;
- runtime unit tests for GPU-independent ABI and contract logic;
- numeric tests for per-operation GPU-versus-CPU correctness;
- E2E or model tests for full compiler/runtime behavior.

See [test/README.md](test/README.md) for test commands and suite-specific guides.

Update documentation in the same PR when behavior, interfaces, pass ordering, configuration, or architecture changes.

Run the repository checks before requesting review:

```bash
pre-commit run --all-files
```

Do not skip existing pre-commit hooks unless a reviewer explicitly approves it.

## Keep changes focused

Small, focused PRs are easier to review, test, revert, and backport.

- PRs over **1,000 changed lines** or **20 files** are automatically labeled `large-pr`. This is a reviewer signal, not a merge block.
- PRs over **2,000 changed lines** or **30 files** should link a design discussion established before implementation.
- A larger PR should explain why the change cannot be split into independently useful and reviewable steps.

A typical compiler/runtime feature can often be divided into:

1. behavior-preserving helpers or API preparation;
2. the smallest useful end-to-end implementation, optionally behind a flag;
3. caller migration and enablement;
4. cleanup of the old path after the new path is proven.

If you are unsure how to divide a change, ask in an issue before investing in the full implementation.

Maintainers may ask for a PR to be reduced or split when its review cost is disproportionate to the benefit of merging it. This is feedback on the shape of the change, not on the contributor.

## Pull requests

Target the repository's default branch and use a draft PR while the change is still under development. Mark it ready and request review when:

- the implementation is complete for the stated scope;
- relevant tests pass;
- documentation is current;
- the description accurately reflects the diff;
- known limitations and follow-ups are documented.

Link the relevant issue or design discussion when the change required prior coordination.

The [pull-request template](.github/pull_request_template.md) asks for:

| Section | What to include |
|---|---|
| Summary | Always: what changed and its user- or developer-visible effect |
| Why | The problem and design rationale when the choice is not obvious |
| What | Concrete implementation details for non-trivial changes |
| Test plan | What was run and the result, or why testing was not needed |
| Notes for reviewers | Optional: known limitations, follow-ups, important assumptions, or ordering constraints |

Keep the description proportional to the change. A small documentation or mechanical fix may need only a Summary and Test plan. A compiler/runtime design change usually benefits from all sections. Write the title and description so they remain useful in the project history after the PR is merged.

Compiler changes may include a concise before/after IR walkthrough. Performance-sensitive changes should state the hardware, workload, build configuration, and measurement method.

PR descriptions may be prepared with tooling, but the contributor must review them for accuracy and own every claim.

Reviewers are assigned through [CODEOWNERS](.github/CODEOWNERS) where ownership is configured. After addressing feedback, re-request review so the updated PR returns to reviewer queues.

## AI-assisted contributions

AI tools are welcome as development aids. The contributor remains the author and is accountable for the submitted code and text.

When using AI tools:

- review and understand all generated content before requesting human review;
- validate the result with appropriate tests;
- be prepared to explain every design decision without delegating the discussion back to the tool;
- disclose substantial AI assistance in the PR description, including what was assisted and how it was validated;
- use the commit trailers below for AI-assisted commits.

Automated review tools may be used privately, but comments or verdicts must be reviewed and approved by a human before they are posted.

## Commit message trailers

For an AI-assisted commit, add these trailers after a blank line:

```text
Co-Authored-By: <tool or model> <email>
Made-with: <tool name>
```

Use the model or tool that actually contributed to the change, and preserve the capitalization `Co-Authored-By`. For human collaborators, use the standard `Co-Authored-By` trailer for each collaborator.

If a pushed commit is missing attribution, do not rewrite reviewed history solely to add it. Add correct attribution to subsequent commits and note the omission in the PR when appropriate.

## Review process

Work with reviewers to resolve correctness, design, testing, documentation, and maintainability concerns. Address each review thread or explain why no change is needed.

Maintainers may request additional tests, documentation, a smaller scope, or a design discussion before approving a change. Keep the conversation focused on the code, its behavior, and the evidence needed to merge it safely.
