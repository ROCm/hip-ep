<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Project Backlog

Issue tracking: `backlog.md` (index) + `issues/NNN-name.md` (detailed files)

Active work tracked in GitHub PRs: `gh pr list --repo ROCm/hip-compiler`

See [CONTRIBUTING.md](CONTRIBUTING.md) for issue quality guidelines.

---

## Backlog

| # | Title | Pri | Est | Group | Blocked | Files |
|---|-------|-----|-----|-------|---------|-------|
| #017 | Route model.dll output through FileSystem; rename `--constants-dir` to `--output-dir` | H | 2h | Compiler/CLI/Tests | - | `CompilerDriver.cpp`, `hip-compile.cpp`, `hip-opt.cpp`, `CMakeLists.txt`, `lit/e2e/*.mlir`, `lit.site.cfg.py.in`, `lit.cfg.py` |

**Quick dependencies:**
- **#017 relates to #014** — both touch constant handling architecture

**Legend:**
- **Pri**: Priority (C=Critical, H=High, M=Medium, L=Low)
- **Est**: Estimated time (15m=15 minutes, 1h=1 hour, etc.)
- **Group**: Feature area grouping
- **Blocked**: Issue numbers that must complete first (`-` if none)
- **Files**: Key files affected (see issue file for complete list)

---

## Completed Issues

See **[completed-issues.md](completed-issues.md)** for full archive of all completed issues.

**Recent (last 5):**
- #016 — Fix OnnxToHip LIT tests — systematic coverage, no overlap (PR #52, 2026-03-04)
- #015 — Remove unused `hipdnn.constant_count` module attribute (PR #52, 2026-03-04)
- #014 — Fix inaccurate `CONSTANT-HANDLING-DESIGN.md` (PR #52, 2026-03-04)
- #013 — Remove redundant `hipdnn.input_ranks` / `hipdnn.output_ranks` module attributes (PR #52, 2026-03-04)
- #012 — Update `docs/design/` for redesigned pipeline (PR #33, 2026-03-03)
