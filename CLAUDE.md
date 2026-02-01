<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# CLAUDE.md

Guidance for Claude Code when working with this repository.

## Project Overview

**MorphiZen**: Hardware-agnostic AI compiler framework for ONNX graph optimization. Serves as ONNX Runtime Execution Provider with pass-based architecture.

**Architecture**: 5 layers - foundation → backend abstraction (MORPHIZEN_ORT_API/111 functions) → graph manipulation/pattern matching → framework core → applications.

## Build System

**CRITICAL**: Non-standard build paths:
- Build: `../../build/$(basename $PWD)` (NOT `./build`)
- Install: `../../local`
- Runtime: `/MTd` via `CMAKE_MSVC_RUNTIME_LIBRARY`

**Configure**:
```bash
cmake -S . -B ../../build/$(basename $PWD) -DBUILD_SHARED_LIBS=OFF \
  "-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded\$<\$<CONFIG:Debug>:Debug>" \
  -DCMAKE_BUILD_TYPE=Debug -DCMAKE_PREFIX_PATH=../../local \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -Dmorphizen_ENABLE_UNIT_TEST=ON --fresh
```

**Build**: `cmake --build ../../build/$(basename $PWD) --config Debug --parallel`

**Test**: `../../build/$(basename $PWD)/bin/morphizen-unit-tests.exe`

**MSVC Setup** (if headers missing):
```bash
cmd /c "call \"\"C:\\msvsn2022\\VC\\Auxiliary\\Build\\vcvars64.bat\"\" && cd /d %CD% && cmake --build \"\"../../build/$(basename $PWD)\"\" --config Debug --parallel"
```

**Key Options** (defaults in cmake/morphizen_options.cmake):
- `morphizen_ENABLE_ORT_BRIDGE=ON` (default)
- `morphizen_ENABLE_MLIR_BACKEND=ON` (default)
- `morphizen_ENABLE_UNIT_TEST=OFF` (default, set to ON for development)
- `morphizen_ENABLE_BOOST=OFF` (default)

## Architecture

**Layers**: Applications → morphizen-core (passes, compilation) → morphizen-graph/pattern → morphizen-ort-api-ext (111 functions) → onnx-ir-imp/mlir-imp/ort-bridge → foundation

**Design Patterns**:
1. **Type-Safe Indices**: 64-bit `NodeIndex`/`NodeArgIndex`, O(1) hashing (`ort-bridge/doc/ORT-BRIDGE-DESIGN.md`)
2. **Backend Abstraction**: 111 function pointers, runtime selection via `MORPHIZEN_ORT_BRIDGE_BACKEND`
3. **Pass System**: Plugin-based via `morphizen_config.json`, implements `IPass`
4. **Pattern Matching**: 12 types (wildcard, node, constant, commutable_node, or, sequence, where, graph_input, graph_output, node_output_arg, pattern_graph_input, pattern_graph_output)

**Key Headers**: `graph.hpp`, `node.hpp`, `node_arg.hpp`, `node_builder.hpp`, `pattern.hpp`, `pass.hpp`, `model.hpp`, `env_config.hpp`

**Directories**: `morphizen-ort-api-ext/`, `morphizen-graph/` (~4K LOC), `morphizen-pattern/` (~2.2K LOC, 12 types), `morphizen-core/`, `onnx-ir-imp/`, `mlir-imp/`, `ort-bridge/`, `unit-test/` (80+ tests)

## Testing

**Framework**: GTest, 80+ tests across 13+ modules
**Run**: `ctest --test-dir ../../build/$(basename $PWD)/unit-test -R TestName --verbose`

## Git Workflow

**Remotes**: `origin` (ROCm/MorphiZen, read-only) | `fork` (your fork, push here)

**CRITICAL**: Always push to `fork`, never `origin`

**Push Policy**: After creating commits, ALWAYS push to fork immediately unless the user says otherwise.

**Auto-PR Policy**: After successfully pushing to fork, IMMEDIATELY check if a PR exists for the branch. If not, create a draft PR automatically with `gh pr create --draft`.

**Required Steps**:
1. Sync: `git checkout main && git pull origin main`
2. Branch: `git checkout -b feature/<name>` (BEFORE changes)
3. Commit: After file changes, BEFORE testing
4. Push: `git push fork <branch>`
5. PR: `gh pr create --draft` (IMMEDIATELY after first push, auto-create if no PR exists)

**CRITICAL - Before Marking PR Ready**:
Before marking PR ready for review, MUST run pre-commit to fix formatting issues:
```bash
pre-commit run --all-files
```
If pre-commit makes changes (formatting, linting), commit and push them BEFORE marking PR ready. This prevents CI pre-commit check failures.

**PR Operations** (fork-based workflow):
- ❌ `gh pr view` (fails - branch not in origin)
- ✅ `gh pr view <number>` (most reliable)
- ✅ `gh pr view <owner>:<branch>` (e.g., `gh pr view your-username:feature/name`)
- ✅ `gh pr list` (to find PR number first)

**Why `gh pr view` fails**: PRs are cross-repository (fork → origin). `gh pr view` searches for the branch in the current repo (origin/ROCm/MorphiZen), but the branch only exists in the fork.

**Commit Rules**:
- ❌ NO AI mentions (Co-Authored-By: Claude, etc.)
- ✅ Conventional commits (`feat:`, `fix:`, `docs:`, etc.)
- ✅ Stage specific files: `git add <file>` (NOT `-A` or `.`)

## Setup

**Pre-commit** (required): `scripts/setup-dev-env.ps1` (Windows) or `scripts/setup-dev-env.sh` (Linux/Mac)

**Backend**: `export MORPHIZEN_ORT_BRIDGE_BACKEND=onnx` (or `mlir`)

## Common Pitfalls

1. Build dir: `../../build/$(basename $PWD)`, NOT `./build`
2. Install prefix: `../../local`
3. Git push: `fork`, NEVER `origin`
4. Never work on `main` branch
5. Run `scripts/setup-dev-env.*` before contributing
6. Launch bash from MSVC Developer Command Prompt (Windows)
7. NO AI/tool mentions in commits/PRs
8. Pre-commit hooks don't enforce when committing via Claude Code - run `pre-commit run --all-files` after commits to verify

## Docs

`docs/architecture.md`, `docs/developer-guide.md`, `docs/workflows/git-workflow.md`, `docs/workflows/build-workflow.md`, `docs/workflows/pr-workflow.md`, `ort-bridge/doc/ORT-BRIDGE-DESIGN.md`

## Project Backlog

**CRITICAL**: When completing an issue, MUST update backlog BEFORE marking PR ready.

**Timing:**
- Update backlog AFTER implementation done (code complete, tests pass)
- Update backlog BEFORE marking PR ready for review
- DO NOT update when starting work (issue file needed during implementation)
- DO NOT update when CI passes (only when implementation done)

**Steps (on feature branch):**
1. Edit backlog.md - move issue to "Recently Completed" with PR number
2. Delete issue file: `git rm docs/project/issues/042-*.md`
3. Commit: `git commit -m "docs: complete issue #042"`
4. Push to fork
5. Mark PR ready for review

This ensures single-PR workflow (code + backlog updates together).

See `docs/project/backlog.md` and `docs/project/CONTRIBUTING.md`.

## Dependencies

**Required**: ONNX Runtime (pre-built, `../../local`)
**Optional**: LLVM/MLIR, Protobuf, GTest, Boost (for tools)
