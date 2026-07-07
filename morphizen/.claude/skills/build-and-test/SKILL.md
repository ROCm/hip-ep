<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->

---
name: build-and-test
description: Build and test MorphiZen with automated dependency management
allowed-tools: [Bash, Read, Grep]
---

# Build and Test MorphiZen

Automates the build/test workflow with dependency checking and error remediation.

## Instructions for Claude

### Step 1: Run Build Script

Execute `build-and-test.sh`. Script handles MSVC check, dependency auto-build, build type detection, and testing.

Parse status from stdout. If not `STATUS:SUCCESS`, proceed to Step 2.

### Step 2: Error Remediation

**MISSING_MSVC:** Direct user to launch bash from "Developer Command Prompt for VS 20XX".

**MISSING_ONNXRUNTIME:** Direct user to `docs/developer-guide.md` for VitisAI build instructions.

**BUILD_FAILED:** Show error with file:line, read affected file for context. If fixing code: verify feature branch, commit, push to fork.

**TEST_FAILED:** Show failed test names and output. Suggest individual test run for debugging.
