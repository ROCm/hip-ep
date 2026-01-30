<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->

# Pre-commit Setup and Troubleshooting

This guide explains how to set up pre-commit hooks and troubleshoot common issues.

## Quick Setup

### Automated Setup (Recommended)

Run the setup script from the project root:

**Windows (PowerShell):**
```powershell
scripts/setup-dev-env.ps1
```

**Linux/Mac:**
```bash
scripts/setup-dev-env.sh
```

This automatically installs pre-commit and sets up git hooks.

### Manual Setup

```bash
# Install pre-commit
pip install pre-commit

# Install git hooks (run from project root)
pre-commit install
```

## How Pre-commit Works

Pre-commit manages all code formatting and linting tools automatically:

1. **Tool Versions**: `.pre-commit-config.yaml` pins exact versions (e.g., clang-format 16.0.1, lintrunner 0.12.7)
2. **Isolated Environment**: Pre-commit creates a Python virtual environment with these exact versions
3. **Automatic Execution**: Tools run automatically on `git commit` or via `pre-commit run`
4. **Version Consistency**: Your local formatting always matches CI because both use the same pinned versions

## IMPORTANT: Never Install Tools Manually

**DO NOT** install or run these tools manually:
- ❌ `pip install clang-format`
- ❌ `clang-format -i file.cpp`
- ❌ `pip install lintrunner`

**ALWAYS** use pre-commit:
- ✅ `pre-commit run --all-files`
- ✅ `pre-commit run` (on staged files)
- ✅ `git commit` (runs automatically)

### Why Manual Installation Causes Problems

**Example of version mismatch:**

If you install clang-format manually, you might get version 19.x, which formats code differently than version 16.0.1 used by CI:

```cpp
// Your local clang-format 19.x produces:
auto env =
    std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_ERROR, "morphizen_unit_test");

// CI's clang-format 16.0.1 expects:
auto env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_ERROR,
                                      "morphizen_unit_test");
```

**Result**: Your PR fails CI checks even though the code "looks formatted."

**Solution**: Let pre-commit manage all tools. It uses the exact versions defined in `.pre-commit-config.yaml`.

## Usage

### Automatic (Recommended)

Pre-commit runs automatically when you commit:

```bash
git add file.cpp
git commit -m "feat: add new feature"
# Pre-commit runs automatically and formats your code
```

### Manual

Format all files:
```bash
pre-commit run --all-files
```

Format only staged files:
```bash
pre-commit run
```

Update hook versions (maintainers only):
```bash
pre-commit autoupdate
```

## Troubleshooting

### Pre-commit Checks Fail in CI But Pass Locally

**Diagnosis**: Your pre-commit hooks may not be installed or are outdated.

**Solution**:
```bash
# Clean and reinstall hooks
pre-commit clean
pre-commit install

# Run on all files to verify
pre-commit run --all-files
```

### Hooks Don't Run on Commit

**Diagnosis**: Git hooks not installed.

**Solution**:
```bash
# Check if hook exists
ls -la .git/hooks/pre-commit

# If missing, install hooks
pre-commit install
```

### "pre-commit: command not found"

**Solution**:
```bash
# Install pre-commit
pip install pre-commit

# Verify installation
pre-commit --version
```

### Lintrunner Errors

**Error**: `ModuleNotFoundError: No module named 'lintrunner'`

**Solution**: Pre-commit will install lintrunner automatically in its isolated environment. If you see this error:
```bash
pre-commit clean
pre-commit run --all-files
```

### Want to Skip Pre-commit (Not Recommended)

To temporarily bypass hooks:
```bash
git commit --no-verify
```

**Warning**: This will likely cause CI to fail. Only use if you know what you're doing.

## CI Configuration

The CI uses these pinned versions (from `.pre-commit-config.yaml`):
- `clang-format==16.0.1`
- `lintrunner==0.12.7`
- `lintrunner-adapters==0.12.4`
- `ruff==0.11.9`

Pre-commit automatically uses these same versions locally, ensuring consistency.

## For Maintainers: Updating Tool Versions

If you need to update clang-format or other tool versions:

1. Update `.pre-commit-config.yaml`:
   ```yaml
   additional_dependencies:
     - clang-format==<NEW_VERSION>
   ```

2. Update all hooks:
   ```bash
   pre-commit autoupdate
   ```

3. Reformat entire codebase:
   ```bash
   pre-commit run --all-files
   ```

4. Create PR with reformatted code

5. Announce version change to team

## Common Scenarios

### New Contributor Setup

```bash
# 1. Clone repository
git clone ../MorphiZen
cd MorphiZen

# 2. Run setup script
scripts/setup-dev-env.sh  # or .ps1 on Windows

# 3. Make changes
# ... edit files ...

# 4. Commit (pre-commit runs automatically)
git add -u
git commit -m "feat: my changes"
```

### Fixing Formatting Issues

```bash
# Run pre-commit to fix formatting
pre-commit run --all-files

# Commit the fixes
git add -u
git commit -m "style: apply pre-commit formatting"
```

### IDE Integration (Optional)

Some developers want IDE integration for real-time formatting. This is **optional** and not required.

**VSCode Example**:
1. Install "pre-commit" extension
2. It will use `.pre-commit-config.yaml` automatically
3. Or configure VSCode to run `pre-commit run` on save

**Important**: Even with IDE integration, let pre-commit manage the tools. Don't install clang-format separately.

## Related Documentation

- [Developer Guide - Pre-commit Hooks](developer-guide.md#6-pre-commit-hooks-required-for-contributors)
- [PR Workflow](workflows/pr-workflow.md)
- [Git Workflow](workflows/git-workflow.md)
