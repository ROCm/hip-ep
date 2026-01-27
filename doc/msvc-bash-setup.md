<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->

# MSVC Environment Setup for Bash

This document describes how to use the MSVC environment setup scripts to build MorphiZen in Git Bash on Windows.

## Quick Start

### For Claude Code (Automated)

Claude automatically sources the MSVC environment when running build commands:

```bash
# Claude's first command - sources environment
source tools/setup_msvc_env_bash.sh && cmake --preset "Morphizen Ninja"

# All subsequent commands automatically have MSVC environment
cmake --build "C:/Develop/m/build/morphizen.ninja" --config Debug --parallel
```

**OR** use the wrapper scripts (auto-setup):

```bash
# Wrappers automatically source MSVC environment if needed
tools/cmake-msvc --preset "Morphizen Ninja"
tools/build-msvc "C:/Develop/m/build/morphizen.ninja" --config Debug --parallel
```

### For Manual Use in Git Bash

Run ONCE at the start of your terminal session:

```bash
cd /c/Develop/m/Source/Morphizen
source tools/setup_msvc_env_bash.sh

# Now run any commands - environment persists for entire session
cl.exe /?
cmake --preset "Morphizen Ninja"
cmake --build "C:/Develop/m/build/morphizen.ninja" --config Debug
```

## How It Works

### The Problem

- MSVC compiler (`cl.exe`) requires environment variables set by `vcvars64.bat`
- `vcvars64.bat` only works in `cmd.exe`, not bash
- CMake configuration fails with "No CMAKE_C_COMPILER could be found"

### The Solution

`tools/setup_msvc_env_bash.sh` bridges the MSVC environment to bash:

1. **Locates Visual Studio** using `vswhere.exe`
2. **Executes vcvars64.bat** in a `cmd.exe` subprocess
3. **Captures environment variables** from cmd.exe using `set` command
4. **Converts PATH** from Windows format to Unix format
5. **Exports variables** to bash environment
6. **Sets up VAI_RT variables** (workspace, build dir, install prefix)

### Environment Variables Set

The script exports these critical MSVC variables:

- **PATH** - Compiler and tool paths (converted to Unix format)
- **INCLUDE** - C/C++ header search paths
- **LIB** - Library search paths
- **LIBPATH** - .NET and C++ library paths
- **VSINSTALLDIR** - Visual Studio installation directory
- **VCINSTALLDIR** - Visual C++ installation directory
- **VCToolsInstallDir** - VC++ tools directory
- **WindowsSDKVersion** - Windows SDK version
- **WindowsSdkDir** - Windows SDK directory

Plus VAI Runtime variables:

- **VAI_RT_WORKSPACE** - Root workspace (e.g., `C:/Develop/m/source`)
- **VAI_RT_BUILD_DIR** - Build directory (e.g., `C:/Develop/m/build`)
- **VAI_RT_PREFIX** - Install prefix (e.g., `C:/Develop/m/local`)

## Usage Patterns

### Pattern 1: Source Once Per Session (Recommended)

```bash
# First command in session
source tools/setup_msvc_env_bash.sh

# Environment persists - run multiple commands
cmake --preset "Morphizen Ninja"
cmake --build "C:/Develop/m/build/morphizen.ninja" --config Debug
ctest --test-dir "C:/Develop/m/build/morphizen.ninja/unit-test"
```

### Pattern 2: Chain with First Command

```bash
# Source and run in one command (for Claude Code)
source tools/setup_msvc_env_bash.sh && cmake --preset "Morphizen Ninja"
```

### Pattern 3: Use Wrapper Scripts (Foolproof)

```bash
# Wrappers check and auto-source environment if needed
tools/cmake-msvc --preset "Morphizen Ninja"
tools/build-msvc "C:/Develop/m/build/morphizen.ninja" --config Debug
```

### Pattern 4: Project .bashrc

```bash
# Source project .bashrc once
source .bashrc

# Environment is ready
cmake --preset "Morphizen Ninja"
```

## Verification

After sourcing the script, verify the environment:

```bash
# Check compiler is available
which cl.exe
cl.exe /?

# Check environment variables
echo $VCINSTALLDIR
echo $INCLUDE
echo $LIB

# Check VAI_RT variables
echo $VAI_RT_WORKSPACE
echo $VAI_RT_BUILD_DIR
echo $VAI_RT_PREFIX
```

## Troubleshooting

### Error: "vswhere.exe not found"

**Cause**: Visual Studio is not installed or not in the default location.

**Solution**: Install Visual Studio 2022 with "Desktop development with C++" workload.

### Error: "Visual Studio installation not found"

**Cause**: Visual Studio is installed but missing C++ tools.

**Solution**: Run Visual Studio Installer and add "Desktop development with C++" workload.

### Error: "cl.exe not found in PATH"

**Cause**: Environment setup failed or PATH conversion has issues.

**Solution**:
1. Check if `vcvars64.bat` exists at the expected location
2. Try running the script again with verbose output
3. Manually verify vcvars64.bat works: `cmd /c "vcvars64.bat && cl.exe /?"`

### Error: "CMAKE_C_COMPILER could not be found"

**Cause**: MSVC environment was not sourced before running CMake.

**Solution**: Source the environment first:
```bash
source tools/setup_msvc_env_bash.sh && cmake --preset "Morphizen Ninja"
```

Or use the wrapper:
```bash
tools/cmake-msvc --preset "Morphizen Ninja"
```

### Environment Variables Not Set After Sourcing

**Cause**: Script was executed instead of sourced.

**Wrong**:
```bash
./tools/setup_msvc_env_bash.sh  # Executes in subshell, doesn't affect current shell
```

**Correct**:
```bash
source tools/setup_msvc_env_bash.sh  # Sources into current shell
```

## Integration with Build Workflow

### Build Script Check

You can add environment checks to build scripts:

```bash
#!/usr/bin/env bash

# Check if MSVC environment is set
if [[ -z "${VCINSTALLDIR:-}" ]]; then
    echo "ERROR: MSVC environment not set."
    echo "Run: source tools/setup_msvc_env_bash.sh"
    exit 1
fi

# Proceed with build
cmake --preset "Morphizen Ninja"
```

### CMakePresets.json

The MSVC environment variables are required for the CMake presets defined in `CMakePresets.json`:

```json
{
  "name": "Morphizen Ninja",
  "generator": "Ninja",
  "cacheVariables": {
    "CMAKE_C_COMPILER": "cl.exe",
    "CMAKE_CXX_COMPILER": "cl.exe"
  }
}
```

CMake will find `cl.exe` in PATH after sourcing the MSVC environment.

## Files

- **tools/setup_msvc_env_bash.sh** - Main MSVC environment setup script
- **tools/cmake-msvc** - CMake wrapper with auto-setup
- **tools/build-msvc** - Build wrapper with auto-setup
- **doc/msvc-bash-setup.md** - This documentation

## See Also

- `.clinerules/workflows/build-and-fix-build-errors.md` - Build workflow
- `tools/setup_msvc_env.ps1` - PowerShell version for reference
- `tools/env.sh` - VAI_RT variable setup (original)
