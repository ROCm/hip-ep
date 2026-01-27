#!/usr/bin/env bash
##
## Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
## Licensed under the MIT License.
##

# MSVC Environment Setup for Bash
# This script bridges the MSVC environment from vcvars64.bat to bash.
# It can be sourced to set up the current shell or executed with --shell to launch a subshell.

set -euo pipefail

# Color codes for output
readonly COLOR_RED='\033[0;31m'
readonly COLOR_GREEN='\033[0;32m'
readonly COLOR_YELLOW='\033[1;33m'
readonly COLOR_RESET='\033[0m'

# Critical MSVC environment variables we need to export
readonly MSVC_VARS=(
    "PATH"
    "INCLUDE"
    "LIB"
    "LIBPATH"
    "VSINSTALLDIR"
    "VCINSTALLDIR"
    "VCToolsInstallDir"
    "WindowsSDKVersion"
    "WindowsSdkDir"
    "WindowsLibPath"
    "WindowsSDKLibVersion"
    "UniversalCRTSdkDir"
    "UCRTVersion"
    "VCIDEInstallDir"
    "VSCMD_ARG_app_plat"
    "VSCMD_ARG_HOST_ARCH"
    "VSCMD_ARG_TGT_ARCH"
    "VisualStudioVersion"
    "Platform"
)

# Find vcvars64.bat using vswhere.exe
find_vcvars() {
    local vswhere="/c/Program Files (x86)/Microsoft Visual Studio/Installer/vswhere.exe"

    if [[ ! -f "$vswhere" ]]; then
        echo -e "${COLOR_RED}ERROR: vswhere.exe not found at: $vswhere${COLOR_RESET}" >&2
        echo "Please install Visual Studio 2022 with C++ tools" >&2
        return 1
    fi

    # Query for Visual Studio installation with C++ tools
    local install_path=$("$vswhere" -latest -products '*' \
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 \
        -property installationPath 2>/dev/null | tr -d '\r')

    if [[ -z "$install_path" ]]; then
        echo -e "${COLOR_RED}ERROR: Visual Studio installation not found${COLOR_RESET}" >&2
        echo "Please install Visual Studio 2022 with 'Desktop development with C++' workload" >&2
        return 1
    fi

    # Convert Unix path to Windows path for vcvars64.bat
    local vcvars_path="${install_path}/VC/Auxiliary/Build/vcvars64.bat"

    # Convert to Windows-style path for cmd.exe
    local vcvars_win_path=$(echo "$vcvars_path" | sed 's|^/\([a-z]\)/|\1:/|' | sed 's|/|\\|g')

    if [[ ! -f "$vcvars_path" ]]; then
        echo -e "${COLOR_RED}ERROR: vcvars64.bat not found at: $vcvars_path${COLOR_RESET}" >&2
        return 1
    fi

    echo "$vcvars_win_path"
}

# Capture MSVC environment using PowerShell
# PowerShell is more reliable than cmd.exe when called from Git Bash
capture_msvc_env() {
    local vcvars_bat="$1"
    local temp_env=$(mktemp)

    # Use PowerShell to execute vcvars64.bat and capture environment
    # PowerShell can call cmd.exe /c more reliably than Git Bash can
    local ps_script="cmd /s /c \"\`\"$vcvars_bat\`\" && set\" | ForEach-Object { \$_ }"

    if ! powershell.exe -NoProfile -NonInteractive -Command "$ps_script" > "$temp_env" 2>&1; then
        echo -e "${COLOR_RED}ERROR: Failed to execute vcvars64.bat${COLOR_RESET}" >&2
        rm -f "$temp_env"
        return 1
    fi

    echo "$temp_env"
}

# Convert Windows path to Unix format for bash PATH variable
convert_path_to_unix() {
    local win_path="$1"

    # Split by semicolon, convert each path, and rejoin with colon
    local unix_path=""
    local IFS=';'
    for path_component in $win_path; do
        # Remove carriage returns
        path_component=$(echo "$path_component" | tr -d '\r')

        # Convert backslashes to forward slashes
        path_component=$(echo "$path_component" | sed 's|\\|/|g')

        # Convert drive letters (C: -> /c, D: -> /d, etc.)
        if [[ "$path_component" =~ ^([A-Za-z]): ]]; then
            local drive="${BASH_REMATCH[1]}"
            drive=$(echo "$drive" | tr '[:upper:]' '[:lower:]')
            path_component=$(echo "$path_component" | sed "s|^[A-Za-z]:|/$drive|")
        fi

        if [[ -n "$unix_path" ]]; then
            unix_path="${unix_path}:${path_component}"
        else
            unix_path="$path_component"
        fi
    done

    echo "$unix_path"
}

# Export MSVC environment variables to bash
export_to_bash() {
    local env_file="$1"
    local exported_count=0

    # Read environment variables from temp file
    while IFS='=' read -r key value || [[ -n "$key" ]]; do
        # Remove carriage returns
        key=$(echo "$key" | tr -d '\r')
        value=$(echo "$value" | tr -d '\r')

        # Skip empty lines
        [[ -z "$key" ]] && continue

        # Check if this is a MSVC variable we need
        for var in "${MSVC_VARS[@]}"; do
            if [[ "$key" == "$var" ]]; then
                if [[ "$key" == "PATH" ]]; then
                    # Special handling for PATH: convert to Unix format
                    local unix_path=$(convert_path_to_unix "$value")
                    export PATH="$unix_path"
                else
                    # Export other variables as-is (bash can handle Windows paths in env vars)
                    export "$key=$value"
                fi
                ((exported_count++))
                break
            fi
        done
    done < "$env_file"

    return $exported_count
}

# Setup VAI_RT environment variables
# This integrates with the existing build workflow
setup_vairt_env() {
    local script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    local project_root="$(cd "$script_dir/.." && pwd)"

    # Convert to Windows-style paths (CMake expects Windows paths on Windows)
    # Use realpath to get absolute path, then convert to Windows format
    export VAI_RT_WORKSPACE=$(cd "$project_root/../.." && pwd -W 2>/dev/null || pwd | sed 's|^/\([a-z]\)/|\1:/|')
    export VAI_RT_BUILD_DIR="${VAI_RT_WORKSPACE}/build"
    export VAI_RT_PREFIX="${VAI_RT_WORKSPACE}/local"

    # Create directories if they don't exist
    mkdir -p "$VAI_RT_BUILD_DIR" 2>/dev/null || true
    mkdir -p "$VAI_RT_PREFIX" 2>/dev/null || true
}

# Validate that MSVC environment is correctly set up
validate_msvc_env() {
    local errors=0

    # Check if cl.exe is available
    if ! command -v cl.exe &>/dev/null; then
        echo -e "${COLOR_RED}ERROR: cl.exe not found in PATH${COLOR_RESET}" >&2
        ((errors++))
    fi

    # Check required environment variables
    for var in INCLUDE LIB LIBPATH VCINSTALLDIR; do
        if [[ -z "${!var:-}" ]]; then
            echo -e "${COLOR_RED}ERROR: $var not set${COLOR_RESET}" >&2
            ((errors++))
        fi
    done

    return $errors
}

# Print environment summary
print_env_summary() {
    echo -e "${COLOR_GREEN}MSVC Environment configured successfully!${COLOR_RESET}"
    echo ""
    echo "Visual Studio:"
    echo "  VCINSTALLDIR:      ${VCINSTALLDIR:-<not set>}"
    echo "  VCToolsInstallDir: ${VCToolsInstallDir:-<not set>}"
    echo ""
    echo "Windows SDK:"
    echo "  WindowsSdkDir:     ${WindowsSdkDir:-<not set>}"
    echo "  WindowsSDKVersion: ${WindowsSDKVersion:-<not set>}"
    echo ""
    echo "VAI Runtime:"
    echo "  VAI_RT_WORKSPACE:  ${VAI_RT_WORKSPACE:-<not set>}"
    echo "  VAI_RT_BUILD_DIR:  ${VAI_RT_BUILD_DIR:-<not set>}"
    echo "  VAI_RT_PREFIX:     ${VAI_RT_PREFIX:-<not set>}"
    echo ""

    if command -v cl.exe &>/dev/null; then
        echo "Compiler:"
        echo "  cl.exe: $(which cl.exe)"
        # Get compiler version (first line only)
        local cl_version=$(cl.exe 2>&1 | head -n 1 | tr -d '\r')
        echo "  Version: $cl_version"
    fi
}

# Main setup function
main() {
    echo -e "${COLOR_YELLOW}Setting up MSVC environment for bash...${COLOR_RESET}"

    # Step 1: Find vcvars64.bat
    local vcvars_bat=$(find_vcvars)
    if [[ $? -ne 0 ]]; then
        return 1
    fi
    echo "Found vcvars64.bat: $vcvars_bat"

    # Step 2: Capture MSVC environment from cmd.exe
    local temp_env=$(capture_msvc_env "$vcvars_bat")
    if [[ $? -ne 0 ]]; then
        return 1
    fi

    # Step 3: Export environment variables to bash
    export_to_bash "$temp_env"
    local export_count=$?

    # Clean up temp file
    rm -f "$temp_env"

    echo "Exported $export_count MSVC environment variables"

    # Step 4: Setup VAI_RT variables
    setup_vairt_env

    # Step 5: Validate environment
    if ! validate_msvc_env; then
        echo -e "${COLOR_RED}Environment validation failed${COLOR_RESET}" >&2
        return 1
    fi

    # Step 6: Print summary
    echo ""
    print_env_summary

    return 0
}

# Detect execution mode: sourced vs executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    # Script is being executed (not sourced)

    if [[ "${1:-}" == "--shell" ]]; then
        # Launch subshell mode
        echo "Launching MSVC-enabled bash subshell..."
        echo "Type 'exit' to return to parent shell"
        echo ""

        # Setup environment in current process
        main
        if [[ $? -eq 0 ]]; then
            # Launch interactive bash with environment
            exec bash -i
        else
            exit 1
        fi
    else
        # Just setup and exit
        main
        exit $?
    fi
else
    # Script is being sourced - setup current shell
    main
    # Don't exit, return instead (we're in a sourced context)
fi
