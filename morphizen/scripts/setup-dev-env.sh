#!/bin/bash
##
## Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
## Licensed under the MIT License.
##

# MorphiZen Developer Environment Setup Script
# This script installs pre-commit hooks for automatic code formatting

echo "=== MorphiZen Developer Environment Setup ==="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "✗ Python not found. Please install Python 3.8+ first."
    echo "  Ubuntu/Debian: sudo apt-get install python3 python3-pip"
    echo "  RHEL/CentOS: sudo yum install python3 python3-pip"
    exit 1
fi

PYTHON_VERSION=$(python3 --version)
echo "✓ Python found: $PYTHON_VERSION"

# Install pre-commit
echo ""
echo "Installing pre-commit..."
pip3 install pre-commit

# Install git hooks
echo ""
echo "Installing pre-commit git hooks..."
pre-commit install

echo ""
echo "=== Setup Complete ==="
echo "Pre-commit hooks are now installed."
echo ""
echo "IMPORTANT:"
echo "  - Pre-commit manages all formatting tools (clang-format, lintrunner, etc.)"
echo "  - Do NOT install or run clang-format manually"
echo "  - Always use pre-commit for code formatting"
echo ""
echo "Usage:"
echo "  - Hooks run automatically on 'git commit'"
echo "  - Manual run: pre-commit run --all-files"
echo "  - Update hooks: pre-commit autoupdate"
echo ""
echo "Next steps:"
echo "  1. Run 'pre-commit run --all-files' to check existing code"
echo "  2. See docs/developer-guide.md for more information"
