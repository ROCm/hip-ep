<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# MorphiZen

**MorphiZen** is a hardware-agnostic AI compiler framework that enables dynamic manipulation of ONNX graphs. MorphiZen provides a flexible, high-performance environment for writing passes, pattern matching, and rule-based transformations to optimize AI models for efficient deployment across diverse hardware targets.

## Key Features

- **ONNX Graph Manipulation**: Freely manipulate and optimize ONNX graphs with pattern matching and rewrite rule libraries.
- **Compiler Pass Framework**: Write custom passes for AI compilers to fine-tune and optimize models for specific hardware.
- **Pattern Matching**: Efficiently match and transform sections of AI models to suit performance and scalability needs.
- **Rewrite Rules**: Apply rule-based optimizations to enhance model inference and deployment.


## Prerequisites

- CMake 3.15 or higher
- C++17 compatible compiler (GCC, Clang, or MSVC)
- Git
- Python 3.8+ (for pre-commit hooks)
- **Pre-commit hooks** (required for contributors) - See [Developer Setup](#developer-setup) below

## Installation

### Basic Build

Clone the repository and build:

```bash
git clone https://github.com/ROCm/MorphiZen.git
cd MorphiZen

# For contributors: Set up pre-commit hooks (required)
# Windows:
scripts/setup-dev-env.ps1
# Linux/Mac:
scripts/setup-dev-env.sh

# Build
mkdir build
cd build
cmake ..
cmake --build .
```

### Windows MSVC Build

For Windows with MSVC and static runtime linking:

```bash
cmake -DCMAKE_CONFIGURATION_TYPES=Release -B $BUILD/morphizen -S $W/MorphiZen/ -DCMAKE_INSTALL_PREFIX=$PREFIX --fresh '-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded$<$<CONFIG:Debug>:Debug>'
cmake --build $BUILD/morphizen --config Release
```

## Developer Setup

**For Contributors**: Before making changes, set up pre-commit hooks to ensure your code passes CI checks:

```bash
# Windows (PowerShell):
scripts/setup-dev-env.ps1

# Linux/Mac:
scripts/setup-dev-env.sh
```

This installs pre-commit hooks that automatically run code formatters and linters before each commit.

**Important**: Pre-commit manages all formatting tools (clang-format, lintrunner, etc.) in an isolated environment. Never install or run these tools manually - pre-commit handles everything.

See [Developer Guide - Pre-commit Hooks](docs/developer-guide.md#6-pre-commit-hooks-required-for-contributors) for details.

## Documentation

- **[Quick Start](docs/developer-guide.md#installation)** - Get up and running
- **[Architecture Guide](docs/architecture.md)** - System design and components
- **[Developer Guide](docs/developer-guide.md)** - Contributing to MorphiZen (includes pre-commit setup)
- **[Documentation Index](docs/)** - Complete documentation navigation
- **[Git Workflow](docs/workflows/git-workflow.md)** - Branch strategy and commits
- **[Build Workflow](docs/workflows/build-workflow.md)** - Build system details
- **[PR Workflow](docs/workflows/pr-workflow.md)** - Pull request process

## Components

### Core Libraries

- **[morphizen-graph](morphizen-graph/)** - Graph manipulation utilities wrapping MORPHIZEN_ORT_API
- **[morphizen-pattern](morphizen-pattern/)** - Pattern matching library for graph transformations
- **[morphizen-utils](morphizen-utils/)** - Core utilities and helper functions
- **[morphizen-ort-api-ext](morphizen-ort-api-ext/)** - MORPHIZEN_ORT_API interface (111-function custom abstraction)

### Backend Implementations

- **[onnx-ir](onnx-ir-imp/)** - ONNX-based IR backend implementation
- **[mlir-imp](mlir-imp/)** - MLIR backend implementation (partial)
- **[ort-bridge](ort-bridge/)** - ORT execution provider bridge

### Tools

- **[graph-opt](graph-opt/)** - Graph optimization tool
- **[onnx-grep](onnx-grep/)** - Pattern search tool for ONNX models
- **[pattern-gen](pattern-gen/)** - Pattern generation tool

See [Architecture Guide](docs/architecture.md) for detailed component relationships.

## Quick Links

- **[Installation Guide](docs/developer-guide.md#installation)** - Detailed build instructions
- **[Testing](docs/developer-guide.md#testing)** - How to run tests
- **[Contributing](docs/developer-guide.md#contributing)** - Contribution guidelines
- **[Troubleshooting](docs/developer-guide.md#troubleshooting)** - Common issues and solutions
- **[CHANGELOG](CHANGELOG.md)** - Release history and breaking changes
