<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# MorphiZen Documentation

Welcome to the MorphiZen documentation! This guide will help you navigate the available resources.

## Getting Started

1. **New Users**: Start with [Installation](../README.md#installation) in the root README
2. **Contributors**: Read the [Developer Guide](developer-guide.md)
3. **Architects**: Understand the [Architecture](architecture.md)

## Documentation Structure

### Core Documentation
- **[Architecture Guide](architecture.md)** - System design, components, and patterns
- **[Developer Guide](developer-guide.md)** - Setup, building, testing, and contributing
- **[CHANGELOG](../CHANGELOG.md)** - Release history and breaking changes

### Technical Deep-Dives
- **[Temporary File Handling](technical/tmpfile-posix-delete.md)** - POSIX file deletion patterns
- **[EP Context](technical/ep-context.md)** - Execution provider context generation, deployment, and internal design
- **[glog Integration](technical/glog-integration.md)** - Logging framework integration
- **[ORT ETW Tracing](technical/enable-ort-etw-trace-and-logs.md)** - Event tracing and logging
- **[Cleanup Provider Options](technical/cleanup-provider-options.md)** - Provider cleanup patterns
- **[Excluded Packages](technical/excluded-packages.md)** - Package exclusion documentation
- **[Target Auto Discovery](technical/target-auto-discovery.md)** - CMake target discovery
- **[Pre-commit Setup](workflows/git-workflow-reference.md#pre-commit-hook-behavior)** - Code quality hooks and linters

### Workflows
- **[Project Backlog](project/backlog.md)** - Issue tracking and roadmap
- **[Git Workflow](workflows/git-workflow.md)** - Critical rules and quick reference
- **[Git Workflow Reference](workflows/git-workflow-reference.md)** - Detailed step-by-step git procedures
- **[Build Workflow](workflows/build-workflow.md)** - CMake build system and configurations
- **[PR Workflow](workflows/pr-workflow.md)** - Pull request creation, review, and cleanup

### Components

See individual component documentation in their respective directories:

#### Core Libraries
- **[morphizen-graph](../morphizen-graph/)** - Graph manipulation utilities wrapping MORPHIZEN_ORT_API
- **[morphizen-pattern](../morphizen-pattern/)** - Pattern matching library for graph transformations
- **[morphizen-utils](../morphizen-utils/)** - Core utilities and helper functions
- **[morphizen-ort-api-ext](../morphizen-ort-api-ext/)** - MORPHIZEN_ORT_API interface definition (111-function custom abstraction)

#### Backend Implementations
- **[onnx-ir](../onnx-ir-imp/)** - ONNX-based IR backend implementation
- **[mlir-imp](../mlir-imp/)** - MLIR backend implementation (partial)
- **[ort-bridge](../ort-bridge/)** - ORT execution provider bridge

#### Tools
- **[graph-opt](../graph-opt/)** - Graph optimization tool
- **[onnx-grep](../onnx-grep/)** - Pattern search tool for ONNX models
- **[pattern-gen](../pattern-gen/)** - Pattern generation tool

## Finding What You Need

- **Want to build MorphiZen?** → [Developer Guide - Installation](developer-guide.md#installation)
- **Need to understand the architecture?** → [Architecture Guide](architecture.md)
- **Looking for component docs?** → See component directories above (morphizen-graph/, morphizen-pattern/, etc.)
- **Planning or tracking work?** → [Project Backlog](project/backlog.md)
- **Reviewing a PR?** → [PR Workflow - Reviewing](workflows/pr-workflow.md#reviewing-a-pull-request)
- **Git workflow questions?** → [Git Workflow](workflows/git-workflow.md)
- **How to run tests?** → [Developer Guide - Testing](developer-guide.md#testing)
- **Build system questions?** → [Build Workflow](workflows/build-workflow.md)
- **Technical deep-dive topics?** → See [Technical Deep-Dives](#technical-deep-dives) section above

## Contributing

Before contributing, please review:
1. [Git Workflow](workflows/git-workflow.md) - Required branch and commit workflow
2. [Developer Guide](developer-guide.md) - Development environment setup
3. [PR Workflow](workflows/pr-workflow.md) - Pull request process

## Documentation Organization

This documentation follows a clear structure:

- **`/docs/`** - All project documentation (this directory)
  - **Core guides** at the top level (architecture, developer guide)
  - **`/technical/`** - Deep-dive technical documentation
  - **`/workflows/`** - Development workflows and processes
- **Component docs** - In each component's directory (morphizen-graph/README.md, etc.)
- **Root README** - Project overview and quick start

## Need Help?

- Check the [Developer Guide troubleshooting section](developer-guide.md#troubleshooting)
- Review [Technical Deep-Dives](#technical-deep-dives) for specific topics
- Look at component-specific READMEs for detailed API information
- See [PR Workflow](workflows/pr-workflow.md) for collaboration processes
