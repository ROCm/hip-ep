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

## Installation

### Basic Build

Clone the repository and build:

```bash
git clone https://github.com/ROCm/MorphiZen.git
cd MorphiZen
git submodule update --init --recursive
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
