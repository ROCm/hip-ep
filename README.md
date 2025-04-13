# MorphiZen

**MorphiZen** is a powerful AI compiler framework that enables dynamic manipulation of ONNX graphs. Built for AMD Ryzen AI, MorphiZen provides a flexible, high-performance environment for writing passes, pattern matching, and rule-based transformations. With its advanced capabilities, MorphiZen optimizes AI models to achieve efficient and scalable deployment.

## Key Features

- **ONNX Graph Manipulation**: Freely manipulate and optimize ONNX graphs with pattern matching and rewrite rule libraries.
- **Compiler Pass Framework**: Write custom passes for AI compilers to fine-tune and optimize models for specific hardware.
- **Pattern Matching**: Efficiently match and transform sections of AI models to suit performance and scalability needs.
- **Rewrite Rules**: Apply rule-based optimizations to enhance model inference and deployment on AMD Ryzen AI hardware.


## Installation

Clone the repository:

```bash
git clone git@gitenterprise.xilinx.com:aisw/MorphiZen.git
cd MorphiZen
git submodule upgrade --init
mkdir build
cd build
cmake ..
cmake --build
```

```
 cmake -DCMAKE_CONFIGURATION_TYPES=Release -B$BUILD/morphizen -S $W/MorphiZen/ -DCMAKE_INSTALL_PREFIX=$PREFIX --fresh '-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded$<$<CONFIG:Debug>:Debug>'
 
 ```
