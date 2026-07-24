<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->


### Option 2: Pre-installed Dependencies


```
cd workspace
git clone git@gitenterprise.xilinx.com:VitisAI/morphizen-demo.git
(git clone git@gitenterprise.xilinx.com:VitisAI/MorphiZen.git;cd MorphiZen;git submodule upgrade --init)

cp morphizen-demo/env.sh .
source env.sh
git clone git@gitenterprise.xilinx.com:VitisAI/vai-rt.git
env USE_CXX_ONLY_ORT=ON python $W/vai-rt/main.py --dev-mode  \
   --project zlib gsl json gtest protobuf eigen glog onnxruntime_cxx

cmake -DCMAKE_CONFIGURATION_TYPES=Debug -B$BUILD/morphizen-demo -S $W/morphizen-demo/ -DCMAKE_INSTALL_PREFIX=$PREFIX -S$W/morphizen-demo
```

In this option, all dependencies are installed beforehand, making incremental builds faster and more convenient for developers.




## concept

### 1. morphizen-demo
How ONNX Runtime Execution Providers (EPs) Work
An EP can partition the Graph into multiple subgraphs and implement custom computation methods.
Subgraph Structure
A subgraph is described by:
- Input tensors
- Output tensors
- Nodes
The structure is roughly as follows:
```
struct{
  std::vector<Tensor> inputs;
  std::vector<Tensor> outputs;
  std::vector<Node> nodes;
}
```

Custom Computation Method
ONNX Runtime provides the input tensors of the subgraph, allowing users to implement a custom computation function to process the input and generate the output data. The output data is then returned to ONNX Runtime via ctx.

Once these two modules are defined, a complete morphizen flow is established.

### 2. Level1-pass
The process of partitioning a Graph into N Subgraphs in an ONNX Runtime Execution Provider (EP) typically follows these steps:
```
   I -> A -> C ->D -> O
          \_ B _/
```
In the structure described above, knowing only I_output and D_output is sufficient to identify the intermediate subgraph.
```
    I -> subgraph -> O
```
In the Level1-pass, users need to call the fuse function and pass in std::vector<Tensor> inputs and std::vector<Tensor> outputs to obtain metadata (metadef) that describes the subgraph.

In the dummy example, the entire network's input and output are passed to the fuse function, making the subgraph the entire model.
### 3. passes
Both Level1-pass and pass inherit from Ipass. In the code, they both represent a type of Ipass, but they differ in concept and responsibility.

A pass takes a Graph as input and produces a Graph as output, typically used for graph transformations and optimizations.
In the dummy example, the pass does nothing, meaning the input model is returned unchanged.
### 4. custom_op
The constructor and compute function serve as the interfaces for interaction between ONNX Runtime and custom_op.

When creating a session, the constructor is called to perform configuration and initialization.
During execution (run), the compute function is invoked, where users need to implement the inference logic for their subgraph.
Each subgraph corresponds to a custom_op object.

In the dummy example, both the constructor and compute function are dummy implementations.
## How to embed to morphizen

After implementing your own morphizen flow, copy it to the morphizen root directory and modify CMakeLists.txt to add your module as a CMake subdirectory, for example: `add_subdirectory(dummy)`
This ensures that morphizen will automatically link the entire morphizen_flow into onnxruntime_vitisai_ep.dll.

Next, modify config.json and add the following content under passes.

```
      {
        "name": "dummy",
        "plugin": "morphizen-pass_level1_dummy",
          "passDpuParam": {
          "subPass": [
            {
              "name": "dummy",
              "plugin": "morphizen-pass_dummy",
              "enable_gc": true,
              "disabled": false
            }
          ]
        }
      }
```
