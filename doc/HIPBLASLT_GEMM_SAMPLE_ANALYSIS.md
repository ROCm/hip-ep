# hipBLASLt GEMM Sample Analysis

This document analyzes the hipBLASLt GEMM sample from the ROCm libraries repository and documents the requirements for running it.

## Sample Source

- **Repository**: [ROCm/rocm-libraries](https://github.com/ROCm/rocm-libraries)
- **Path**: `projects/hipblaslt/clients/samples/01_hipblaslt_gemm/`
- **Files**:
  - `sample_hipblaslt_gemm.cpp` - Main sample code
  - Helper header from `projects/hipblaslt/clients/samples/common/helper.h`

## Sample Overview

The sample demonstrates a basic **General Matrix Multiplication (GEMM)** operation using the hipBLASLt library on AMD GPUs.

### Operation Performed

The sample computes: **D = α × A × B + β × C**

Where:
- **A**: Input matrix (M × K) in FP16 format
- **B**: Input matrix (K × N) in FP16 format
- **C**: Input matrix (M × N) in FP16 format  
- **D**: Output matrix (M × N) in FP16 format
- **α, β**: Scalars in FP32 format
- **Compute Type**: FP32 (for higher precision accumulation)

### Sample Parameters

| Parameter | Value |
|-----------|-------|
| M (rows of A, C, D) | 1024 |
| N (cols of B, C, D) | 512 |
| K (cols of A, rows of B) | 1024 |
| Batch Count | 1 |
| Alpha (α) | 1.0 |
| Beta (β) | 1.0 |
| Workspace Size | 32 MB |
| Data Type A | FP16 (HIP_R_16F) |
| Data Type B | FP16 (HIP_R_16F) |
| Data Type C | FP16 (HIP_R_16F) |
| Data Type D | FP16 (HIP_R_16F) |
| Compute Type | FP32 (HIPBLAS_COMPUTE_32F) |
| Transpose A | No (HIPBLAS_OP_N) |
| Transpose B | No (HIPBLAS_OP_N) |

## Key API Calls

The sample demonstrates the following hipBLASLt API usage pattern:

### 1. Library Initialization
```cpp
hipblasLtHandle_t handle;
hipblasLtCreate(&handle);
```

### 2. Matrix Layout Creation
```cpp
hipblasLtMatrixLayout_t matA, matB, matC, matD;
hipblasLtMatrixLayoutCreate(&matA, HIP_R_16F, m, k, m);  // lda = m
hipblasLtMatrixLayoutCreate(&matB, HIP_R_16F, k, n, k);  // ldb = k
hipblasLtMatrixLayoutCreate(&matC, HIP_R_16F, m, n, m);  // ldc = m
hipblasLtMatrixLayoutCreate(&matD, HIP_R_16F, m, n, m);  // ldd = m
```

### 3. Matmul Descriptor Creation
```cpp
hipblasLtMatmulDesc_t matmul;
hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F);
hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &trans_a, ...);
hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &trans_b, ...);
hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, ...);
```

### 4. Algorithm Selection (Heuristic)
```cpp
hipblasLtMatmulPreference_t pref;
hipblasLtMatmulPreferenceCreate(&pref);
hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, ...);

hipblasLtMatmulHeuristicResult_t heuristicResult[1];
int returnedAlgoCount;
hipblasLtMatmulAlgoGetHeuristic(handle, matmul, matA, matB, matC, matD,
                                 pref, 1, heuristicResult, &returnedAlgoCount);
```

### 5. Execute GEMM
```cpp
hipblasLtMatmul(handle, matmul,
                &alpha, d_a, matA, d_b, matB,
                &beta, d_c, matC, d_d, matD,
                &heuristicResult[0].algo,
                d_workspace, workspace_size, stream);
```

### 6. Cleanup
```cpp
hipblasLtMatrixLayoutDestroy(matA);
hipblasLtMatrixLayoutDestroy(matB);
hipblasLtMatrixLayoutDestroy(matC);
hipblasLtMatrixLayoutDestroy(matD);
hipblasLtMatmulDescDestroy(matmul);
hipblasLtMatmulPreferenceDestroy(pref);
hipblasLtDestroy(handle);
```

## Memory Management: Create/Destroy Pairs

All hipBLASLt descriptor creation functions **allocate memory** for their opaque structures and **require explicit destruction** to avoid memory leaks.

### Official API Documentation Excerpts

#### `hipblasLtMatrixLayoutCreate`
> This function creates a matrix layout descriptor **by allocating the memory needed to hold its opaque structure**.
>
> **Return values:**
> - `HIPBLAS_STATUS_SUCCESS` – If the descriptor was created successfully.
> - `HIPBLAS_STATUS_ALLOC_FAILED` – **If the memory could not be allocated.**

#### `hipblasLtMatrixLayoutDestroy`
> This function **destroys a previously created matrix layout descriptor** object.

### Complete Create/Destroy Reference Table

| Create Function | Allocates Memory? | Destroy Function | Notes |
|-----------------|-------------------|------------------|-------|
| `hipblasLtCreate` | Yes (host/device resources) | `hipblasLtDestroy` | Calls `hipDeviceSynchronize()` implicitly |
| `hipblasLtMatrixLayoutCreate` | Yes (opaque structure) | `hipblasLtMatrixLayoutDestroy` | For matrix metadata only |
| `hipblasLtMatmulDescCreate` | Yes (opaque structure) | `hipblasLtMatmulDescDestroy` | Stores operation config |
| `hipblasLtMatmulPreferenceCreate` | Yes (opaque structure) | `hipblasLtMatmulPreferenceDestroy` | Heuristic preferences |
| `hipblasLtMatrixTransformDescCreate` | Yes (opaque structure) | `hipblasLtMatrixTransformDescDestroy` | For matrix transforms |

### Best Practices

1. **Always call destroy functions** - Memory leaks will occur otherwise
2. **Check return values** - `HIPBLAS_STATUS_ALLOC_FAILED` indicates allocation failure
3. **Minimize Create/Destroy calls** - Re-use descriptors when possible for performance
4. **Handle cleanup on errors** - Use RAII or cleanup labels to ensure proper destruction

### Note on GPU Memory

The descriptor functions (`hipblasLtMatrixLayoutCreate`, etc.) only allocate **CPU-side memory** for metadata. The actual **GPU memory** for matrices (`d_a`, `d_b`, `d_c`, `d_d`) is managed separately with `hipMalloc`/`hipFree`.

Source: [hipBLASLt API Reference](https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/api-reference.html)

## Helper Class: Runner

The `helper.h` provides a `Runner` template class that handles:

1. **Memory Management**:
   - Device memory allocation (`hipMalloc`)
   - Host memory allocation (`hipHostMalloc`)
   - Automatic cleanup in destructor

2. **Data Initialization**:
   - Random data generation for input matrices
   - Values in range [-3, 3]

3. **Data Transfer**:
   - `hostToDevice()`: Copies input data to GPU
   - `deviceToHost()`: Copies results back to CPU

4. **Stream Synchronization**:
   - Uses HIP streams for async operations
   - Synchronizes after computation

## Requirements to Run

### Hardware Requirements
- AMD GPU with ROCm support (gfx9xx, gfx10xx, gfx11xx, or gfx12xx series)
- Recommended: At least 4GB VRAM

### Software Requirements

#### Linux
```bash
# Install ROCm (Ubuntu example)
sudo apt update
sudo apt install rocm-hip-sdk rocm-libs

# hipBLASLt is included in rocm-libs
```

#### Windows
1. **Install TheRock ROCm SDK** (nightly tarball):
   - Download from: https://therock-nightly-tarball.s3.amazonaws.com/index.html
   - Extract to `C:\dist\therock`

2. **Set Environment Variables**:
   ```cmd
   set HIP_PLATFORM=amd
   set PATH=C:\dist\therock\bin;%PATH%
   ```

3. **Build hipBLASLt** (if not included in TheRock):
   - Clone rocm-libraries
   - Build the hipblaslt project with CMake

### Build Commands

#### Linux
```bash
cd samples/hipblaslt_gemm
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=/opt/rocm ..
make
./sample_hipblaslt_gemm
```

#### Windows (with TheRock)
```cmd
cd samples\hipblaslt_gemm
mkdir build
cd build
cmake -G Ninja -DCMAKE_PREFIX_PATH=C:\dist\therock ..
ninja
sample_hipblaslt_gemm.exe
```

## Expected Output

When run successfully, the sample should output:
```
=== hipBLASLt GEMM Sample ===
Matrix dimensions: M=1024, N=512, K=1024
Data type: FP16 (half precision)
Compute type: FP32

Starting GEMM computation...
  Creating matrix layouts...
  Creating matmul descriptor...
  Setting user preferences...
  Getting heuristic algorithm...
  Found 1 algorithm(s)
  Required workspace size: XXXXX bytes
  Executing GEMM...
  Cleaning up resources...
GEMM computation completed successfully!

=== Sample finished ===
```

## Key Concepts Demonstrated

### 1. Half-Precision Computing
- Uses FP16 for storage (efficient memory bandwidth)
- Uses FP32 for computation (numerical stability)
- This is common for deep learning workloads

### 2. Heuristic Algorithm Selection
- hipBLASLt automatically selects optimal kernel
- Based on matrix sizes and available hardware
- Workspace size affects algorithm selection

### 3. Workspace Memory
- Some algorithms require temporary workspace
- Pre-allocate workspace for best performance
- Sample uses 32MB workspace

### 4. Batched Operations Support
- Code supports batched GEMM (batch_count > 1)
- Uses strided layout for batched matrices
- Useful for deep learning batch processing

## Related Samples in rocm-libraries

The hipBLASLt samples directory contains additional samples:

| Sample | Description |
|--------|-------------|
| 01_hipblaslt_gemm | Basic GEMM (this sample) |
| 02_hipblaslt_gemm_batched | Batched GEMM operations |
| 04_hipblaslt_gemm_bias | GEMM with bias addition |
| 05_hipblaslt_gemm_get_all_algos | Query all available algorithms |
| 08_hipblaslt_gemm_gelu_aux_bias | GEMM with GELU activation |
| 16_hipblaslt_groupedgemm_ext | Grouped GEMM operations |
| 19_hipblaslt_gemm_mix_precision | Mixed precision GEMM |

## Integration Notes

### For ONNX Runtime / Deep Learning Frameworks

hipBLASLt is commonly used as a backend for:
- **ONNX Runtime**: MatMul and GEMM operators
- **PyTorch**: torch.mm, torch.bmm operations
- **TensorFlow**: Dense layers, attention mechanisms

The heuristic-based algorithm selection makes it suitable for dynamic shapes while maintaining high performance.

### Performance Considerations

1. **Warm-up**: First call may be slower due to JIT compilation
2. **Algorithm Caching**: Re-use descriptors when possible
3. **Workspace**: Larger workspace may enable faster algorithms
4. **Memory Layout**: Column-major (Fortran-style) layout expected

## Files Created in This Analysis

The following files were created as part of this analysis:

1. `samples/hipblaslt_gemm/sample_hipblaslt_gemm.cpp` - Main sample code
2. `samples/hipblaslt_gemm/helper.h` - Helper utilities
3. `samples/hipblaslt_gemm/CMakeLists.txt` - Build configuration

## Building the Sample with CMake (Windows)

### Prerequisites

1. **TheRock SDK** installed at `C:\Develop\m\dist\therock`
2. **CMake 4.x+** and **Ninja** build system
3. **Windows SDK** (for resource compiler `rc.exe`)

### Step 1: Configure with CMake

The key is to use Clang from TheRock's LLVM distribution (MSVC is incompatible with TheRock headers):

```powershell
# Configure the build
cmake -G Ninja `
    -B build\samples\hipblaslt_gemm `
    -S samples\hipblaslt_gemm `
    -DCMAKE_CXX_COMPILER="C:/Develop/m/dist/therock/lib/llvm/bin/clang++.exe" `
    -DCMAKE_C_COMPILER="C:/Develop/m/dist/therock/lib/llvm/bin/clang.exe" `
    -DCMAKE_RC_COMPILER="C:/Program Files (x86)/Windows Kits/10/bin/10.0.26100.0/x64/rc.exe" `
    -DCMAKE_BUILD_TYPE=Release `
    -DTHEROCK_DIST="C:/Develop/m/dist/therock"
```

**Note**: The `CMAKE_RC_COMPILER` must point to the Windows resource compiler. Find yours with:
```powershell
Get-ChildItem -Path 'C:\Program Files*\Windows Kits\10\bin\*\x64\rc.exe' -Recurse | Select-Object -First 1
```

### Step 2: Build

```cmd
cmake --build build\samples\hipblaslt_gemm --config Release
```

### Step 3: Run

```cmd
cd build\samples\hipblaslt_gemm
set PATH=C:\Develop\m\dist\therock\bin;%PATH%
sample_hipblaslt_gemm.exe
```

### CMakeLists.txt Explained

The sample's `CMakeLists.txt` uses standard CMake with HIP support:

```cmake
cmake_minimum_required(VERSION 3.21)
project(sample_hipblaslt_gemm CXX)

# Set TheRock distribution path
set(THEROCK_DIST "C:/Develop/m/dist/therock" CACHE PATH "TheRock distribution path")
message(STATUS "Using TheRock at: ${THEROCK_DIST}")

# Configure CMake to find packages in TheRock
list(APPEND CMAKE_PREFIX_PATH "${THEROCK_DIST}")
list(APPEND CMAKE_PREFIX_PATH "${THEROCK_DIST}/lib/cmake")

# Find HIP package
find_package(hip REQUIRED)
message(STATUS "Found HIP: ${hip_DIR}")

# Find hipBLASLt package  
find_package(hipblaslt REQUIRED)
message(STATUS "Found hipBLASLt: ${hipblaslt_DIR}")

# Create executable
add_executable(sample_hipblaslt_gemm sample_hipblaslt_gemm.cpp)

# Link to hipBLASLt (brings in HIP automatically)
target_link_libraries(sample_hipblaslt_gemm PRIVATE roc::hipblaslt)

# Add include path for helper.h
target_include_directories(sample_hipblaslt_gemm PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Enable HIP compilation
set_source_files_properties(sample_hipblaslt_gemm.cpp PROPERTIES LANGUAGE HIP)
```

### Common Build Issues

| Issue | Solution |
|-------|----------|
| `No CMAKE_RC_COMPILER` | Add `-DCMAKE_RC_COMPILER=...` pointing to rc.exe |
| `MSVC compatibility errors` | Use Clang from TheRock, not MSVC |
| `hipblasLtHalf conversion error` | Use `__float2half()` with bit-level copy (see Build Notes) |
| `DLL not found at runtime` | Add TheRock's bin directory to PATH |

## Actual Run Results

The sample was successfully built and executed on Windows 11 with TheRock SDK:

```
=== hipBLASLt GEMM Sample ===
Matrix dimensions: M=1024, N=512, K=1024
Data type: FP16 (half precision)
Compute type: FP32

Starting GEMM computation...
  Creating matrix layouts...
  Creating matmul descriptor...
  Setting user preferences...
  Getting heuristic algorithm...
  Found 1 algorithm(s)
  Required workspace size: 0 bytes
  Executing GEMM...
  Cleaning up resources...
GEMM computation completed successfully!

=== Sample finished ===
```

### Build Configuration Used

- **Compiler**: Clang 22.0.0 (from TheRock SDK at `C:\Develop\m\dist\therock\lib\llvm\bin`)
- **Build System**: CMake 4.2 + Ninja
- **HIP SDK**: TheRock distribution at `C:\Develop\m\dist\therock`
- **Target Architecture**: gfx906 (auto-detected)

### Key Observations

1. **Zero Workspace Size**: The heuristic algorithm selected required 0 bytes of workspace memory. This indicates the algorithm can operate directly on the input/output buffers without additional temporary storage.

2. **Single Algorithm Found**: The heuristic search returned exactly 1 algorithm, suggesting the dimensions (1024×512×1024) with FP16/FP32 types have a well-defined optimal implementation.

3. **Successful GPU Execution**: The GEMM operation completed without errors, demonstrating that:
   - HIP runtime initialized correctly
   - GPU memory allocation worked
   - Data transfer between host and device succeeded
   - The hipBLASLt kernel executed properly

### Build Notes

The original sample code from ROCm required modification to handle the `hipblasLtHalf` type correctly on Windows with the TheRock SDK. The `hipblasLtHalf` type is defined as a struct with a `uint16_t data` member, requiring explicit bit-level conversion:

```cpp
__half h = __float2half(static_cast<float>(value));
((hipblasLtHalf*)ptr)[i].data = *reinterpret_cast<uint16_t*>(&h);
```

## References

1. [hipBLASLt Documentation](https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/)
2. [ROCm Libraries Repository](https://github.com/ROCm/rocm-libraries)
3. [TheRock Releases](https://github.com/ROCm/TheRock/blob/main/RELEASES.md)
4. [hipBLASLt API Reference](https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/api-reference.html)
