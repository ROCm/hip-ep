# Classification Test Executable

## Overview

The `test_classification` executable is a dedicated tool for testing ONNX classification models with the VitisAI Execution Provider in the morphizen-hipdnn project.

## Getting Started

### 1. Generate Input Binary File (Recommended)

You can generate a fresh `input.bin` file from any image using the provided Python tool:

```bash
# Navigate to test/data directory
cd test/data

# Generate input.bin from resnet50.jpg (or any other image)
python image_to_bin.py resnet50.jpg -o input.bin

# This creates input.bin with:
# - Shape: (3, 224, 224) in NCHW format
# - Dtype: float32
# - Size: 602,112 bytes
# - ImageNet standard normalization
```

**Alternative: Use custom image**
```bash
python image_to_bin.py your_image.jpg -o input.bin
```

**Alternative: Use custom size**
```bash
python image_to_bin.py your_image.jpg -o input.bin --size 256 256
```

For more details, see `test/data/README_image_to_bin.md`

### 2. Pull ONNX Model (Git LFS)

The ONNX model is stored using Git LFS:

```bash
# Install Git LFS if not already installed
git lfs install

# Pull LFS files
cd morphizen-hipdnn
git lfs pull
```

This will download:
- `test/data/pt_resnet50.onnx` (102 MB) - ResNet50 ONNX model
- `test/data/resnet50.jpg` (58 KB) - Test image

## Building

### Enable in CMake

The test is controlled by the `BUILD_TEST_CLASSIFICATION` CMake option (enabled by default):

```bash
cmake -DBUILD_TEST_CLASSIFICATION=ON ..
cmake --build . --target test_classification
```

### Requirements

- ONNX Runtime with VitisAI EP support
- glog library
- C++17 compiler

## Usage

### Basic Usage

```bash
test_classification <model.onnx> <image_file>
```

### Command-Line Options

- `-k <num>` - Number of top-K results to display (default: 5)
- `-n` - Disable VitisAI EP and use CPU only
- `-p` - Enable ONNX Runtime profiler
- `-l <file>` - Path to label file (one label per line)
- `-h` - Show help message

### Examples

#### Run with default settings (top-5 predictions)
```bash
test_classification data/pt_resnet50.onnx data/resnet50.jpg
```

#### Show top-10 predictions
```bash
test_classification -k 10 data/pt_resnet50.onnx data/resnet50.jpg
```

#### Use CPU only (disable VitisAI EP)
```bash
test_classification -n data/pt_resnet50.onnx data/resnet50.jpg
```

#### Use with ImageNet labels
```bash
test_classification -l imagenet_labels.txt data/pt_resnet50.onnx data/resnet50.jpg
```

#### Enable profiling
```bash
test_classification -p data/pt_resnet50.onnx data/resnet50.jpg
```

## Environment Variables

The test executable supports the same environment variables as `test_onnx_runner`:

- `VITISAI_EP_JSON_CONFIG` - Path to VitisAI EP configuration file
- `XLNX_USE_CACHE_KEY` - Cache key for model compilation
- `XLNX_USE_CACHE_DIR` - Directory for cache storage
- `XLNX_ENABLE_CACHE_CONTEXT` - Enable context caching (0 or 1)
- `CACHE_CONTEXT_EMBEDED_MODE` - Embed context in model (0 or 1)
- `XLNX_ENABLE_EP_SHARED_CONTEXT` - Enable shared EP context (0 or 1)
- `DEBUG_LOG_LEVEL` - Logging level (info, warning, error)
- `XLNX_ONNX_EP_VERBOSE` - Verbosity level (0-3)

### Example with Environment Variables

```bash
# Windows (PowerShell)
$env:VITISAI_EP_JSON_CONFIG = "C:/path/to/vaip_config.json"
$env:XLNX_USE_CACHE_DIR = "C:/Temp/"
$env:DEBUG_LOG_LEVEL = "info"
test_classification data/pt_resnet50.onnx data/resnet50.jpg

# Linux/macOS
export VITISAI_EP_JSON_CONFIG=/path/to/vaip_config.json
export XLNX_USE_CACHE_DIR=/tmp/
export DEBUG_LOG_LEVEL=info
./test_classification data/pt_resnet50.onnx data/resnet50.jpg
```

## Output

The executable provides:

1. **Model Information** - Input/output tensor shapes
2. **Session Creation Time** - Time to load and initialize the model
3. **Inference Time** - Time to run inference (in microseconds and milliseconds)
4. **Top-K Predictions** - Class IDs, labels (if provided), and confidence scores

### Sample Output

```
================VitisAIExecutionProviderenable_ep = true
HIP Library Path: C:\Windows\SYSTEM32\amdhip64_7.dll
Running model...
done
batch_index: 0
score[109]  =  0.997308     text: brain coral,,
score[973]  =  0.00116773   text: coral reef,,
score[5]    =  0.000909427  text: electric ray, crampfish, numbfish, torpedo,,
score[397]  =  0.000158035  text: puffer, pufferfish, blowfish, globefish,,
score[955]  =  0.000123078  text: jackfruit, jak, jack,,
```

## CTest Integration

The executable is integrated with CTest:

```bash
# Run the classification test
ctest -R test_classification_resnet50 -V

# Run all tests
ctest -V
```

## Test Data Location

Test data files are located in `test/data/` directory:
- Model: `test/data/pt_resnet50.onnx` (managed by Git LFS)
- Image: `test/data/resnet50.jpg` (managed by Git LFS)
- Input binary: `test/data/input.bin` (generated using image_to_bin.py)

## Running the Classification Test

### Step 1: Generate Input Binary

```bash
cd test/data
python image_to_bin.py resnet50.jpg -o input.bin
cd ../..
```

### Step 2: Run test_classification

After generating `input.bin`, run the classification test:

```bash
# Windows (PowerShell)
.\build\test\Release\test_classification.exe test\data\pt_resnet50.onnx test\data\input.bin

# Linux/macOS
./build/test/test_classification test/data/pt_resnet50.onnx test/data/input.bin
```

### Complete Workflow Example

```bash
# 1. Generate input binary from image
cd test/data
python image_to_bin.py resnet50.jpg -o input.bin
cd ../..

# 2. Set environment variables (optional)
# Windows
set USE_ORT_API_2_0=1
set DEBUG_LOG_LEVEL=info

# Linux/macOS
export USE_ORT_API_2_0=1
export DEBUG_LOG_LEVEL=info

# 3. Run classification test
# Windows
.\build\test\Release\test_classification.exe test\data\pt_resnet50.onnx test\data\input.bin

# Linux/macOS
./build/test/test_classification test/data/pt_resnet50.onnx test/data/input.bin
```

## Notes

- Input binary files are generated using the Python tool with ImageNet standard preprocessing
- The test supports any ONNX classification model with a single input and single output tensor
- Top predictions show class ID, label (if provided), and confidence score

## Troubleshooting

### Model not found
Ensure the model path is correct and the file exists.

### Image loading fails
Verify the image file path and format.

### VitisAI EP errors
Check that:
- VitisAI EP is properly installed
- Configuration file path is correct
- Required environment variables are set

### Build errors
Ensure all dependencies are installed:
- ONNX Runtime
- glog
- C++17 compatible compiler
