# Classification Test Executable

## Overview

The `test_classification` executable is a dedicated tool for testing ONNX classification models with the VitisAI Execution Provider in the morphizen-hipdnn project.

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
=== Classification Test ===
Model: data/pt_resnet50.onnx
Image: data/resnet50.jpg
Top-K: 5

Using VitisAI EP config: etc/vaip_config.json
VitisAI EP: ENABLED

Creating session...
Session created in 1234 ms

Model Info:
  Input: input [1, 3, 224, 224]
  Output: output [1, 1000]

Loading image...

Running inference...
Inference completed in 5678 μs (5.678 ms)

=== Top-5 Predictions ===
1. golden_retriever (class 207) - confidence: 85.23%
2. labrador_retriever (class 208) - confidence: 8.45%
3. cocker_spaniel (class 219) - confidence: 3.12%
4. irish_setter (class 212) - confidence: 1.89%
5. english_setter (class 213) - confidence: 0.67%

=== Test Complete ===
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

By default, the test expects:
- Model: `${CMAKE_SOURCE_DIR}/data/pt_resnet50.onnx`
- Image: `${CMAKE_SOURCE_DIR}/data/resnet50.jpg`

Ensure these files exist in the `data/` directory at the project root.

## Notes

- The current implementation uses simple image preprocessing (normalization to [0,1])
- For production use, you may need to implement proper image decoding, resizing, and model-specific preprocessing
- The test supports any ONNX classification model with a single input and single output tensor
- Confidence scores are displayed as percentages

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
