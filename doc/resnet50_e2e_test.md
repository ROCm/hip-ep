# ResNet50 End-to-End Test Guide

## Quick Start

### 1. Generate Input Binary

```bash
cd test/data
python image_to_bin.py resnet50.jpg -o input.bin
cd ../..
```

### 2. Run Test

```bash
# Windows
export PATH="$THEROCK_DIST/bin;$PATH"
.\build\test\Release\test_classification.exe test\data\pt_resnet50.onnx test\data\input.bin

# Linux/macOS
./build/test/test_classification test/data/pt_resnet50.onnx test/data/input.bin
```

## Expected Output

```
================MorphiZenExecutionProviderenable_ep = true
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

## Command-Line Options

- `-k <num>` - Top-K results (default: 5)
- `-n` - Disable Morphizen EP (CPU only)
- `-p` - Enable profiler
- `-l <file>` - Label file path
- `-h` - Show help

## Test Data

Located in `test/data/`:
- `pt_resnet50.onnx` (102 MB) - Model (Git LFS)
- `resnet50.jpg` (58 KB) - Image (Git LFS)
- `input.bin` (602 KB) - Generated binary

## Image to Binary Tool

See [test/data/README_image_to_bin.md](../test/data/README_image_to_bin.md) for details.

```bash
# Basic usage
python test/data/image_to_bin.py input.jpg -o output.bin

# Custom size
python test/data/image_to_bin.py input.jpg --size 256 256 -o output.bin
