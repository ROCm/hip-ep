# Image to Binary Converter

Simple image to binary conversion tool for ONNX model inference.

## Dependencies

```bash
pip install numpy pillow
```

## Usage

### Basic Usage

```bash
python image_to_bin.py input.jpg
```

Generates `input.bin` file with default settings:
- Size: 224x224
- Format: NCHW (channels first)
- Normalization: ImageNet standard (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- Data type: float32

### Specify Output File

```bash
python image_to_bin.py input.jpg -o output.bin
```

### Custom Size

```bash
python image_to_bin.py input.jpg --size 256 256
```

## Output Format

- **Shape**: (3, H, W) - NCHW format
- **Dtype**: float32
- **Size**: 3 × H × W × 4 bytes

Example for 224x224: 3 × 224 × 224 × 4 = 602,112 bytes

## Python API

```python
from image_to_bin import image_to_bin

# Convert image
image_to_bin('input.jpg', 'output.bin', size=(224, 224))
```

## Reading Binary Files

```python
import numpy as np

# Read binary file
data = np.fromfile('output.bin', dtype=np.float32)
data = data.reshape(3, 224, 224)
