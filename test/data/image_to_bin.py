#!/usr/bin/env python3
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""
Simple Image to Binary Converter
Converts images to binary format for ONNX model inference
"""

import argparse
import numpy as np
from PIL import Image


def image_to_bin(image_path, output_path, size=(224, 224)):
    """
    Convert image to binary file with standard preprocessing

    Args:
        image_path: Input image path
        output_path: Output binary file path
        size: Target size (height, width)
    """
    # Load and preprocess image
    img = Image.open(image_path).convert("RGB")
    img = img.resize(size, Image.BILINEAR)

    # Convert to numpy array and normalize
    img_array = np.array(img, dtype=np.float32)
    img_array = img_array / 255.0  # Scale to [0, 1]

    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_array = (img_array - mean) / std

    # Convert to NCHW format (channels first)
    img_array = np.transpose(img_array, (2, 0, 1))

    # Save to binary file
    img_array.tofile(output_path)

    print(f"Converted: {image_path} -> {output_path}")
    print(f"Shape: {img_array.shape}, Size: {img_array.nbytes} bytes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert image to binary")
    parser.add_argument("input", help="Input image path")
    parser.add_argument("-o", "--output", help="Output binary path")
    parser.add_argument(
        "--size",
        type=int,
        nargs=2,
        default=[224, 224],
        help="Target size (height width), default: 224 224",
    )

    args = parser.parse_args()

    output = args.output or args.input.rsplit(".", 1)[0] + ".bin"

    try:
        image_to_bin(args.input, output, tuple(args.size))
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
