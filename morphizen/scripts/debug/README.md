<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Debugging Utilities

Troubleshooting and debugging scripts for MorphiZen development.

## morphizen_check_version.py

Extract build information from MorphiZen DLL.

**Purpose:** Verify DLL build info, check component versions

**Usage:**
```bash
python morphizen_check_version.py <path_to_onnxruntime_vitisai_ep.dll>
```

**Example:**
```bash
python scripts/debug/morphizen_check_version.py ../../build/morphizen/bin/onnxruntime_vitisai_ep.dll
```

**Output:** Build info including version, commit hash, build date, etc.

---

## convert_onnx_to_external_data_mode.py

Convert ONNX models to external data format (separates large tensors into .data file).

**Purpose:** Debug large models, reduce .onnx file size

**Usage:**
```bash
python convert_onnx_to_external_data_mode.py <input.onnx> <output.onnx>
```

**Example:**
```bash
python scripts/debug/convert_onnx_to_external_data_mode.py model.onnx model_external.onnx
```

**Output:**
- `output.onnx` - Model structure
- `output.data` - External tensor data (for tensors > 128 bytes)

**Note:** ONNX library also provides similar functionality:
```bash
python -m onnx.tools.convert_model_to_external_data input.onnx output.onnx
```
