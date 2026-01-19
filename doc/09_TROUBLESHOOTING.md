# Troubleshooting Guide

## Disk Cache Debugging

When debugging compilation passes, use disk cache to inspect generated files on disk instead of in-memory mode.

### Prerequisites

The disk cache config file is located at `etc/vaip_config_disk.json`.

### Enable Disk Cache

```powershell
# Set the config file path
$env:VITISAI_EP_PROVIDER_OPTION_config_file = "C:\path\to\etc\vaip_config_disk.json"

# IMPORTANT: Disable cache hit detection to force recompilation
$env:XLNX_ENABLE_CACHE = "0"

# Run test
.\ort_integration_test.exe
```

### Key Environment Variables

| Variable | Description |
|----------|-------------|
| `VITISAI_EP_PROVIDER_OPTION_config_file` | Path to custom config file (e.g., `etc/vaip_config_disk.json`) |
| `XLNX_ENABLE_CACHE=0` | **Required for debugging**: Disables cache hit detection, forces recompilation every run |
| `XLNX_CACHE_DIR` | Override cache directory location |

### Generated Cache Files

After running with disk cache, inspect files in `rocm_cache/<hash>/`:

| File | Description |
|------|-------------|
| `context.json` | Full compilation context with meta definitions, pass events, version info |
| `rocm_conv_W.bin` | Cached convolution weights (binary) |
| `rocm_param_Y.json` | Convolution parameters |
| `rocm_subgraph_*.json` | Subgraph definitions for custom ops |

### Example context.json Structure

```json
{
  "metaDef": [
    {
      "id": "rocm_subgraph_0",
      "inputs": ["X"],
      "outputs": ["Y"],
      "nodes": ["Y"],
      "device": "ROCm_EP",
      "param": {
        "nodes": [
          {
            "params": {
              "convParams": {
                "inWidth": "8",
                "outChannels": "16",
                "filterHeight": "3",
                "weightFilePath": "rocm_conv_W.bin"
              },
              "opType": "conv"
            }
          }
        ]
      }
    }
  ],
  "config": {
    "passes": [...],
    "cacheDir": "rocm_cache",
    "cacheKey": "<hash>"
  },
  "events": [
    {
      "name": "morphizen-level2-pass-rocm-conv",
      "dur": "1748"
    }
  ],
  "cacheFiles": [
    "rocm_conv_W.bin",
    "rocm_param_Y.json",
    "context.json"
  ]
}
```

### Complete Debugging Example

```powershell
# Set up environment
$env:MORPHIZEN_VITISAI_EP_ENABLE_CPU_DEVICE = "1"
$env:THEROCK_DIST = "C:\Develop\m\dist\therock"
$env:PATH = "$env:THEROCK_DIST\bin;$env:PATH"

# Enable disk cache
$env:VITISAI_EP_PROVIDER_OPTION_config_file = "C:\Develop\m\Source\morphizen-rocm\etc\vaip_config_disk.json"
$env:XLNX_ENABLE_CACHE = "0"

# Enable debug logging
$env:MORPHIZEN_DEBUG_ROCM = "2"
$env:GLOG_logtostderr = "1"

# Run test
cd C:\Develop\m\build\morphizen-rocm\bin
.\ort_integration_test.exe --gtest_filter=*VitisAI*

# Inspect generated files
Get-ChildItem -Recurse rocm_cache
Get-Content rocm_cache\*\context.json
```

## Common Issues

### Cache Hit Skipping Recompilation

**Symptom**: Changes to passes not taking effect

**Solution**: Set `XLNX_ENABLE_CACHE=0` to force recompilation:
```powershell
$env:XLNX_ENABLE_CACHE = "0"
```

### In-Memory Mode Active

**Symptom**: No `rocm_cache` directory created, log shows "skip update cache dir: in-mem mode"

**Solution**: Use the disk cache config file:
```powershell
$env:VITISAI_EP_PROVIDER_OPTION_config_file = "path/to/vaip_config_disk.json"
```

### Pass Not Executing

**Symptom**: Pass events missing from context.json

**Solution**: Enable debug logging to see pass execution:
```powershell
$env:MORPHIZEN_DEBUG_ROCM = "2"
$env:GLOG_logtostderr = "1"
