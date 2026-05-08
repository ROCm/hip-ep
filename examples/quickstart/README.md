<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->

# Quick start example

This subdirectory holds the toy ONNX model used by the end-to-end
walkthrough in
[`docs/quick_start.md`](../../docs/quick_start.md#end-to-end-toy-model-walkthrough).
It is intentionally minimal: a single Conv layer that loads, JIT-compiles,
and runs in seconds, so a fresh build can be smoke-tested through the
MorphiZen Execution Provider in well under a minute.

## Running the example

The full sequence (clone, build, generate model, run inference) lives in
the quick-start guide. The two model-specific commands are summarized
here for convenience.

### 1. Generate the model

From the repository root, with the Python venv active:

```powershell
cd .\examples\quickstart
python .\gen_conv_only.py --output conv_hybrid.onnx `
    --batch 1 --in-channels 16 --out-channels 16 `
    --height 16 --width 16 --kernel 3 --pad 1 --stride 1
cd ..\..
```

This writes two files alongside the script:

| File | Format | Notes |
|------|--------|-------|
| `conv_hybrid.onnx` | ONNX | Model graph plus initializer weights. |
| `conv_hybrid_weights.npy` | NumPy | Weight tensor saved separately for reference comparisons. |

Both are excluded from version control via [`.gitignore`](./.gitignore);
generate them on demand rather than committing them.

### 2. Run it through the MorphiZen plugin EP

```powershell
$bin = "C:\Users\<you>\ROCmEP\OnnxHipDNN\prebuilt-local\bin"
& "$bin\onnxruntime_perf_test.exe" `
    .\examples\quickstart\conv_hybrid.onnx `
    --plugin_ep_libs    "MorphiZenExecutionProvider|$bin\onnxruntime_morphizen_ep.dll" `
    --plugin_eps        "MorphiZenExecutionProvider" `
    --plugin_ep_options "config_file|$bin\morphizen_config.json" `
    -t 60 -c 1 -s -I
```

See [`docs/quick_start.md`](../../docs/quick_start.md) for an explanation
of every flag, the expected output, and troubleshooting tips.

## Tweaking the model

`gen_conv_only.py` accepts CLI overrides for batch size, channels,
spatial dimensions, kernel size, padding, and stride. The defaults
reproduce the YOLOv8x backbone 3x3 stage-4 conv (4.72M parameters,
~7.5 GFLOPs) -- useful when you want a longer measurement window. Run
`python gen_conv_only.py --help` for the full flag list.
