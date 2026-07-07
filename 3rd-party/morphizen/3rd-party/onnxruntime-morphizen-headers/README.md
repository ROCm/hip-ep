# ONNXRuntime VitisAI Provider Headers

This directory contains vendored header files from the ONNXRuntime VitisAI provider.

## Rationale

MorphiZen requires the VitisAI provider headers to build, but ONNXRuntime does not install these headers when built. To enable building MorphiZen without requiring the full ONNXRuntime source tree, we vendor these headers here.

## Source Information

**ONNXRuntime Repository**: https://github.com/microsoft/onnxruntime
**Source Path**: `onnxruntime/core/providers/vitisai/include/morphizen/`
**Commit Hash**: bf267a61cc3921746c74fa368cf7471789e74b34
**Commit Date**: 2026-01-11 16:16:47 +0000
**Commit Message**: skip eigen sha1 check

## Files

The following headers are vendored from ONNXRuntime:

### Main Headers (morphizen/)
- `_sanity_check.h` - Sanity checks for MorphiZen
- `custom_op.h` - Custom operator support
- `dll_safe.h` - DLL-safe data structures
- `export.h` - DLL export macros
- `my_ort.h` - ORT API wrappers
- `morphizen_gsl.h` - Guidelines Support Library types
- `morphizen_ort_api.h` - Main MorphiZen ORT API header


## Updating Headers

If you need to update these headers to a newer ONNXRuntime version:

1. Navigate to the ONNXRuntime repository:
   ```bash
   cd ../onnxruntime
   git pull
   ```

2. Copy the updated headers:
   ```bash
   cp onnxruntime/core/providers/vitisai/include/morphizen/*.h \
      ../Morphizen/3rd-party/onnxruntime-morphizen-headers/morphizen/

   ```

3. Update this README with the new commit information:
   ```bash
   git log -1 --format="%H%n%ci%n%s"
   ```

4. Test the build to ensure compatibility:
   ```bash
   cd ../Morphizen
   cmake -G Ninja -B ../build/morphizen.ninja -S . \
     -DCMAKE_BUILD_TYPE=Debug \
     -Dmorphizen_ENABLE_UNIT_TEST=ON \
     -Dmorphizen_ENABLE_ORT_BRIDGE=ON \
     -Dmorphizen_ENABLE_MLIR_BACKEND=ON
   cmake --build "../build/morphizen.ninja" --config Debug
   ```

## License

These headers are from the ONNXRuntime project and are subject to the MIT License.
See: https://github.com/microsoft/onnxruntime/blob/main/LICENSE
