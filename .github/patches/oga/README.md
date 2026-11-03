<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# OGA PR patches

CI builds ONNX Runtime GenAI (OGA) from the `v<OGA_VERSION>` tag and applies
the microsoft/onnxruntime-genai pull requests listed in `OGA_PR_PATCHES`, in
that order. Each `<n>.patch` here is the `pull/<n>.diff` of that PR, stored at
the head recorded below.

The patches are stored rather than downloaded at build time because an open PR
keeps changing. The OGA cache key carries a hash of the listed patches, so a
cold rebuild applies exactly the bytes that produced the cached build, and a
new push to an upstream PR changes nothing until a patch here is refreshed.

| PR | Head | Title |
|---|---|---|
| [2685](https://github.com/microsoft/onnxruntime-genai/pull/2685) | `ef8d5a4bccc1865b8b47fc6657024f07f22607d3` | Support Gemma 4 12B Unified native multimodal generation |
| [2691](https://github.com/microsoft/onnxruntime-genai/pull/2691) | `b806448763ea151f50b2f3475c13a60d465c926b` | Enable continuous decoding for AMDGPU |
| [2705](https://github.com/microsoft/onnxruntime-genai/pull/2705) | `6420d2f45184a3239ebc339bf4352e38377a8c7e` | Allow opting out of AMDGPU graph capture via provider option |
| [2712](https://github.com/microsoft/onnxruntime-genai/pull/2712) | `0a0d14c4e3e7cc9d25d1f0177589d509925f74de` | Update CI to ORT 1.31.0 and CUDA EP 0.3.0 |
| [2792](https://github.com/microsoft/onnxruntime-genai/pull/2792) | `01391367fc909dc53c42acb53a82223efe54a4c8` | [Do not merge] Pin onnxruntime-extensions to main plus #1136 |
| [2801](https://github.com/microsoft/onnxruntime-genai/pull/2801) | `874b3fd5a59facb6a372346d857b23f72a8e06d4` | [Do not merge] Qwen-3.8 Flash Next runtime on v0.17.0 |

## Adding or refreshing a patch

1. Download the diff and check that the PR head is the one you meant to take:

   ```bash
   curl -fsSL "https://github.com/microsoft/onnxruntime-genai/pull/<n>.diff" \
     -o .github/patches/oga/<n>.patch
   ```

2. Check that every listed patch still applies to the tag, in order, with the
   command CI uses:

   ```bash
   git clone --branch v<OGA_VERSION> https://github.com/microsoft/onnxruntime-genai.git
   cd onnxruntime-genai
   for pr in <OGA_PR_PATCHES>; do
     git apply --whitespace=nowarn --3way --exclude='test/*' \
       --exclude='.github/*' --exclude='*.md' "../hip-ep/.github/patches/oga/${pr}.patch"
   done
   ```

3. Update the table above, and `OGA_PR_PATCHES` in every workflow that sets it
   when a PR is added or removed. Remove a patch once its PR lands in the
   pinned OGA release.

Do not edit a patch by hand. Its bytes are part of the cache key, and
`.gitattributes` keeps them exact on Windows checkouts.
