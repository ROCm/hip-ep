<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Third-party notices

## Radeon Developer Tool Suite runtime

`build.bat` can download the Radeon Developer Tool Suite runtime from
https://gpuopen.com/rdts-windows/. `RdpCaptureApi.dll`,
`GPUPerfAPICounters-x64.dll`, `amd_comgr_3.dll`, and `amdgpu_dis.dll` remain
externally provided binary components and are not part of this project's
source code. Their use and redistribution are governed by
the terms included with the downloaded RDTS package.

## AMD COMGR header

The generated public COMGR API header is vendored at
`src/external/amd_comgr/amd_comgr.h` from ROCm's `amd-staging` branch:
https://github.com/ROCm/llvm-project/tree/amd-staging/amd/comgr.
It is distributed under the Apache License v2.0 with LLVM Exceptions; the
license text is retained in `src/external/amd_comgr/LICENSE.txt`.

## ROCTx

The ROCTx public headers and lightweight annotation implementation are adapted
from ROCm roctracer:
https://github.com/ROCm/rocm-systems/tree/develop/projects/roctracer.
ROCTx is distributed under the MIT License; the license text is retained in
`src/external/roctx/LICENSE.md`.

The rocprofv3 exporter in `python/rocprofv3_export.py`, together with the
schema assertions, constants, and fixtures under `tests/`, is copied or
adapted from the ROCm 7.2.1 `rocprofiler-sdk` project:

- `source/lib/output/generateCSV.cpp`, `csv.hpp`, and `generateJSON.cpp`
- `source/include/rocprofiler-sdk/cxx/serialization/save.hpp`
- `source/lib/output/stream_info.hpp`, `agent_info.hpp`, and
  `kernel_symbol_info.hpp`
- `source/include/rocprofiler-sdk/fwd.h`
- `tests/rocprofv3/tracing/validate.py`
- `tests/rocprofv3/memory-allocation/validate.py`

Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.

SPDX-License-Identifier: MIT

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## RapidJSON

The native post-processing tools use RapidJSON 1.1.0, which CMake downloads
from https://github.com/Tencent/rapidjson at configure time; it is not stored in
this repository. RapidJSON is distributed under the MIT License; release
packages include its complete license text as `licenses/RapidJSON-LICENSE.txt`.

## LLVM llvm-cxxfilt

Windows release packages may include `llvm-cxxfilt.exe` from LLVM 22.1.0 for
kernel-name demangling. LLVM is distributed under the Apache License v2.0 with
LLVM Exceptions; the complete license is retained in
`src/external/llvm/LICENSE.txt`.
