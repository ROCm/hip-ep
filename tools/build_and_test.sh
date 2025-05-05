##
## Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
## Licensed under the MIT License.
##
set -e

cmake -DBUILD_SHARED_LIBS=OFF \
    "-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded$<$<CONFIG:Debug>:Debug>" \
    -S "$VAI_RT_WORKSPACE/morphizen-demo" -B "$VAI_RT_BUILD_DIR/morphizen-demo" \
    "-DCMAKE_INSTALL_PREFIX=$VAI_RT_PREFIX" \
    "-DCMAKE_PREFIX_PATH=$VAI_RT_PREFIX" \
    --fresh

jobs=$(nproc)

cmake  --build  "$VAI_RT_BUILD_DIR/morphizen-demo" -j $jobs --target install

export LD_LIBRARY_PATH=$VAI_RT_PREFIX/lib

ctest -j $jobs --test-dir "$VAI_RT_BUILD_DIR/morphizen-demo" -C Debug --output-on-failure
