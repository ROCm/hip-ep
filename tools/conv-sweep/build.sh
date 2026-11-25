# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
# Build the conv sweep against one conv_kernel.hip (Windows, Git Bash, TheRock hipcc).
# usage: THEROCK=<path to _therock> [ARCH=gfx1151] bash tools/conv-sweep/build.sh [conv_kernel.hip] [out.exe]
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
K=$ROOT/lib/Runtime/Kernels
src=${1:-$K/hip/conv_kernel.hip}
out=${2:-$HERE/sweep.exe}
TR=${THEROCK:?set THEROCK to the TheRock dist folder (the one with bin/hipcc.exe)}
ARCH=${ARCH:-gfx1151}
tu=${out%.exe}_tu.hip
printf '#include "%s"\n#include "%s"\n' "$(cygpath -m "$src")" "$(cygpath -m "$HERE/sweep.hip")" > "$tu"
# -DSWEEP_SPLITK enables the workspace (split-K) entry points when the kernel file has hip_conv_ws.
HIP_CLANG_PATH=$(cygpath -m "$TR/bin") "$TR/bin/hipcc.exe" "$(cygpath -m "$tu")" -o "$(cygpath -m "$out")" \
  --offload-arch="$ARCH" -I "$(cygpath -m "$TR/include")" -I "$(cygpath -m "$K/include")" -I "$(cygpath -m "$K/hip")" \
  -fms-extensions -fms-compatibility -fexceptions \
  -Xclang -fno-cuda-host-device-constexpr -include "$(cygpath -m "$K/include/msvc_hip_cmath_workaround.h")" \
  -Wno-ignored-attributes -DNDEBUG -D_ITERATOR_DEBUG_LEVEL=0 -O2 -D_MT -Xclang --dependent-lib=libcmt \
  $(grep -q hip_conv_ws "$src" && echo -DSWEEP_SPLITK) "${@:3}"
echo "built $out"
