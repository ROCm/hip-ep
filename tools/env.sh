##
## Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
## Licensed under the MIT License.
##
export C=$(dirname $(realpath --no-symlinks ${BASH_SOURCE[0]}) | sed 's?/c/?C:/?g')
export CW=$(dirname $(realpath  ${BASH_SOURCE[0]}) | sed 's?/c/?C:/?g')
export BUILD_TYPE=debug
export W=$CW/source
export EN_VAIML=ON
export WIN24_BUILD=ON
export VAI_RT_WORKSPACE=$(realpath $CW/../..)
export VAI_RT_BUILD_DIR=$C/build/${BUILD_TYPE}
export VAI_RT_PREFIX=$C/local/${BUILD_TYPE}
export VAI_RT_GENERATE_CMAKE_PRESET=ON
export BUILD=$VAI_RT_BUILD_DIR
export PREFIX=$VAI_RT_PREFIX
mkdir -p $VAI_RT_PREFIX
mkdir -p $VAI_RT_BUILD_DIR
mkdir -p $VAI_RT_WORKSPACE
cd $VAI_RT_WORKSPACE

function vaip_banner() {

    cat <<EOF
Welcome to VAIP dev environment

  VAI_RT_WORKSPACE=$VAI_RT_WORKSPACE
  VAI_RT_BUILD_DIR=$VAI_RT_BUILD_DIR
  VAI_RT_PREFIX=$VAI_RT_PREFIX
  W=$W
  BUILD=$BUILD
  PREFIX=$PREFIX

to build whole project from scratch

   % python \$W/vai-rt/main.py --dev-mode --release_file=\$W/vai-rt/release_file/latest_stx.txt --type ${BUILD_TYPE}

to build VAIP

   % python \$W/vai-rt/main.py --dev-mode --type ${BUILD_TYPE} --project vaip

to rebuild onnxruntime

   % python \$W/vai-rt/main.py --dev-mode --type ${BUILD_TYPE} --project onnxruntime

to show this help again

   % vaip_banner

EOF
}

vaip_banner

cat <<EOF >$C/env.txt
VAI_RT_WORKSPACE=$VAI_RT_WORKSPACE
VAI_RT_BUILD_DIR=$VAI_RT_BUILD_DIR
VAI_RT_PREFIX=$VAI_RT_PREFIX
W=$W
BUILD=$BUILD
PREFIX=$PREFIX
EOF
