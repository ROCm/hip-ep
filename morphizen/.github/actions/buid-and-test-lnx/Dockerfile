##
## Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
## Licensed under the MIT License.
##

# Container image that runs your code
FROM ubuntu:22.04
RUN apt-get update && apt-get install -y \
    sudo \
    g++ \
    git \
    gcc uuid-dev \
    python3 \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    python3-venv \
    git \
    curl \
    jq \
    python-is-python3 \
    emacs \
    vim \
    bash-completion \
    gdb \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install cmake==3.29.6 requests --force-reinstall
RUN cmake --version; g++ --version; python --version; git --version
WORKDIR /workspace
ARG ORT_COMMIT_ID=173a11a4e7a2f7a360c9db6abbe601a06a16f004
RUN --mount=type=secret,id=github_token  \
    echo "clone onnxruntime to commit id ${ORT_COMMIT_ID}"; git clone https://$(cat /run/secrets/github_token)@gitenterprise.xilinx.com/VitisAI/onnxruntime.git -b morphizen-ci;
RUN --mount=type=secret,id=github_token  \
    git clone https://$(cat /run/secrets/github_token)@gitenterprise.xilinx.com/VitisAI/vai-rt.git -b cp_dev

WORKDIR /workspace/onnxruntime
ARG BUILD_TYPE=Debug
RUN git checkout ${ORT_COMMIT_ID}
RUN python tools/ci_build/build.py --use_vitisai --config ${BUILD_TYPE} --build_shared_lib --parallel --compile_no_warning_as_error \
    --skip_submodule_sync --build_dir "$PWD/../build/onnxruntime"   \
    --skip_tests \
    --allow_running_as_root \
    --cmake_extra_defines \
    "CMAKE_INSTALL_PREFIX=$PWD/../local" \
    onnxruntime_BUILD_UNIT_TESTS=OFF

RUN cmake --install $PWD/../build/onnxruntime/${BUILD_TYPE}
RUN env VAI_RT_BUILD_DIR=/workspace/build VAI_RT_PREFIX=/workspace/local python ../vai-rt/main.py --type ${BUILD_TYPE} --dev-mode --release_file=../vai-rt/release_file/latest_stx.txt --project  gsl gtest protobuf glog
RUN mkdir -p /github/workspace;  \
    chmod o+rwx /workspace/build \
    /workspace/local \
    /workspace/local/lib \
    /workspace/local/share \
    /workspace/local/share/cmake \
    /workspace/local/bin \
    /workspace/local/include \
    /github/workspace


COPY entrypoint.sh /entrypoint.sh
COPY entrypoint.user.sh /entrypoint.user.sh
# Code file to execute when the docker container starts up (`entrypoint.sh`)
ENTRYPOINT ["/bin/bash","-ex", "/entrypoint.sh"]
