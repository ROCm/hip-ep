##
## Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
## Licensed under the MIT License.
##

# hipdnn-ep Linux build environment. A thin isolation wrapper around the native
# driver build.py: it provides only the base toolchain. LLVM/MLIR/LLD,
# protobuf, flatbuffers, ONNX Runtime and the TheRock ROCm SDK are all resolved
# from source/download by cmake/deps.cmake at build time -- no system LLVM here.
FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive

# Base toolchain. LLVM is built from source by build.py (no apt LLVM).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake ninja-build pkg-config \
        git ca-certificates curl wget gnupg lsb-release \
        python3 python3-pip python3-venv python3-dev \
        sudo gosu \
        zip unzip xz-utils \
    && rm -rf /var/lib/apt/lists/*

# pip bits. ubuntu:24.04 ships cmake 3.28.3 but backend-mlir-compiler needs
# >=3.29; pip cmake lands at /usr/local/bin and wins over apt's.
RUN pip3 install --no-cache-dir --break-system-packages \
        cmake lit setuptools wheel numpy requests packaging \
    && ln -sf /usr/local/bin/lit /usr/bin/lit

# NOPASSWD sudo for any user the entrypoint puts into the `dev` group.
RUN echo "%dev ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/dev-nopasswd \
    && chmod 0440 /etc/sudoers.d/dev-nopasswd

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

WORKDIR /work
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]
