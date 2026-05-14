##
## Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
## Licensed under the MIT License.
##

# hipdnn-ep Linux build environment. Mirrors the apt setup in
# .github/workflows/linux-build.yml so local matches CI on ubuntu-latest.
FROM ubuntu:24.04

ARG LLVM_MAJOR=22
ARG DEBIAN_FRONTEND=noninteractive

# Layer 1: base toolchain for ORT / protobuf / flatbuffers / OGA sub-builds.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake ninja-build pkg-config \
        git ca-certificates curl wget gnupg lsb-release \
        python3 python3-pip python3-venv python3-dev \
        sudo gosu \
        zip unzip xz-utils \
    && rm -rf /var/lib/apt/lists/*

# Layer 2: LLVM/MLIR/clang/lld from apt.llvm.org (pinned via LLVM_MAJOR).
RUN wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key \
        | tee /etc/apt/trusted.gpg.d/apt.llvm.org.asc >/dev/null \
    && echo "deb http://apt.llvm.org/noble/ llvm-toolchain-noble-${LLVM_MAJOR} main" \
        > /etc/apt/sources.list.d/llvm-${LLVM_MAJOR}.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        llvm-${LLVM_MAJOR}-dev \
        llvm-${LLVM_MAJOR}-tools \
        libpolly-${LLVM_MAJOR}-dev \
        libmlir-${LLVM_MAJOR}-dev \
        mlir-${LLVM_MAJOR}-tools \
        clang-${LLVM_MAJOR} \
        lld-${LLVM_MAJOR} \
        liblld-${LLVM_MAJOR}-dev \
    && rm -rf /var/lib/apt/lists/*

# Layer 3: pip bits. ubuntu:24.04 ships cmake 3.28.3 but backend-mlir-compiler
# needs >=3.29; pip cmake lands at /usr/local/bin and wins over apt's.
RUN pip3 install --no-cache-dir --break-system-packages \
        cmake lit setuptools wheel numpy requests packaging \
    && ln -sf /usr/local/bin/lit /usr/bin/lit

# Layer 4: sccache for CI cache hits. Pinned to 0.10.0 — earlier 0.8.x only
# knows GHA cache v1 (ACTIONS_CACHE_URL) and fails on the v2-only runners
# GitHub rolled out in early 2026 (ACTIONS_RESULTS_URL).
ARG SCCACHE_VERSION=0.10.0
RUN curl -fsSL "https://github.com/mozilla/sccache/releases/download/v${SCCACHE_VERSION}/sccache-v${SCCACHE_VERSION}-x86_64-unknown-linux-musl.tar.gz" \
        | tar -xz -C /tmp \
    && mv "/tmp/sccache-v${SCCACHE_VERSION}-x86_64-unknown-linux-musl/sccache" /usr/local/bin/ \
    && rm -rf "/tmp/sccache-v${SCCACHE_VERSION}-x86_64-unknown-linux-musl" \
    && chmod +x /usr/local/bin/sccache

# Sanity: the cmake configs the build needs are actually present.
RUN test -f /usr/lib/llvm-${LLVM_MAJOR}/lib/cmake/llvm/LLVMConfig.cmake \
    && test -f /usr/lib/llvm-${LLVM_MAJOR}/lib/cmake/mlir/MLIRConfig.cmake \
    && test -f /usr/lib/llvm-${LLVM_MAJOR}/lib/cmake/lld/LLDConfig.cmake \
    && test -f /usr/lib/llvm-${LLVM_MAJOR}/lib/libPolly.a

# NOPASSWD sudo for any user the entrypoint puts into the `dev` group.
RUN echo "%dev ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/dev-nopasswd \
    && chmod 0440 /etc/sudoers.d/dev-nopasswd

ENV LLVM_MAJOR=${LLVM_MAJOR}
ENV PATH=/usr/lib/llvm-${LLVM_MAJOR}/bin:${PATH}

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

WORKDIR /work
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]
