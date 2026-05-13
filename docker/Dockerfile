##
## Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
## Licensed under the MIT License.
##

# hipdnn-ep build environment.
# Mirrors `.github/workflows/linux-build.yml` step "Install apt build deps + LLVM 22"
# so a local build matches what CI does on ubuntu-latest.
FROM ubuntu:24.04

ARG LLVM_MAJOR=22
ARG DEBIAN_FRONTEND=noninteractive

# Layer 1: base toolchain + tools that the various sub-builds (ORT, protobuf,
# flatbuffers, OGA) reach for. Kept in one RUN so apt cache can be cleaned
# atomically and the layer stays small.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake ninja-build pkg-config \
        git ca-certificates curl wget gnupg lsb-release \
        python3 python3-pip python3-venv python3-dev \
        sudo gosu \
        zip unzip xz-utils \
    && rm -rf /var/lib/apt/lists/*

# Layer 2: LLVM/MLIR/clang/lld 22 from apt.llvm.org. Pinned LLVM_MAJOR so the
# image rebuilds when CI bumps the version.
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

# Layer 3: pip-installed bits.
#  - lit: required by `check-hip-mlir-lit`
#  - cmake: ubuntu:24.04 ships 3.28.3 but backend-mlir-compiler needs >=3.29.
#    pip cmake is staged earlier in PATH (/usr/local/bin) than apt's, so it wins.
RUN pip3 install --no-cache-dir --break-system-packages \
        cmake lit setuptools wheel numpy requests packaging \
    && ln -sf /usr/local/bin/lit /usr/bin/lit

# Layer 4: sccache for CI cache hits.
# linux-build.yml runs mozilla-actions/sccache-action on the host runner which
# sets ACTIONS_CACHE_URL + ACTIONS_RUNTIME_TOKEN + SCCACHE_GHA_ENABLED=true;
# docker/run.sh forwards these into the container so the in-container sccache
# talks to the same GitHub Actions cache backend. docker/build.sh wires
# CMAKE_*_COMPILER_LAUNCHER=sccache only when SCCACHE_GHA_ENABLED is "true"
# AND a GHA cache URL is forwarded in, so this layer is a no-op for local
# non-CI builds (the binary just sits idle).
#
# Pinned to 0.10.0 (vs the 2024-vintage 0.8.2): GitHub rolled out the
# Actions Cache v2 service in early 2026 and started exporting
# ACTIONS_RESULTS_URL instead of (or alongside) ACTIONS_CACHE_URL.
# sccache <0.9 only knows about the v1 URL and fails at startup with
# `ACTIONS_CACHE_URL not found, maybe not in github action environment?`
# when only the v2 URL is set on the runner. 0.10+ handles both URLs.
ARG SCCACHE_VERSION=0.10.0
RUN curl -fsSL "https://github.com/mozilla/sccache/releases/download/v${SCCACHE_VERSION}/sccache-v${SCCACHE_VERSION}-x86_64-unknown-linux-musl.tar.gz" \
        | tar -xz -C /tmp \
    && mv "/tmp/sccache-v${SCCACHE_VERSION}-x86_64-unknown-linux-musl/sccache" /usr/local/bin/ \
    && rm -rf "/tmp/sccache-v${SCCACHE_VERSION}-x86_64-unknown-linux-musl" \
    && chmod +x /usr/local/bin/sccache
# ^ test/lit/CMakeLists.txt does find_program(LIT_EXECUTABLE NAMES lit
#   HINTS ${python_dir}/bin NO_DEFAULT_PATH). System python lives in
#   /usr/bin, but pip puts lit at /usr/local/bin/lit. Symlink so the
#   HINT-based search hits it.

# Sanity: cmake configs the build will look for.
RUN test -f /usr/lib/llvm-${LLVM_MAJOR}/lib/cmake/llvm/LLVMConfig.cmake \
    && test -f /usr/lib/llvm-${LLVM_MAJOR}/lib/cmake/mlir/MLIRConfig.cmake \
    && test -f /usr/lib/llvm-${LLVM_MAJOR}/lib/cmake/lld/LLDConfig.cmake \
    && test -f /usr/lib/llvm-${LLVM_MAJOR}/lib/libPolly.a

# Allow any user added at runtime to use sudo without a password (entrypoint
# adds the host UID/GID to the `dev` group).
RUN echo "%dev ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/dev-nopasswd \
    && chmod 0440 /etc/sudoers.d/dev-nopasswd

ENV LLVM_MAJOR=${LLVM_MAJOR}
ENV PATH=/usr/lib/llvm-${LLVM_MAJOR}/bin:${PATH}

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

WORKDIR /work
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]
