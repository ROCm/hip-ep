<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# ONNX Runtime patches

Patches applied on top of the upstream ONNX Runtime source tree pinned at
`ONNXRUNTIME_VERSION` (see CI env). Each patch is a single self-contained
`git format-patch` produced from a microsoft/onnxruntime PR; the upstream
PR URL is recorded in the commit message.

## Layout

Patches in this directory are applied in lexicographic order by both
`.github/workflows/windows-build.yml` (Windows ORT source checkout) and
`docker/build.sh` (Linux Docker ORT clone), via
`git -C <ort-source-dir> apply <repo>/patches/onnxruntime/*.patch`.

The pre-built binary path used by `.github/workflows/windows-build-mock.yml`
does NOT apply these patches (no source rebuild step). That CI exercises
the mock GPU runtime and does not hit any path the patches affect.

## Removing patches

Each patch should be removed from this directory once its corresponding
upstream PR lands and `ONNXRUNTIME_VERSION` is bumped to a release that
includes it. Leaving the patch in after upstream merge will cause `git
apply` to fail with "already applied" — that failure is the trigger to
delete the file.

## Current patches

- `0001-partitioning-utils-walk-implicit-input-defs.patch` — make
  `utils::MakeComputeCapability` walk `ImplicitInputDefs()` so plugin
  EPs that claim `Loop` / `If` / `Scan` as part of a fused partition
  receive the body-subgraph captures at the fused-node boundary.
  Upstream: <https://github.com/microsoft/onnxruntime/pull/28608>.
  Remove when this is in the pinned `ONNXRUNTIME_VERSION`.
