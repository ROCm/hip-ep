<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Completed Issues

Chronological archive of all completed backlog issues (newest first).

<!-- Add completed issues here as they are merged. -->

## 2026-03

| # | Author | PR | Commit | Date | Title |
|---|--------|----|--------|------|-------|
| #016 | wcy123 | #52 | TBD | 2026-03-04 | Fix OnnxToHip LIT tests — systematic coverage, no overlap |
| #015 | wcy123 | #52 | TBD | 2026-03-04 | Remove unused `hipdnn.constant_count` module attribute |
| #014 | wcy123 | #52 | TBD | 2026-03-04 | Fix inaccurate `CONSTANT-HANDLING-DESIGN.md` |
| #013 | wcy123 | #52 | TBD | 2026-03-04 | Remove redundant `hipdnn.input_ranks` / `hipdnn.output_ranks` module attributes |
| #012 | wcy123 | #33 | 35a05d2 | 2026-03-03 | Update `docs/design/` for redesigned pipeline |
| #011 | wcy123 | #33 | 35a05d2 | 2026-03-03 | Declare in-place aliasing for element-wise HIP ops in HipDstBufferizableModel |
| #010 | wcy123 | #33 | 35a05d2 | 2026-03-03 | Redesign HIP dialect memory management (supersedes #001, #004) |
| #009 | wcy123 | #33 | 35a05d2 | 2026-03-03 | Remove `hip-buffer-deallocation` pass and test infrastructure |
| #008 | wcy123 | #33 | 35a05d2 | 2026-03-03 | Remove no-op `reconcile-unrealized-casts` from pipeline |
| #007 | wcy123 | #33 | 35a05d2 | 2026-03-03 | Add `hip-add-context-arg` pass to thread `%ctx` into ONNX functions |
| #006 | wcy123 | #33 | 35a05d2 | 2026-03-03 | Update LIT tests for ins/outs format and tensor-first pipeline |
| #005 | wcy123 | #33 | 35a05d2 | 2026-03-03 | Tensor-first OnnxToHip + one-shot-bufferize in pipeline |
| #004 | wcy123 | #33 | 35a05d2 | 2026-03-03 | Add `hip.get_pool` op + redesign MemoryPoolingPass *(superseded by #010)* |
| #003 | wcy123 | #33 | 35a05d2 | 2026-03-03 | Add HipDstBufferizableModel for one-shot-bufferize |
| #002 | wcy123 | #33 | 35a05d2 | 2026-03-03 | HIP compute ops: DPS ins/outs + Hip_TensorOrMemRef |
| #001 | wcy123 | #33 | 35a05d2 | 2026-03-03 | Fix MemoryPoolingPass buffer reuse *(superseded by #010)* |
