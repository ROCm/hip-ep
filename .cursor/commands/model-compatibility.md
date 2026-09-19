<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Model compatibility (ONNX → HIP EP analysis)

Follow the **model-compatibility** project skill (`.cursor/skills/model-compatibility/SKILL.md` in this workspace).

1. Use the path to the `.onnx` model from my message; if missing, ask for it. Do not invent or reuse paths from earlier chats unless I confirm them.
2. **Run the pipeline yourself** (do not only print instructions):
   ```powershell
   & ".cursor/skills/model-compatibility/scripts/run_ep_compatibility_check.ps1" -ModelPath "<model.onnx>"
   ```
   - Add `-HipEpPackageRoot "<path>"` when `$env:HIP_EP_PACKAGE_ROOT` is not set.
   - If you see `[HIP_EP_NOT_CONFIGURED]`, use **AskQuestion**: provide the hip-ep package path (re-run with `-HipEpPackageRoot`) or skip the dump (re-run with `-SkipDump`, which verifies nothing).
3. Read `<OutputDir>/model_compatibility_report.md` (OutputDir is auto-derived unless I pass `-OutputDir`).
4. For every **unsupported** or **partial** operator, run the diagnose playbook in `.cursor/skills/model-compatibility/diagnose.md` and tell me the constraint that blocked it, not just its name.
5. Report using `.cursor/skills/model-compatibility/report_template.md` (status display: full→supported, partial→partial, unsupported→unsupported). Include supported-instance percentage in the summary line.
