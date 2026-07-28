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
   - Add `-VoePackageRoot "<path>"` when `$env:VOE_PACKAGE_ROOT` is not set.
   - If you see `[VOE_NOT_CONFIGURED]`, use **AskQuestion**: provide VOE path (re-run with `-VoePackageRoot`) or skip dump (re-run with `-SkipDump`).
3. Read `<OutputDir>/model_compatibility_report.md` (OutputDir is auto-derived unless I pass `-OutputDir`).
4. For every **unsupported** or **partial** operator, run the diagnose playbook in `.cursor/skills/model-compatibility/diagnose.md` before responding.
5. Report using `.cursor/skills/model-compatibility/report_template.md` (status display: full→supported, partial→partial, unsupported→unsupported). Include supported-instance percentage in the summary line.
