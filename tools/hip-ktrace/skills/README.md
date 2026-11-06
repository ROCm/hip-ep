<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# ktrace agent skills

Each subdirectory contains an agent skill that follows the standard `SKILL.md`
convention supported by AI coding assistants such as Cursor, Claude Code, and
Codex. The YAML front matter (`name`, `description`) is what the assistant reads
to decide when to load the full instructions.

| Skill | Use it to |
| --- | --- |
| [`ktrace-profiling`](ktrace-profiling/SKILL.md) | Profile a HIP workload on Windows: rank kernels with a ktrace `system` capture, then capture one kernel with RGP and analyse it per instruction with rgpparse. |
| [`roctx-pipeline-stages`](roctx-pipeline-stages/SKILL.md) | Add ROCTx ranges that name the pipeline stage (embeddings, prefill, decode, lm_head) in an onnxruntime-genai model, and read the per-stage breakdown out of a ktrace capture. |

## Installing

Copy a skill directory into the skills folder recognized by your assistant,
either at the user level (available in every project) or at the workspace level
(scoped to one project):

| Assistant | User level | Workspace level |
| --- | --- | --- |
| Cursor | `~/.cursor/skills/<name>/` | `<project>/.cursor/skills/<name>/` |
| Claude Code | `~/.claude/skills/<name>/` | `<project>/.claude/skills/<name>/` |

Other assistants that support `SKILL.md` document their own skills folder.

For example, in PowerShell from this directory:

```powershell
Copy-Item -Recurse ktrace-profiling "$env:USERPROFILE\.cursor\skills\"
```

Release packages also carry these skills under `.claude/skills/`, so an
assistant working inside an extracted ktrace package discovers them without
any copy step.
