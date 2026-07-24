<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Copy remote model via scp

Follow the **scp-remote-model** project skill (`.cursor/skills/scp-remote-model/SKILL.md` in this workspace).

1. Ask for remote host (e.g. `halo47` → `xconucstrhalo47.amd.com`) and the absolute Windows path to the model directory on that machine.
2. Output a ready-to-run **PowerShell** `scp -r` command that copies into `$env:USERPROFILE\workspace\models\`, using a dedicated `-o UserKnownHostsFile="$env:USERPROFILE\.ssh\known_hosts_<shortid>"` when appropriate.
3. Use forward slashes in the remote scp path (`C:/Users/...`).
4. Do **not** run `scp` unless I explicitly ask you to execute the transfer.
