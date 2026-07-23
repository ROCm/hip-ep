<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
---
name: scp-remote-model
description: >-
  Builds a PowerShell scp -r command to copy an ONNX/model directory from a
  remote AMD Windows machine into ~/workspace/models on this PC. Use when the
  user wants to copy, fetch, or scp a model from another machine (halo47,
  halo45, xconucstr*, remote path, rocm_*, models folder).
disable-model-invocation: true
---

# Copy remote model via scp

Prepare a **ready-to-run PowerShell** `scp` command. Do **not** run it unless the user explicitly asks to execute the transfer (models are large).

## Local destination

Always target:

```powershell
$env:USERPROFILE\workspace\models\
```

- Create the folder if missing: `New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\workspace\models"`
- Default remote folder name: use the **last segment** of the remote path (e.g. `.../stabilityai-stable-diffusion-3.5-medium-onnx-cpu` → copy into `...\models\stabilityai-stable-diffusion-3.5-medium-onnx-cpu`).
- If the user wants a different local name, append that name under `models\`.

## Information to collect

1. **Remote machine** — short id (e.g. `halo47`) or full host `xconucstrhalo47.amd.com`.
2. **Remote path** — absolute Windows path to the **model directory** on that machine (e.g. `C:/Users/mounikk/rocm_mounik/models/<model-dir>`).

If host is only a short id, expand to: `xconucstr<shortid>.amd.com` (lowercase id, no domain suffix on the id).

## SSH known_hosts file

Use a **dedicated** known_hosts file per remote box (avoids polluting the default file):

```powershell
-o UserKnownHostsFile="$env:USERPROFILE\.ssh\known_hosts_<shortid>"
```

`<shortid>` is the machine label after `xconucstr` (e.g. `halo47` → `known_hosts_halo47`). If the user already names a file, use theirs.

## Command template (reference — verified working)

Use **forward slashes** on the remote Windows path inside the scp source. Quote both source and destination.

```powershell
scp -r -o UserKnownHostsFile="$env:USERPROFILE\.ssh\known_hosts_halo47" `
  "xconucstrhalo47.amd.com:C:/Users/mounikk/rocm_mounik/models/stabilityai-stable-diffusion-3.5-medium-onnx-cpu" `
  "$env:USERPROFILE\workspace\models\"
```

Substitute host, known_hosts filename, remote path, and destination only.

## Agent workflow

1. Confirm remote host and full remote directory path; infer local folder name if not specified.
2. Ensure `~\workspace\models` exists (suggest `New-Item` once if needed).
3. Output the full multi-line PowerShell command (backtick continuations as above).
4. Briefly note: first connection may prompt to trust the host (writes into the dedicated `known_hosts_*` file); transfer may take a long time.
5. Do **not** run `scp` unless the user asks to run it.

## Path rules

| Side | Format |
|------|--------|
| Remote source in scp | `host: C:/Users/.../model-dir` (forward slashes after drive letter) |
| Local destination | `$env:USERPROFILE\workspace\models\` or `...\models\<name>\` |
| Remote path from user | Accept `C:\Users\...` input; normalize to `C:/Users/...` in the scp string |

## Optional one-liner variant

If the user prefers a single line, join without backticks; keep the same `-o UserKnownHostsFile` and quoted paths.
