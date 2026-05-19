<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Bazel Registry Upgrade Workflow

## Why We Pin the Registry to a SHA

`.bazelrc` uses a SHA-pinned URL for the ROCm registry:

```
common --registry=https://raw.githubusercontent.com/ROCm/bazel-registry/<SHA>
```

`MODULE.bazel.lock` records the SHA256 hash of each file fetched from that URL. Because
the URL contains an immutable commit SHA, the content behind it never changes — so the
lock never goes stale.

Using `refs/heads/main` instead would mean any push to the registry immediately
invalidates the lock (same URL, different content → Bazel checksum mismatch error).

## Workflow: Upgrading the Registry

After merging a PR into `ROCm/bazel-registry` main:

### 1. Get the new SHA

```bash
cd /mnt/c/Develop/w/source/bazel-registry-1
git fetch origin
git rev-parse origin/main
# e.g. ef797daffaa786aa0c67a1e6e025e2b79774bc9c
```

### 2. Update `.bazelrc`

In `MorphiZen/.bazelrc`, replace the old SHA:

```
common --registry=https://raw.githubusercontent.com/ROCm/bazel-registry/<NEW_SHA>
```

### 3. Regenerate the lock file

```bash
bazelisk.exe build --config=remote //...
```

Bazel sees new URLs (SHA changed) → treats all registry entries as new → fetches fresh
→ writes updated hashes into `MODULE.bazel.lock` automatically. No manual intervention.

### 4. Commit both files together

```bash
git add .bazelrc MODULE.bazel.lock
git commit -m "build(deps): upgrade bazel-registry to <SHORT_SHA>"
```

## What NOT to Do

| Action | Problem |
|--------|---------|
| Use `refs/heads/main` in `.bazelrc` | Lock goes stale on every registry push |
| Edit `MODULE.bazel.lock` manually | Hashes must match actual fetched bytes |
| Delete `MODULE.bazel.lock` to force regeneration | Works but loses all other pinned hashes; only needed when URL stays the same but content changed (should not happen with SHA pinning) |
| Evict individual entries from `C:/bazel_cache/repository_cache/` | Workaround for the mutable-URL problem; unnecessary with SHA pinning |
