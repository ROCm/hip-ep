<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->

# EP Context Internal Design

The EP context is becoming more and more complex because we keep adding new features:

1. Encryption
2. Compression
3. mmap [PR #136][136]
4. Shared EP context [PR #44][44]
5. Prebuilt EP context [PR #238][238]

## Generate EP Context Model

### Non-embed Mode

Using `tar_file_` because of the [mmap feature][136]. `tar_file_` is created upon the EP context binary file; we don't need any extra operations.

### Embed Mode

Using `tar_file_` because of the [mmap feature][136]. `tar_file_` is created upon `std::tmpfile`.

`tar_file_->dump_to` is used to filling the EP context node attribute.

#### morphizen_compile_model, the producer

There are two producers:

   1. `tar_file_`, unified storage mechanism for all scenarios.
   2. `prebuilt_ep_context=1`, the PR.


#### create_ep_context, the consumer

  1. Uses `tar_file_` for all scenarios. Falls back to `mem_files_` only when tmpfile() fails.

## Deploy EP Context Model

### Non-embed Mode

  1. Uses `tar_file_` because of the [mmap feature][136].

### Embed Mode

  1. Uses `tar_file_`, depending on `cache_file_use_cache_key_prefix_`.

TODO:
  1. Add `TarFile::to_memory()` return.
  2. `MemStream` does not support write.

## Prebuilt EP Context

### Non-embed Mode

  1. Uses `tar_file_` because of the [mmap feature][136].

### Embed Mode

  1. Uses `tar_file_`. DO NOT USE IT.

## Deploy EP Context Model

### Non-embed Mode

  1. Uses `tar_file_` because of the [mmap feature][136].

### Embed Mode

  1. Uses `tar_file_` for all cache storage.

### "ep.share_ep_contexts" = 1

`cache_file_use_cache_key_prefix_ = 1`; otherwise, `cache_file_use_cache_key_prefix_ = 0`.

`cache_file_use_cache_key_prefix_` is only meaningful when `tar_file_` is in use.

`cache_file_use_cache_key_prefix_ = 1`

## Compression or Encryption is Enabled

Uses `tar_file_` for all storage scenarios. Encryption and compression are handled at serialization boundaries (when saving/loading EP context), not within tar_file_ itself.

When encryption is enabled:
- During save: tar stream is encrypted using `morphizen_encryption::aes_encryption`
- During load: encrypted stream is decrypted using `morphizen_encryption::aes_decryption`, then loaded into tar_file_

This approach keeps tar_file_ simple while supporting encryption transparently.

[136]: #136 (mmap)
[44]: #44 (shared ep context)
[238]: #238 (prebuilt ep context)
