<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Published QSA selector assets

The native exporters load selector-only safetensors from Hugging Face. They pin
immutable revisions, verify the weights/config/provenance file SHA256 values,
and check every BF16 tensor name, shape, offset and raw-byte hash. No repository
Python code is downloaded or executed. The registry is
[`scripts/qsa/selector_assets.json`](../../scripts/qsa/selector_assets.json).

| Configuration | Hugging Face repository | Pinned revision |
|---|---|---|
| OLMo Four | [olmo-3-1025-7b-qsa-4-selectors](https://huggingface.co/rajeevp/olmo-3-1025-7b-qsa-4-selectors) | `4fd10ef4e5dd1dca49d84491d54663980f7cc33a` |
| Qwen Four | [qwen3.8-27b-qsa-4-selectors](https://huggingface.co/rajeevp/qwen3.8-27b-qsa-4-selectors) | `6041df9d852345aaf26f0c63c304e0ef9c846074` |
| Qwen Eight | [qwen3.8-27b-qsa-8-selectors](https://huggingface.co/rajeevp/qwen3.8-27b-qsa-8-selectors) | `f59b0c17664755c1d070b3d6269bddc2452e5cda` |

## Download and verify

Use the existing QSA Python environment with `huggingface_hub`, `safetensors`,
PyTorch and ONNX. Downloading selectors does not initialize a GPU or download a
backbone model:

```text
python scripts/qsa/selector_assets.py --variant all
python scripts/qsa/selector_assets.py --variant all --offline
python scripts/qwen_qsa/verify_handoff.py --selectors-only
```

The default cache is `_local/qsa/selector-assets/<variant>/`. Set
`HIP_EP_QSA_SELECTOR_CACHE` to relocate it, or use `--cache-root` for the download
command. Variants are `olmo-four`, `qwen-four` and `qwen-eight`. Missing files in
offline mode and corrupt files fail explicitly. Imports do not download assets.
Advancing a Hub repository's `main` does not change the selected revision.

After the existing backbone exports are prepared, the native export entry points
use these assets automatically:

```text
python scripts/qsa/export_native_gqa.py
python scripts/qwen_qsa/export_native_gqa.py
python scripts/qwen_qsa/export_eight.py
```

For an additional export, give either Four exporter a fresh `--output-root`
containing its Original/Changed folders, or give Eight a fresh `--output-dir`.
Existing exports are protected from overwrite. The exporters no longer require
the private sibling project's selector checkpoints, selector handoff manifest or
Eight training protocol. Base-model exports, the native toolchain and historical
evaluation datasets remain separate prerequisites.

## Identity and historical results

All **60 BF16 tensors** match the original measured exports: OLMo12, Qwen Four16,
Qwen Eight32. Qwen Four's entire safetensors file is also byte-identical. OLMo and
Qwen Eight were repackaged from PyTorch containers, so their new file hashes
differ while every tensor byte remains exact. Fresh exports of all three
configurations preserve the measured ONNX graph and `qsa.selectors.bin` bytes.

| Configuration | Published `selectors.safetensors` SHA256 |
|---|---|
| OLMo Four | `42fdd0a611e8479fadf386ace09ef513e304dd671a60c9efe545ea8da9638aa2` |
| Qwen Four | `d7729599f421d80ace536f84145bafa9a6e1eaad22dde18ce42392014a58b5e4` |
| Qwen Eight | `360f8006ac85ca5d6f7cf85a6c964ca452e7433fb279a6aba3d0957dbe7ae4e5` |

New export receipts contain `selector_asset` with the Hub repository, revision,
actual container hash and metadata hashes. Existing `selector_sha256` fields and
OLMo's `selector_fingerprint()` retain the historical export identity for
compatibility with prior protocols; they must not be read as the new Hub file's
hash. Guards now verify the public assets before returning that identity.
Stage receipts also record `selector_file_sha256_before/after` and snapshot the
pinned registry alongside the stage's source files.

Historical raw measurements, binary identities and archived receipts are
preserved. The older boundary-prototype byte auditor still needs its original
archive when checking that archive's PyTorch container itself. Publication does
not rerun accuracy/performance, alter kernel math, or change the Eight
checkpoint's failed source32K PPL gate. See the
[migration evidence](../reproducibility/qwen-eight-hip-ep/hf-selector-migration.json)
and [measurement coverage](../reproducibility/qwen-eight-hip-ep/README.md).
