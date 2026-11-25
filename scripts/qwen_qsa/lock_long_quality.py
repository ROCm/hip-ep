#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Lock natural 16K/32K quality inputs before any native Qwen selector tuning."""

import hashlib
import io
import json
import random
import time
import urllib.request

import zstandard
from huggingface_hub import HfApi, hf_hub_url
from tokenizers import Tokenizer
from verify_handoff import SOURCE, WORK, sha

REPO = "allenai/dolma3_mix-150B-1025"
REVISION = "afa92bfb22366821c5e6cd427cdd036b34b713ef"
SEED = 20261124


class ShardLimit(Exception):
    pass


class BoundedReader:
    def __init__(self, response):
        self.response, self.count = response, 0

    def read(self, size=-1):
        remaining = 32 * 2**20 - self.count
        if remaining <= 0:
            raise ShardLimit("32 MiB compressed shard read limit")
        data = self.response.read(min(remaining, size if size >= 0 else remaining))
        self.count += len(data)
        return data

    def close(self):
        self.response.close()


def fingerprints(ids):
    if len(ids) < 16:
        return
    mask, base = (1 << 64) - 1, 1000003
    factor, value = pow(base, 15, 1 << 64), 0
    for token in ids[:16]:
        value = (value * base + token + 1) & mask
    yield value
    for i in range(16, len(ids)):
        value = ((value - (ids[i - 16] + 1) * factor) * base + ids[i] + 1) & mask
        yield value


def main():
    output = WORK / "long-quality-v1"
    assert not output.exists(), (
        "Retain locked or failed preparation evidence; choose a new explicit version"
    )
    output.mkdir(parents=True)
    verified = json.loads((WORK / "handoff-verification.json").read_text())
    from pathlib import Path

    snapshot = Path(verified["backbone"]["snapshot"])
    tokenizer_path = snapshot / "tokenizer.json"
    source_manifest = json.loads(
        (SOURCE / "data/qwen-selector-v1/manifest.json").read_text()
    )
    assert sha(tokenizer_path) == source_manifest["tokenizer_file_sha256"]
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    excluded_shards = {s["path"] for s in source_manifest["source_shards"]}
    excluded_documents = set()
    for name in ["train.json", "validation.json", "test.json"]:
        path = SOURCE / "data/qwen-selector-v1" / name
        assert sha(path) == verified["data_sha256"][name]
        excluded_documents.update(
            row["document_sha256"] for row in json.loads(path.read_text())
        )
    gsm = json.loads((SOURCE / "data/qwen-selector-v1/gsm8k.json").read_text())
    blocked = {
        f
        for row in gsm
        for f in fingerprints(
            tokenizer.encode(row["question"], add_special_tokens=False).ids
        )
    }
    files = sorted(
        name
        for name in HfApi().list_repo_files(
            REPO, repo_type="dataset", revision=REVISION
        )
        if name.startswith("data/olmocr_science_pdfs-")
        and name.endswith(".jsonl.zst")
        and name not in excluded_shards
    )
    random.Random(SEED).shuffle(files)
    started = time.monotonic()
    selected, attempts, seen = [], [], set(excluded_documents)
    for name in files[:8]:
        attempt = {"path": name, "accepted": 0, "scanned": 0}
        url = hf_hub_url(REPO, name, repo_type="dataset", revision=REVISION)
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                bounded = BoundedReader(response)
                with zstandard.ZstdDecompressor().stream_reader(bounded) as reader:
                    with io.TextIOWrapper(reader, encoding="utf-8") as lines:
                        for line_number, line in enumerate(lines):
                            assert time.monotonic() - started < 900, (
                                "CPU preparation time limit; preserve partial evidence"
                            )
                            if line_number >= 2000:
                                break
                            attempt["scanned"] += 1
                            row = json.loads(line)
                            text = row.get("text", "")
                            if (
                                not isinstance(text, str)
                                or not 100000 <= len(text) <= 4000000
                                or "[REMOVED]" in text
                            ):
                                continue
                            digest = hashlib.sha256(text.encode()).hexdigest()
                            if digest in seen:
                                continue
                            tokens = tokenizer.encode(
                                text, add_special_tokens=False
                            ).ids
                            length = 32768 + 128
                            if len(tokens) < length:
                                continue
                            offset = int(
                                hashlib.sha256(
                                    (str(SEED) + digest).encode()
                                ).hexdigest()[:16],
                                16,
                            ) % (len(tokens) - length + 1)
                            ids = tokens[offset : offset + length]
                            if any(f in blocked for f in fingerprints(ids)):
                                continue
                            selected.append(
                                {
                                    "document_sha256": digest,
                                    "source_file": name,
                                    "source_line": line_number,
                                    "source_document_id": str(row.get("id", digest)),
                                    "source_document_tokens": len(tokens),
                                    "window_start": offset,
                                    "input_ids": ids,
                                    "window_sha256": hashlib.sha256(
                                        json.dumps(ids).encode()
                                    ).hexdigest(),
                                }
                            )
                            seen.add(digest)
                            (output / "documents.partial.json").write_text(
                                json.dumps(selected, ensure_ascii=False) + "\n",
                                encoding="utf-8",
                            )
                            attempt["accepted"] = 1
                            break
                attempt["compressed_bytes_read"] = bounded.count
        except ShardLimit as error:
            attempt["limit"] = str(error)
        attempts.append(attempt)
        (output / "preparation.json").write_text(
            json.dumps({"attempts": attempts, "selected": len(selected)}, indent=2)
            + "\n"
        )
        print(
            {"shard": name, "selected": len(selected), "scanned": attempt["scanned"]},
            flush=True,
        )
        if len(selected) == 4:
            break
    assert len(selected) == 4, (
        "Insufficient documents within declared bounded preparation; partial evidence retained"
    )
    path = output / "documents.json"
    path.write_text(json.dumps(selected, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest = {
        "state": "locked",
        "dataset": REPO,
        "revision": REVISION,
        "seed": SEED,
        "documents": 4,
        "contexts": [16384, 32768],
        "suffix_scored_tokens": 1023,
        "input_tokens_per_document": 32896,
        "reserve_tokens": 128,
        "tokenizer_sha256": sha(tokenizer_path),
        "documents_sha256": sha(path),
        "source_parameter_data_manifest_sha256": sha(
            SOURCE / "data/qwen-selector-v1/manifest.json"
        ),
        "excluded_prior_qwen_shards": sorted(excluded_shards),
        "excluded_prior_qwen_document_hashes": sorted(excluded_documents),
        "screening": "Exact document disjointness from all Qwen selector splits; no shared 16-token fingerprints with the 128 GSM questions. No semantic contamination guarantee.",
        "scope": "Four natural science/PDF documents, fixed seed and bounded shard scan; not corpus representative. Two nested prefix lengths per document; score the final 1023 next-token targets at each length.",
        "locked_before_native_qsa_integration": True,
        "training_updates": 0,
        "attempts": attempts,
        "script_sha256": sha(Path(__file__)),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {"state": "locked", "documents": 4, "sha256": manifest["documents_sha256"]}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
