#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Bounded, resumable HTTPS fallback for stalled large-file Hub transfers."""

import shutil
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
from huggingface_hub import HfApi, hf_hub_download, hf_hub_url
from prepare_baseline import MODEL, REVISION, WORK, sha


def download(item):
    folder = WORK / "base-model"
    target = folder / item.rfilename
    expected = item.lfs.sha256
    if (
        target.exists()
        and target.stat().st_size == item.size
        and sha(target) == expected
    ):
        print("Verified", target.name, flush=True)
        return
    partial = target.with_suffix(target.suffix + ".range.part")
    if not partial.exists():
        # Reuse a contiguous prefix from the stopped Hub downloader. Final
        # SHA256 validation protects against a damaged or mismatched prefix.
        candidates = list(
            (folder / ".cache/huggingface/download").glob(f"*.{expected}.incomplete")
        )
        if candidates:
            shutil.copyfile(candidates[0], partial)
    offset = partial.stat().st_size if partial.exists() else 0
    with (
        httpx.Client(
            follow_redirects=True, timeout=httpx.Timeout(60, read=30)
        ) as client,
        partial.open("ab") as stream,
    ):
        while offset < item.size:
            end = min(item.size, offset + (32 << 20)) - 1
            url = hf_hub_url(MODEL, item.rfilename, revision=REVISION)
            for attempt in range(4):
                try:
                    response = client.get(
                        url,
                        params={"range_start": offset},
                        headers={"Range": f"bytes={offset}-{end}"},
                    )
                    response.raise_for_status()
                    if (
                        response.status_code != 206
                        or response.headers.get("content-range")
                        != f"bytes {offset}-{end}/{item.size}"
                    ):
                        raise RuntimeError("Server did not honor requested range")
                    if len(response.content) != end - offset + 1:
                        raise RuntimeError("Truncated range response")
                    break
                except (httpx.HTTPError, RuntimeError):
                    if attempt == 3:
                        raise
                    time.sleep(1)
            stream.write(response.content)
            stream.flush()
            offset = end + 1
            print(f"{item.rfilename}: {offset}/{item.size}", flush=True)
    actual = sha(partial)
    if actual != expected:
        raise RuntimeError(
            f"SHA256 mismatch for {item.rfilename}: {actual} != {expected}"
        )
    partial.replace(target)
    print("Verified", target.name, expected, flush=True)


def main():
    info = HfApi().model_info(MODEL, revision=REVISION, files_metadata=True)
    assert info.sha == REVISION
    large = [s for s in info.siblings if s.rfilename.endswith(".safetensors")]
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(download, large))
    for item in info.siblings:
        if item.rfilename.endswith((".json", ".txt", ".model")):
            hf_hub_download(
                MODEL, item.rfilename, revision=REVISION, local_dir=WORK / "base-model"
            )
    print("All pinned model assets verified", flush=True)


if __name__ == "__main__":
    main()
