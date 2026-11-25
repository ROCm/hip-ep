#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Preserve the CP1252 attempt and invalidate only the three affected pairs."""

import json
import shutil

from prepare_baseline import REFERENCE, WORK, sha


def main():
    folder = WORK / "paired-quality"
    backup = WORK / "excluded-gsm8k-cp1252"
    assert not backup.exists(), "Preserve the correction receipt; do not run twice"
    data = json.loads((REFERENCE / "data/gsm8k.json").read_text(encoding="utf-8"))
    prefix = "\n\n".join(
        f"Question: {x['question']}\nAnswer: {x['answer']}" for x in data["fewshot"]
    )
    historical = {
        r["id"]: r["prompt"]
        for r in map(
            json.loads,
            (REFERENCE / "runs/300258/gsm8k.jsonl")
            .read_text(encoding="utf-8")
            .splitlines(),
        )
    }
    raw = [json.loads(x) for x in (folder / "gsm8k.jsonl").read_text().splitlines()]
    assert len(raw) == 256
    keep, affected = [], []
    for row in raw:
        item = data["items"][row["index"]]
        expected = prefix + "\n\nQuestion: " + item["question"] + "\nAnswer:"
        assert expected == historical[row["id"]]
        if expected == row["prompt"]:
            keep.append(row)
        else:
            assert expected.encode("utf-8").decode("cp1252") == row["prompt"]
            affected.append(
                {"variant": row["variant"], "index": row["index"], "id": row["id"]}
            )
    assert {r["index"] for r in affected} == {36, 50, 120} and len(affected) == 6
    backup.mkdir()
    originals = {}
    for name in [
        "gsm8k.jsonl",
        "gsm8k-summary.json",
        "gsm8k-protocol.json",
        "gsm8k-integrity.json",
    ]:
        shutil.copyfile(folder / name, backup / name)
        originals[name] = sha(backup / name)
        assert originals[name] == sha(folder / name)
    (folder / "gsm8k.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in keep), encoding="utf-8"
    )
    protocol = json.loads((backup / "gsm8k-protocol.json").read_text())
    protocol["source_text_encoding"] = "UTF-8"
    (folder / "gsm8k-protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    for name in ["gsm8k-summary.json", "gsm8k-integrity.json"]:
        (folder / name).unlink()
    receipt = {
        "reason": "Windows CP1252 decoded UTF-8 apostrophes incorrectly in three prompts",
        "original_sha256": originals,
        "affected_rows": affected,
        "retained_identical_prompt_rows": len(keep),
        "retained_rows_sha256": sha(folder / "gsm8k.jsonl"),
        "historical_utf8_prompts_verified": 256,
        "training_updates": 0,
    }
    (backup / "correction.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
