#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Validate saved native responses for exact-prompt recovery of partial cases."""

import hashlib
import json
from pathlib import Path


def load_responses(task, queues):
    """Reuse actual generations only; never choose among outputs using scores."""
    responses = {}
    protocol = json.loads((task / "protocol.json").read_text())
    for folder in queues:
        folder = Path(folder).resolve()
        assert folder.parent == (task / "queues").resolve()
        ready = json.loads((folder / "ready.json").read_text())
        assert ready["selector_sha256"] == protocol["selector_sha256"]
        assert ready["capacity"] == protocol["capacity"] == 65536
        for path in sorted(folder.glob("*.response.json")):
            result = json.loads(path.read_text(encoding="utf-8"))
            if result.get("status") != "complete":
                continue
            identifier = result["request_id"]
            request_path = folder / (identifier + ".request.done")
            if not request_path.exists():
                request_path = folder / (identifier + ".request.json")
            request = json.loads(request_path.read_text(encoding="utf-8"))
            prompt_sha = hashlib.sha256(request["prompt"].encode("utf-8")).hexdigest()
            assert result["prompt_sha256"] == prompt_sha
            assert request["variant"] == result["variant"]
            assert request["temperature"] == 0 and request["max_new_tokens"] == 4096
            assert request["stop"] == protocol["stops"]
            assert len(result["generated_token_ids"]) == result["completion_tokens"]
            key = result["variant"], prompt_sha
            source = {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            if key in responses:
                assert (
                    responses[key][0]["generated_token_ids"]
                    == result["generated_token_ids"]
                ), "Conflicting saved responses; investigate nondeterminism"
            else:
                responses[key] = result, source
    return responses
