#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""CPU-only fixture controls for the mini-SWE transport and official action parser."""
# ruff: noqa: E402

import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "_local/qsa/benchmarks"
SOURCE = WORK / "mini-swe-agent"
os.environ["MSWEA_GLOBAL_CONFIG_DIR"] = str(WORK / "mini-config")
os.environ["MSWEA_SILENT_STARTUP"] = "1"
sys.path.insert(0, str(SOURCE / "src"))

from mini_native_model import NativeMiniModel
from minisweagent.exceptions import FormatError


def main():
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=SOURCE, text=True
    ).strip()
    assert revision == "ce91a8c6449bc5a686df8f6b7ee8ca9433d893bc"
    queue = WORK / ("mini-model-control-" + uuid.uuid4().hex)
    queue.mkdir()
    (queue / "ready.json").write_text('{"fixture_only":true}\n')
    completed = []

    def fixture_response():
        for text in [
            "THOUGHT: inspect\n```mswea_bash_command\npwd\n```",
            "```mswea_bash_command\npwd\n```\n```mswea_bash_command\nls\n```",
        ]:
            deadline = time.monotonic() + 20
            pending = []
            while not pending:
                assert time.monotonic() < deadline
                pending = list(queue.glob("*.request.json"))
                time.sleep(0.05)
            path = pending[0]
            request = json.loads(path.read_text())
            response = {
                "status": "complete",
                "variant": request["variant"],
                "text": text,
                "prompt_sha256": hashlib.sha256(
                    request["prompt"].encode("utf-8")
                ).hexdigest(),
            }
            target = queue / (request["request_id"] + ".response.json")
            temporary = target.with_suffix(".tmp")
            temporary.write_text(json.dumps(response))
            path.rename(path.with_suffix(".done"))
            temporary.replace(target)
            completed.append(request["request_id"])

    producer = threading.Thread(target=fixture_response, daemon=True)
    producer.start()
    model = NativeMiniModel(
        model_name="HipEP-OLMo3-Original",
        queue=queue,
        model_kwargs={"temperature": 0, "max_tokens": 4096},
    )
    messages = [
        model.format_message(
            role="user", content="Protocol fixture; no command is executed."
        )
    ]
    response = model.query(messages)
    assert response["extra"]["actions"] == [{"command": "pwd"}]
    try:
        model.query(messages)
    except FormatError:
        pass
    else:
        raise AssertionError("Official parser must reject multiple actions")
    producer.join(timeout=20)
    assert not producer.is_alive() and len(completed) == 2
    observation = model.format_observation_messages(
        response, [{"returncode": 0, "output": "fixture output"}]
    )
    assert (
        observation[0]["role"] == "user"
        and "fixture output" in observation[0]["content"]
    )
    assert (
        model.serialize()["info"]["config"]["model"]["model_name"]
        == "HipEP-OLMo3-Original"
    )
    receipt = {
        "status": "passed",
        "mini_swe_revision": revision,
        "fixture_model_responses": 2,
        "real_model_inferences": 0,
        "shell_commands_executed": 0,
        "swebench_predictions": 0,
        "single_action_accepted": True,
        "multiple_actions_rejected": True,
        "fixture_queue": str(queue),
        "packages": {
            p: importlib.metadata.version(p)
            for p in ["pydantic", "jinja2", "rich", "platformdirs"]
        },
    }
    (WORK / "mini-model-controls.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
