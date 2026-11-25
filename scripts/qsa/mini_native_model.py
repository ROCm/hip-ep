#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Native model transport for pinned mini-SWE-agent's text-action interface."""

import hashlib
import json
import time
import uuid
from pathlib import Path

from minisweagent.models import GLOBAL_MODEL_STATS
from minisweagent.models.utils.actions_text import (
    format_observation_messages,
    parse_regex_actions,
)
from pydantic import BaseModel, Field


class NativeMiniConfig(BaseModel):
    model_name: str
    queue: Path
    model_kwargs: dict = Field(default_factory=dict)
    action_regex: str = r"```mswea_bash_command\s*\n(.*?)\n```"
    format_error_template: str = (
        "Expected exactly one bash action; found {{actions|length}}."
    )
    observation_template: str = "<returncode>{{output.returncode}}</returncode>\n<output>\n{{output.output}}</output>"
    multimodal_regex: str = ""


class NativeMiniModel:
    def __init__(self, **kwargs):
        self.config = NativeMiniConfig(**kwargs)
        assert self.config.model_name in ["HipEP-OLMo3-Original", "HipEP-OLMo3-Changed"]
        self.variant = self.config.model_name.rsplit("-", 1)[1]
        assert self.config.model_kwargs.get("temperature", 0) == 0
        assert self.config.model_kwargs.get("top_p", 1) == 1
        assert not self.config.multimodal_regex, (
            "This adapter handles the text-only Verified tasks"
        )

    def query(self, messages, **kwargs):
        assert not kwargs, "Set fixed generation options in the model configuration"
        parts = []
        for message in messages:
            assert isinstance(message["content"], str)
            parts.append(message["role"].capitalize() + ":\n" + message["content"])
        prompt = "\n\n".join(parts) + "\n\nAssistant:\n"
        identifier = uuid.uuid4().hex
        request = {
            "request_id": identifier,
            "variant": self.variant,
            "prompt": prompt,
            "max_new_tokens": self.config.model_kwargs.get("max_tokens", 4096),
            "temperature": 0,
            "stop": ["\n\nUser:", "\n\nSystem:", "\n\nTool:"],
        }
        assert 0 < request["max_new_tokens"] <= 4096
        queue = self.config.queue
        assert (queue / "ready.json").exists(), (
            "Start the native generation worker first"
        )
        temporary = queue / (identifier + ".request.tmp")
        temporary.write_text(
            json.dumps(request, ensure_ascii=True) + "\n", encoding="utf-8"
        )
        temporary.replace(queue / (identifier + ".request.json"))
        target = queue / (identifier + ".response.json")
        deadline = time.monotonic() + 3500
        while not target.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError("Native SWE generation timed out")
            time.sleep(0.1)
        response = json.loads(target.read_text(encoding="utf-8"))
        assert response["status"] == "complete", response
        assert response["variant"] == self.variant
        assert (
            response["prompt_sha256"]
            == hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        )
        content = response["text"]
        GLOBAL_MODEL_STATS.add(0.0)
        return {
            "role": "assistant",
            "content": content,
            "extra": {
                "actions": parse_regex_actions(
                    content,
                    action_regex=self.config.action_regex,
                    format_error_template=self.config.format_error_template,
                ),
                "response": response,
                "cost": 0.0,
                "timestamp": time.time(),
            },
        }

    def format_message(self, **kwargs):
        return kwargs

    def format_observation_messages(self, message, outputs, template_vars=None):
        return format_observation_messages(
            outputs,
            observation_template=self.config.observation_template,
            template_vars=template_vars,
        )

    def get_template_vars(self, **kwargs):
        return self.config.model_dump() | kwargs

    def serialize(self):
        return {
            "info": {
                "config": {
                    "model": self.config.model_dump(mode="json"),
                    "model_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                }
            }
        }
