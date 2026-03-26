"""Concrete OpenAI implementation of the LLM client interface using the Responses API."""

import json
import logging
import os
from typing import Any, TypeVar

from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel

from common.utilities.llm_client.llm_interface import LLMClientInterface

T = TypeVar("T", bound=BaseModel)

logger = logging.getLogger(__name__)


class OpenAIClient(LLMClientInterface):
    """Concrete implementation of an LLM client using the OpenAI SDK (Responses API)."""

    def __init__(self, model: str = "gpt-4.1", organization: str | None = None, project: str | None = None):
        api_key = os.getenv("API_KEY_OPENAI")
        if not api_key:
            raise ValueError("No OpenAI API key found. Set API_KEY_OPENAI.")
        self._model = model
        self._client = OpenAI(api_key=api_key, organization=organization, project=project)
        self._async_client = AsyncOpenAI(api_key=api_key, organization=organization, project=project)

    @property
    def model(self) -> str:
        return self._model

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        response = self._client.responses.create(
            model=self._model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.output_text or ""

    def chat_structured(self, system_prompt: str, user_prompt: str, response_type: type[T]) -> T:
        schema = response_type.model_json_schema()
        result = self.chat_json(
            system_prompt,
            user_prompt,
            response_schema={"name": response_type.__name__, "schema": schema},
        )
        return response_type.model_validate(result)

    def chat_json(self, system_prompt: str, user_prompt: str,
                  tools: list[dict[str, Any]] | None = None,
                  response_schema: dict[str, Any] | None = None) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if tools:
            kwargs["tools"] = tools
        if response_schema:
            kwargs["text"] = {"format": {
                "type": "json_schema",
                "name": response_schema.get("name", "response"),
                "schema": response_schema["schema"],
                "strict": True,
            }}
        else:
            kwargs["text"] = {"format": {"type": "json_object"}}
        response = self._client.responses.create(**kwargs)
        content = response.output_text
        if not content:
            raise ValueError("API returned empty response for JSON request")
        try:
            return json.loads(content)
        except json.JSONDecodeError as je:
            logger.error("Invalid JSON from model: %s", content)
            raise ValueError(f"Model did not return valid JSON: {je}") from je

    async def chat_async(self, system_prompt: str, user_prompt: str) -> str:
        response = await self._async_client.responses.create(
            model=self._model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.output_text or ""

    async def chat_structured_async(self, system_prompt: str, user_prompt: str, response_type: type[T]) -> T:
        schema = response_type.model_json_schema()
        result = await self.chat_json_async(
            system_prompt,
            user_prompt,
            response_schema={"name": response_type.__name__, "schema": schema},
        )
        return response_type.model_validate(result)

    async def chat_json_async(self, system_prompt: str, user_prompt: str,
                              tools: list[dict[str, Any]] | None = None,
                              response_schema: dict[str, Any] | None = None) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if tools:
            kwargs["tools"] = tools
        if response_schema:
            kwargs["text"] = {"format": {
                "type": "json_schema",
                "name": response_schema.get("name", "response"),
                "schema": response_schema["schema"],
                "strict": True,
            }}
        else:
            kwargs["text"] = {"format": {"type": "json_object"}}
        response = await self._async_client.responses.create(**kwargs)
        content = response.output_text
        if not content:
            raise ValueError("API returned empty response for JSON request")
        try:
            return json.loads(content)
        except json.JSONDecodeError as je:
            logger.error("Invalid JSON from model: %s", content)
            raise ValueError(f"Model did not return valid JSON: {je}") from je
