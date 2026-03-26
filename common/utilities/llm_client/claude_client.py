"""Concrete Anthropic implementation of the LLM client interface."""

import json
import logging
import os
from typing import Any, TypeVar

import anthropic
from pydantic import BaseModel

from common.utilities.llm_client.llm_interface import LLMClientInterface

T = TypeVar("T", bound=BaseModel)

logger = logging.getLogger(__name__)

_STRUCTURED_TOOL_NAME = "structured_response"


class ClaudeClient(LLMClientInterface):
    """Concrete implementation of an LLM client using the Anthropic SDK.

    Structured outputs are enforced via tool use: a single tool whose input
    schema matches the requested Pydantic model is defined and Claude is
    forced to call it, guaranteeing a schema-valid response.
    """

    def __init__(self, model: str = "claude-sonnet-4-6"):
        api_key = os.getenv("API_KEY_ANTHROPIC")
        if not api_key:
            raise ValueError("No Anthropic API key found. Set API_KEY_ANTHROPIC.")
        self._model = model
        self._client = anthropic.Anthropic(api_key=api_key)
        self._async_client = anthropic.AsyncAnthropic(api_key=api_key)

    @property
    def model(self) -> str:
        return self._model

    # ------------------------------------------------------------------
    # Sync methods
    # ------------------------------------------------------------------

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text if response.content else ""

    def chat_structured(self, system_prompt: str, user_prompt: str, response_type: type[T]) -> T:
        result = self._call_with_schema(
            system_prompt, user_prompt, response_type.__name__, response_type.model_json_schema(), sync=True
        )
        return response_type.model_validate(result)

    def chat_json(self, system_prompt: str, user_prompt: str,
                  tools: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        # Use a generic open-schema tool to guarantee a JSON dict response.
        generic_schema = {
            "type": "object",
            "properties": {
                "result": {
                    "type": "object",
                    "description": "The full JSON response",
                    "additionalProperties": True,
                }
            },
            "required": ["result"],
        }
        augmented_prompt = user_prompt
        if tools:
            augmented_prompt += f"\n\nAvailable tools context:\n{json.dumps(tools)}"
        raw = self._call_with_schema(system_prompt, augmented_prompt, "json_response", generic_schema, sync=True)
        # If the model wrapped the response in {"result": {...}} unwrap it
        if set(raw.keys()) == {"result"} and isinstance(raw["result"], dict):
            return raw["result"]
        return raw

    # ------------------------------------------------------------------
    # Async methods
    # ------------------------------------------------------------------

    async def chat_async(self, system_prompt: str, user_prompt: str) -> str:
        response = await self._async_client.messages.create(
            model=self._model,
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text if response.content else ""

    async def chat_structured_async(self, system_prompt: str, user_prompt: str, response_type: type[T]) -> T:
        result = await self._call_with_schema_async(
            system_prompt, user_prompt, response_type.__name__, response_type.model_json_schema()
        )
        return response_type.model_validate(result)

    async def chat_json_async(self, system_prompt: str, user_prompt: str,
                              tools: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        generic_schema = {
            "type": "object",
            "properties": {
                "result": {
                    "type": "object",
                    "description": "The full JSON response",
                    "additionalProperties": True,
                }
            },
            "required": ["result"],
        }
        augmented_prompt = user_prompt
        if tools:
            augmented_prompt += f"\n\nAvailable tools context:\n{json.dumps(tools)}"
        raw = await self._call_with_schema_async(system_prompt, augmented_prompt, "json_response", generic_schema)
        if set(raw.keys()) == {"result"} and isinstance(raw["result"], dict):
            return raw["result"]
        return raw

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_tool(self, name: str, schema: dict[str, Any]) -> dict[str, Any]:
        return {
            "name": name,
            "description": "Return a structured response that exactly matches the provided schema.",
            "input_schema": schema,
        }

    def _extract_tool_result(self, response: Any, tool_name: str) -> dict[str, Any]:
        for block in response.content:
            if block.type == "tool_use" and block.name == tool_name:
                return block.input
        raise ValueError(f"Model did not call the expected tool '{tool_name}'. Stop reason: {response.stop_reason}")

    def _call_with_schema(
        self,
        system_prompt: str,
        user_prompt: str,
        tool_name: str,
        schema: dict[str, Any],
        sync: bool = True,
    ) -> dict[str, Any]:
        tool = self._build_tool(tool_name, schema)
        response = self._client.messages.create(
            model=self._model,
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            tools=[tool],
            tool_choice={"type": "tool", "name": tool_name},
        )
        return self._extract_tool_result(response, tool_name)

    async def _call_with_schema_async(
        self,
        system_prompt: str,
        user_prompt: str,
        tool_name: str,
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        tool = self._build_tool(tool_name, schema)
        response = await self._async_client.messages.create(
            model=self._model,
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            tools=[tool],
            tool_choice={"type": "tool", "name": tool_name},
        )
        return self._extract_tool_result(response, tool_name)