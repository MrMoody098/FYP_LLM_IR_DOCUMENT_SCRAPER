"""Abstract interface defining the contract for LLM provider implementations."""

from abc import ABC, abstractmethod
from typing import Any, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

class LLMClientInterface(ABC):
    """Abstract Interface for LLM Interactions.    
    This defines the contract that any LLM provider must fulfill.
    """
    @property
    @abstractmethod
    def model(self) -> str:
        """Return the name of the model being used."""
        pass

    @abstractmethod
    def chat(self, system_prompt: str, user_prompt: str) -> str:
        """Send a chat completion request and return the response as a string.

        Args:
            system_prompt: Instructions for the model's behavior.
            user_prompt: The user's message/query.

        Returns:
            The model's response content as a string.
        """
        pass

    @abstractmethod
    def chat_structured(self, system_prompt: str, user_prompt: str, response_type: type[T]) -> T:
        """Send a chat completion request and return the response as a validated Pydantic model.

        Args:
            system_prompt: Instructions for the model's behavior.
            user_prompt: The user's message/query.
            response_type: A Pydantic BaseModel subclass defining the expected response schema.

        Returns:
            A validated instance of response_type.
        """
        pass

    @abstractmethod
    def chat_json(self, system_prompt: str, user_prompt: str,
                  tools: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        """Send a chat completion request and return the response as a parsed JSON dict.

        Args:
            system_prompt: Instructions for the model's behavior.
            user_prompt: The user's message/query.
            tools: Optional list of tool definitions (e.g. web search) to provide to the model.

        Returns:
            The model's response parsed as a dictionary.
        """
        pass

    @abstractmethod
    async def chat_async(self, system_prompt: str, user_prompt: str) -> str:
        """Send a chat completion request asynchronously and return the response as a string.

        Args:
            system_prompt: Instructions for the model's behavior.
            user_prompt: The user's message/query.

        Returns:
            The model's response content as a string.
        """
        pass

    @abstractmethod
    async def chat_structured_async(self, system_prompt: str, user_prompt: str, response_type: type[T]) -> T:
        """Send a chat completion request asynchronously and return the response as a validated Pydantic model.

        Args:
            system_prompt: Instructions for the model's behavior.
            user_prompt: The user's message/query.
            response_type: A Pydantic BaseModel subclass defining the expected response schema.

        Returns:
            A validated instance of response_type.
        """
        pass

    @abstractmethod
    async def chat_json_async(self, system_prompt: str, user_prompt: str,
                              tools: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        """Send a chat completion request asynchronously and return the response as a parsed JSON dict.

        Args:
            system_prompt: Instructions for the model's behavior.
            user_prompt: The user's message/query.
            tools: Optional list of tool definitions (e.g. web search) to provide to the model.

        Returns:
            The model's response parsed as a dictionary.
        """
        pass
