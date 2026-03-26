"""Shared iterative LLM-guided page navigation loop."""

import logging
from dataclasses import dataclass, field
from typing import Literal
from urllib.parse import urljoin

from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)

from common.utilities.llm_client.llm_interface import LLMClientInterface
from common.utilities.web_crawler.http_fetcher import HttpFetcher

DEFAULT_MAX_DEPTH = 8


class NavigationDecision(BaseModel):
    """Structured LLM response from a single navigation step.

    Used as the response schema for structured output — the LLM is guaranteed
    to return a valid instance of this model.
    """

    model_config = ConfigDict(extra="forbid")

    action: Literal["FOUND", "NEXT", "NONE"]
    url: str | None
    confidence_message: str

    def __str__(self) -> str:
        base = f"{self.action}: {self.url}" if self.url else self.action
        return f"{base} | {self.confidence_message}" if self.confidence_message else base


@dataclass
class VisitedStep:
    """A single page fetched and evaluated by the LLM during navigation."""

    url: str
    confidence_message: str | None = None


@dataclass
class NavigationOutcome:
    """Result of a single LLMNavigator run."""

    url: str | None
    steps: list[VisitedStep] = field(default_factory=list)

    @property
    def path(self) -> list[str]:
        """Backward-compatible list of visited URLs."""
        return [s.url for s in self.steps]


class LLMNavigator:
    """Navigates from a start URL to a target by iterative LLM-guided page traversal.

    Each step fetches a page, asks the LLM whether the target is present or
    which link to follow next, and repeats up to max_depth times.

    The prompt_template must contain {current_url}, {visited}, and {content}
    placeholders, plus any additional keys passed as kwargs to navigate().
    """

    def __init__(
        self,
        fetcher: HttpFetcher,
        llm_client: LLMClientInterface,
        prompt_template: str,
        max_depth: int = DEFAULT_MAX_DEPTH,
        label: str = "nav",
    ) -> None:
        self._fetcher = fetcher
        self._llm_client = llm_client
        self._prompt_template = prompt_template
        self._max_depth = max_depth
        self._label = label

    async def navigate(self, start_url: str, label: str | None = None, **prompt_kwargs) -> NavigationOutcome:
        """Traverse pages from start_url until the target is found or max_depth is reached.

        label overrides the instance label for this call (useful when one LLMNavigator
        instance is reused for both nav and doc phases).
        Additional keyword arguments are forwarded into the prompt template.
        Returns a NavigationOutcome with the found URL (or None) and all visited steps,
        each carrying the LLM's confidence_message for that page.
        """
        effective_label = label if label is not None else self._label
        current_url = start_url
        steps = [VisitedStep(start_url)]
        visited = {start_url}

        for depth in range(self._max_depth):
            print(f"[{effective_label}] depth {depth + 1}: {current_url}")
            content = await self._fetcher.fetch(current_url)
            if not content:
                return NavigationOutcome(None, steps)

            decision = await self._decide(current_url, steps, content, **prompt_kwargs)
            print(f"[{effective_label}] LLM: {decision}")

            steps[-1].confidence_message = decision.confidence_message

            if decision.action == "FOUND":
                if not decision.url:
                    print(f"[{effective_label}] LLM returned FOUND with no URL — ignoring")
                    return NavigationOutcome(None, steps)
                return NavigationOutcome(urljoin(current_url, decision.url), steps)

            if decision.action == "NEXT":
                if not decision.url:
                    return NavigationOutcome(None, steps)
                next_url = urljoin(current_url, decision.url)
                if next_url in visited:
                    print(f"[{effective_label}] Loop detected — already visited {next_url}")
                    return NavigationOutcome(None, steps)
                visited.add(next_url)
                steps.append(VisitedStep(next_url))
                current_url = next_url
                continue

            return NavigationOutcome(None, steps)

        print(f"[{effective_label}] reached max depth ({self._max_depth})")
        return NavigationOutcome(None, steps)

    async def _decide(
        self, current_url: str, steps: list[VisitedStep], content: str, **prompt_kwargs
    ) -> NavigationDecision:
        prompt = self._prompt_template.format(
            current_url=current_url,
            visited="\n".join(s.url for s in steps),
            content=content.replace("{", "{{").replace("}", "}}"),
            **prompt_kwargs,
        )
        return await self._llm_client.chat_structured_async("", prompt, NavigationDecision)
