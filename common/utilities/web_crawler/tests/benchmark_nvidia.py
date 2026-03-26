"""Benchmark: curl_cffi vs Playwright fetcher speeds for finding Nvidia's annual report PDF.

Shows per-step timing for each fetch and LLM call in both pipelines so you can
see where time is spent and how the two approaches compare.

Run:
    pytest common/utilities/web_crawler/tests/benchmark_nvidia.py -v -s
"""

import time
from dataclasses import dataclass, field
from typing import Any, TypeVar

import pytest
from pydantic import BaseModel

from common.utilities.llm_client.llm_interface import LLMClientInterface
from common.utilities.llm_client.open_ai_client import OpenAIClient
from common.utilities.web_crawler.company_crawler import FORM_10K
from common.utilities.web_crawler.document_discovery import TwoPhaseDiscovery
from common.utilities.web_crawler.document_query import DocumentQuery
from common.utilities.web_crawler.http_fetcher import HttpFetcher, PlaywrightFetcher

T = TypeVar("T", bound=BaseModel)

NVIDIA_URL = "https://www.apple.com"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class StepTiming:
    step: int
    url: str
    operation: str  # "fetch" or "llm"
    elapsed_s: float


@dataclass
class PipelineResult:
    name: str
    document_url: str | None
    steps: list[StepTiming] = field(default_factory=list)
    total_elapsed_s: float = 0.0

    @property
    def fetch_steps(self) -> list[StepTiming]:
        return [s for s in self.steps if s.operation == "fetch"]

    @property
    def llm_steps(self) -> list[StepTiming]:
        return [s for s in self.steps if s.operation == "llm"]


# ---------------------------------------------------------------------------
# Timing wrappers
# ---------------------------------------------------------------------------


class TimedFetcher:
    """Wraps HttpFetcher or PlaywrightFetcher to record per-fetch timing."""

    def __init__(self, inner, timings: list[StepTiming]) -> None:
        self._inner = inner
        self._timings = timings

    async def fetch(self, url: str) -> str | None:
        t0 = time.perf_counter()
        result = await self._inner.fetch(url)
        elapsed = time.perf_counter() - t0
        self._timings.append(
            StepTiming(step=len(self._timings) + 1, url=url, operation="fetch", elapsed_s=elapsed)
        )
        return result

    async def probe(self, url: str) -> bool:
        return await self._inner.probe(url)


def _extract_current_url(prompt: str) -> str:
    """Pull the 'Current page: <url>' line from the navigator prompt."""
    for line in prompt.splitlines():
        if line.startswith("Current page:"):
            return line.removeprefix("Current page:").strip()
    return "llm_call"


class TimedLLMClient(LLMClientInterface):
    """Wraps LLMClientInterface to record per-call timing for async structured calls.

    Only chat_structured_async is instrumented — that is the only method the
    navigator uses.  All other methods are pass-throughs.
    """

    def __init__(self, inner: LLMClientInterface, timings: list[StepTiming]) -> None:
        self._inner = inner
        self._timings = timings

    @property
    def model(self) -> str:
        return self._inner.model

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        return self._inner.chat(system_prompt, user_prompt)

    def chat_structured(self, system_prompt: str, user_prompt: str, response_type: type[T]) -> T:
        return self._inner.chat_structured(system_prompt, user_prompt, response_type)

    def chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        return self._inner.chat_json(system_prompt, user_prompt, tools)

    async def chat_async(self, system_prompt: str, user_prompt: str) -> str:
        return await self._inner.chat_async(system_prompt, user_prompt)

    async def chat_structured_async(
        self, system_prompt: str, user_prompt: str, response_type: type[T]
    ) -> T:
        url = _extract_current_url(user_prompt)
        t0 = time.perf_counter()
        result = await self._inner.chat_structured_async(system_prompt, user_prompt, response_type)
        elapsed = time.perf_counter() - t0
        self._timings.append(
            StepTiming(step=len(self._timings) + 1, url=url, operation="llm", elapsed_s=elapsed)
        )
        return result

    async def chat_json_async(
        self,
        system_prompt: str,
        user_prompt: str,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        return await self._inner.chat_json_async(system_prompt, user_prompt, tools)


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------


async def _run_pipeline(
    name: str,
    fetcher,  # HttpFetcher() or PlaywrightFetcher() — used as async context manager
    llm_client: LLMClientInterface,
    query: DocumentQuery,
) -> PipelineResult:
    timings: list[StepTiming] = []
    timed_llm = TimedLLMClient(llm_client, timings)

    async with fetcher as inner:
        timed_fetcher = TimedFetcher(inner, timings)
        discovery = TwoPhaseDiscovery(timed_fetcher, timed_llm)
        t0 = time.perf_counter()
        result = await discovery.find(query)
        total = time.perf_counter() - t0

    return PipelineResult(
        name=name,
        document_url=result.document_url,
        steps=timings,
        total_elapsed_s=total,
    )


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _print_result(result: PipelineResult) -> None:
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  Pipeline : {result.name}")
    print(f"  Document : {result.document_url or 'NOT FOUND'}")
    print(f"  Total    : {result.total_elapsed_s:.2f}s")
    print(sep)
    for step in result.steps:
        tag = "FETCH" if step.operation == "fetch" else "LLM  "
        url = step.url if len(step.url) <= 64 else step.url[:61] + "..."
        print(f"  {step.step:2d}. [{tag}] {step.elapsed_s:5.2f}s  {url}")
    fetch_total = sum(s.elapsed_s for s in result.fetch_steps)
    llm_total = sum(s.elapsed_s for s in result.llm_steps)
    print(f"\n  Fetch : {fetch_total:.2f}s total  ({len(result.fetch_steps)} calls)")
    print(f"  LLM   : {llm_total:.2f}s total  ({len(result.llm_steps)} calls)")


def _print_comparison(r1: PipelineResult, r2: PipelineResult) -> None:
    sep = "=" * 72
    print(f"\n{sep}")
    print("  COMPARISON SUMMARY")
    print(f"  {'Metric':<22} {r1.name:>18} {r2.name:>18}  faster")
    print(f"  {'-' * 68}")

    def row(label: str, v1: float, v2: float) -> None:
        faster = r1.name if v1 < v2 else (r2.name if v2 < v1 else "tie")
        print(f"  {label:<22} {v1:>16.2f}s {v2:>16.2f}s  {faster}")

    def count_row(label: str, v1: int, v2: int) -> None:
        print(f"  {label:<22} {v1:>17d}  {v2:>17d}")

    row("Total time", r1.total_elapsed_s, r2.total_elapsed_s)
    row(
        "Fetch total",
        sum(s.elapsed_s for s in r1.fetch_steps),
        sum(s.elapsed_s for s in r2.fetch_steps),
    )
    row(
        "LLM total",
        sum(s.elapsed_s for s in r1.llm_steps),
        sum(s.elapsed_s for s in r2.llm_steps),
    )
    count_row("Fetch calls", len(r1.fetch_steps), len(r2.fetch_steps))
    count_row("LLM calls", len(r1.llm_steps), len(r2.llm_steps))
    print(sep)


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def llm_client() -> LLMClientInterface:
    return OpenAIClient()


@pytest.mark.asyncio
async def test_benchmark_nvidia_fetchers(llm_client: LLMClientInterface) -> None:
    """Compare curl_cffi vs Playwright for finding Nvidia's annual report PDF.

    Prints a per-step breakdown of fetch and LLM call timing for both
    pipelines, followed by a side-by-side comparison.

    Run with:
        pytest common/utilities/web_crawler/tests/benchmark_nvidia.py -v -s
    """
    query = DocumentQuery(
        start_url=NVIDIA_URL,
        nav_path=FORM_10K.nav_path,
        document_type=FORM_10K.document_type,
        document_filter=FORM_10K.document_filter,
    )

    print("\n\n--- Running curl_cffi + OpenAI pipeline ---")
    curl_result = await _run_pipeline("curl_cffi + OpenAI", HttpFetcher(), llm_client, query)

    print("\n--- Running Playwright + OpenAI pipeline ---")
    playwright_result = await _run_pipeline(
        "Playwright + OpenAI", PlaywrightFetcher(), llm_client, query
    )

    _print_result(curl_result)
    _print_result(playwright_result)
    _print_comparison(curl_result, playwright_result)