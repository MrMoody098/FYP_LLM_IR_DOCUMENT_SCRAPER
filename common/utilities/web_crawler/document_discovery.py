"""Document discovery protocol and concrete engine implementations.

DocumentDiscovery  — shared protocol; engines and fallback wrappers both implement it
TwoPhaseDiscovery  — traverse nav_path steps, then search for the document
DirectDiscovery    — search for a document directly on the start URL, no nav step
"""

import logging
from dataclasses import dataclass, field
from typing import Protocol

from common.utilities.llm_client.llm_interface import LLMClientInterface
from common.utilities.web_crawler.document_query import DocumentQuery
from common.utilities.web_crawler.llm_navigator import DEFAULT_MAX_DEPTH, LLMNavigator, VisitedStep
from common.utilities.web_crawler.http_fetcher import HttpFetcher
from common.utilities.web_crawler.prompts import NAVIGATION_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class DiscoveryResult:
    """Outcome of a document discovery run.

    Attributes:
        query: The query that produced this result.
        document_urls: URLs of discovered documents (empty if none found).
        nav_url: The URL reached after traversing nav_path (e.g. the IR page).
            None if navigation failed before reaching any nav target.
        nav_steps: Pages visited (with LLM confidence messages) during the nav phase.
            Populated even on failure so the full attempted path is always recorded.
        doc_steps: Pages visited (with LLM confidence messages) during the document
            search phase. Empty if nav failed before the doc phase ran.
    """

    query: DocumentQuery
    document_urls: list[str]
    nav_url: str | None
    nav_steps: list[VisitedStep] = field(default_factory=list)
    doc_steps: list[VisitedStep] = field(default_factory=list)
    used_subdomain_fallback: bool = False

    @property
    def path(self) -> list[str]:
        """Backward-compatible list of doc-phase visited URLs."""
        return [s.url for s in self.doc_steps]

    @property
    def document_url(self) -> str | None:
        """Convenience accessor for the first discovered document URL."""
        return self.document_urls[0] if self.document_urls else None


class DocumentDiscovery(Protocol):
    """Shared interface for discovery engines and fallback wrappers.

    Both concrete engines (TwoPhaseDiscovery, DirectDiscovery) and fallback
    decorators (WithSubdomainFallback, WithBrowserUseFallback) implement this,
    so they are interchangeable and composable.
    """

    async def find(self, query: DocumentQuery) -> DiscoveryResult: ...


class TwoPhaseDiscovery:
    """Navigate to an intermediate page, then search for the document.

    Traverses each step in query.nav_path via LLM-guided navigation, then
    runs a document search from the final URL reached.

    Use when the document is not reachable from the start URL directly —
    you need to pass through intermediate pages first (e.g. homepage → IR
    page → annual report).
    """

    def __init__(
        self,
        fetcher: HttpFetcher,
        llm_client: LLMClientInterface,
        max_depth: int = DEFAULT_MAX_DEPTH,
    ) -> None:
        self._navigator = LLMNavigator(
            fetcher, llm_client, NAVIGATION_PROMPT, max_depth
        )

    async def find(self, query: DocumentQuery) -> DiscoveryResult:
        nav_url, nav_steps = await self._navigate_path(query)
        if nav_url is None:
            return DiscoveryResult(query, [], None, nav_steps=nav_steps)
        return await self._search_for_document(nav_url, query, nav_steps)

    async def _navigate_path(
        self, query: DocumentQuery
    ) -> tuple[str | None, list[VisitedStep]]:
        current_url = query.start_url
        all_steps: list[VisitedStep] = []
        for target in query.nav_path:
            logger.debug("Navigating to: '%s'", target.description)
            outcome = await self._navigator.navigate(
                current_url,
                label="nav",
                objective=(
                    f"Objective: Find the {target.description}.\n"
                    f"Set url to the direct link to this page when FOUND."
                ),
                hints=f"Hints:\n{target.hints}\n" if target.hints else "",
            )
            all_steps.extend(outcome.steps)
            if outcome.url is None:
                return None, all_steps
            current_url = outcome.url
        return current_url, all_steps

    async def _search_for_document(
        self, start_url: str, query: DocumentQuery, nav_steps: list[VisitedStep]
    ) -> DiscoveryResult:
        logger.debug("Searching for: '%s' [%s]", query.document_type, query.document_filter)
        outcome = await self._navigator.navigate(
            start_url,
            label="doc",
            objective=(
                f"Objective: Find the {query.document_filter} {query.document_type}.\n"
                f"Set url to a direct file download (PDF, DOCX, etc.) when FOUND, "
                f"NOT a listing or navigation page."
            ),
            hints=f"Hints:\n{query.hints}\n" if query.hints else "",
        )
        urls = [outcome.url] if outcome.url else []
        return DiscoveryResult(query, urls, start_url, nav_steps=nav_steps, doc_steps=outcome.steps)


class DirectDiscovery:
    """Search for a document directly on the start URL, no nav step.

    Use when the start URL is already the page that should contain the
    document — no intermediate navigation needed.
    """

    def __init__(
        self,
        fetcher: HttpFetcher,
        llm_client: LLMClientInterface,
        max_depth: int = DEFAULT_MAX_DEPTH,
    ) -> None:
        self._navigator = LLMNavigator(
            fetcher, llm_client, NAVIGATION_PROMPT, max_depth
        )

    async def find(self, query: DocumentQuery) -> DiscoveryResult:
        logger.debug("Searching for: '%s' [%s]", query.document_type, query.document_filter)
        outcome = await self._navigator.navigate(
            query.start_url,
            label="doc",
            objective=(
                f"Objective: Find the {query.document_filter} {query.document_type}.\n"
                f"Set url to a direct file download (PDF, DOCX, etc.) when FOUND, "
                f"NOT a listing or navigation page."
            ),
            hints=f"Hints:\n{query.hints}\n" if query.hints else "",
        )
        urls = [outcome.url] if outcome.url else []
        return DiscoveryResult(query, urls, query.start_url, doc_steps=outcome.steps)
