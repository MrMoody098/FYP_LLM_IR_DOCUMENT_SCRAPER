"""Fallback decorators for document discovery.

Each wraps any DocumentDiscovery and activates based on an injectable predicate.
If the inner engine's result satisfies activate_when, the fallback strategy runs.

WithSubdomainFallback   — probes common IR subdomains, retries inner on each
WithBrowserUseFallback  — placeholder for a future browser-based fallback
"""

import asyncio
import logging
from dataclasses import replace as dc_replace
from typing import Callable
from urllib.parse import urlparse

from common.utilities.web_crawler.document_discovery import DiscoveryResult, DocumentDiscovery
from common.utilities.web_crawler.document_query import DocumentQuery
from common.utilities.web_crawler.http_fetcher import HttpFetcher

logger = logging.getLogger(__name__)

IR_SUBDOMAIN_PATTERNS = [
    "investor.{domain}",
    "investors.{domain}",
    "ir.{domain}",
    "corporate.{domain}",
    "stock.{domain}",
]

_no_documents: Callable[[DiscoveryResult], bool] = lambda r: not r.document_urls


class WithSubdomainFallback:
    """Wraps any DocumentDiscovery — if activate_when is true, probes subdomain candidates.

    Concurrently probes each pattern in subdomain_patterns. For each that resolves,
    retries the inner engine starting from that subdomain (with nav_path cleared,
    since subdomain URLs are already the target intermediate page).
    Returns the first successful result, or the original result unchanged.

    subdomain_patterns: list of '{domain}' format strings to probe.
    activate_when: predicate over DiscoveryResult that decides whether to fire.
        Defaults to: no documents found.
    """

    def __init__(
        self,
        inner: DocumentDiscovery,
        fetcher: HttpFetcher,
        subdomain_patterns: list[str],
        activate_when: Callable[[DiscoveryResult], bool] = _no_documents,
    ) -> None:
        self._inner = inner
        self._fetcher = fetcher
        self._subdomain_patterns = subdomain_patterns
        self._activate_when = activate_when

    async def find(self, query: DocumentQuery) -> DiscoveryResult:
        result = await self._inner.find(query)
        if not self._activate_when(result):
            return result

        candidates = await self._probe_subdomains(query.start_url)
        for candidate in candidates:
            logger.debug("Retrying from subdomain candidate: %s", candidate)
            retry_query = dc_replace(query, start_url=candidate, nav_path=[])
            fallback_result = await self._inner.find(retry_query)
            if fallback_result.document_urls:
                return dc_replace(fallback_result, used_subdomain_fallback=True)

        return result

    async def _probe_subdomains(self, start_url: str) -> list[str]:
        domain = urlparse(start_url).netloc.removeprefix("www.")
        logger.debug("Probing IR subdomains for %s", domain)

        async def probe(pattern: str) -> str | None:
            candidate = f"https://{pattern.format(domain=domain)}"
            if await self._fetcher.probe(candidate):
                logger.debug("Resolved: %s", candidate)
                return candidate
            return None

        results = await asyncio.gather(*[probe(p) for p in self._subdomain_patterns])
        return [r for r in results if r is not None]


class WithBrowserUseFallback:
    """Wraps any DocumentDiscovery — if activate_when is true, delegates to a browser engine.

    Placeholder for future integration with a browser-based fetcher that can
    render JavaScript and see dynamically loaded content.

    activate_when: predicate over DiscoveryResult that decides whether to fire.
        Defaults to: no documents found.
    """

    def __init__(
        self,
        inner: DocumentDiscovery,
        activate_when: Callable[[DiscoveryResult], bool] = _no_documents,
    ) -> None:
        self._inner = inner
        self._activate_when = activate_when

    async def find(self, query: DocumentQuery) -> DiscoveryResult:
        result = await self._inner.find(query)
        if not self._activate_when(result):
            return result
        return result
