"""Page content fetching via Playwright and html2text."""

import logging
from typing import Any

import html2text
from playwright.async_api import Browser, BrowserContext, async_playwright

logger = logging.getLogger(__name__)

_WAIT_UNTIL = "networkidle"
_TIMEOUT_MS = 20_000  # 20 s per page


class HttpFetcher:
    """Fetches page content using a headless Chromium browser via Playwright.

    Fully renders JavaScript before extracting content, making it suitable
    for corporate IR pages that rely on client-side rendering.

    HTML responses are converted to markdown to minimise token usage when
    passed to an LLM.

    Must be used as an async context manager — the browser is launched on
    entry and closed on exit:

        async with HttpFetcher() as fetcher:
            content = await fetcher.fetch("https://example.com")
            ok = await fetcher.probe("https://example.com/ir")

    Provides two strategies:
    - fetch(): navigate and return markdown content, or None on failure
    - probe(): navigate and return True if the page loads successfully
    """

    def __init__(self, timeout: int = 20):
        self._timeout_ms = timeout * 1000
        self._playwright = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._converter = html2text.HTML2Text()
        self._converter.ignore_images = True
        self._converter.body_width = 0  # don't wrap lines — preserves full URLs

    async def __aenter__(self) -> "HttpFetcher":
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(headless=True)
        self._context = await self._browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/131.0.0.0 Safari/537.36"
            ),
            java_script_enabled=True,
        )
        return self

    async def __aexit__(self, *_: Any) -> None:
        if self._context:
            await self._context.close()
            self._context = None
        if self._browser:
            await self._browser.close()
            self._browser = None
        if self._playwright:
            await self._playwright.stop()
            self._playwright = None

    async def fetch(self, url: str) -> str | None:
        """Return the markdown content of a fully-rendered page, or None on failure."""
        page = await self._context.new_page()
        try:
            await page.goto(url, wait_until=_WAIT_UNTIL, timeout=self._timeout_ms)
            html = await page.content()
            markdown = self._converter.handle(html)
            logger.debug("Fetched %s: %d chars", url, len(markdown))
            return markdown
        except Exception as exc:
            logger.warning("Failed to fetch %s: %s", url, exc)
            return None
        finally:
            await page.close()

    async def probe(self, url: str) -> bool:
        """Return True if the URL loads successfully."""
        page = await self._context.new_page()
        try:
            response = await page.goto(url, wait_until="load", timeout=self._timeout_ms)
            return response is not None and response.ok
        except Exception:
            return False
        finally:
            await page.close()


_IMPERSONATE = "chrome131"


class CffiHttpFetcher:
    """Fetches page content using curl_cffi with Chrome TLS impersonation.

    Sends requests indistinguishable from real Chrome at the TLS level,
    bypassing fingerprint-based bot detection without launching a browser.
    HTML responses are converted to markdown to minimise token usage when
    passed to an LLM.

    Must be used as an async context manager — the session is created on
    entry and closed on exit, allowing all calls within the block to share
    a single connection pool:

        async with CffiHttpFetcher() as fetcher:
            content = await fetcher.fetch("https://example.com")
            ok = await fetcher.probe("https://example.com/ir")

    Provides two strategies:
    - fetch(): GET request, returns markdown content or None on failure
    - probe(): GET request, returns True if the URL resolves
    """

    def __init__(self, timeout: int = 15):
        self._timeout = timeout
        self._session: Any = None
        self._converter = html2text.HTML2Text()
        self._converter.ignore_images = True
        self._converter.body_width = 0  # don't wrap lines — preserves full URLs

    async def __aenter__(self) -> "CffiHttpFetcher":
        from curl_cffi.requests import AsyncSession
        self._session = AsyncSession(impersonate=_IMPERSONATE)
        return self

    async def __aexit__(self, *_: Any) -> None:
        if self._session:
            await self._session.close()
            self._session = None

    @property
    def _s(self):
        if self._session is None:
            raise RuntimeError("CffiHttpFetcher must be used as an async context manager")
        return self._session

    async def fetch(self, url: str) -> str | None:
        """Return the markdown content of a page, or None on failure."""
        try:
            response = await self._s.get(url, timeout=self._timeout)
            response.raise_for_status()
            markdown = self._converter.handle(response.text)
            logger.debug("Fetched %s: %d chars", url, len(markdown))
            return markdown
        except Exception as exc:
            logger.warning("Failed to fetch %s: %s", url, exc)
            return None

    async def probe(self, url: str) -> bool:
        """Return True if the URL resolves successfully.

        Uses GET rather than HEAD because many corporate IR sites block HEAD
        requests or return misleading status codes for them.
        """
        try:
            response = await self._s.get(url, timeout=self._timeout, allow_redirects=True)
            return response.ok
        except Exception:
            return False
