"""Orchestrator for company financial document discovery.

Owns the business logic of what documents to find for a company.
Composes discovery engines and fallbacks, then maps DiscoveryResult
to the FinancialDocument / LinkTrace data model.
"""

from dataclasses import replace
from datetime import datetime, timezone

from common.utilities.llm_client.llm_interface import LLMClientInterface
from common.utilities.web_crawler.crawler_data import (
    Company,
    FinancialDocument,
    LinkTrace,
    NavigationStep,
)
from common.utilities.web_crawler.document_discovery import DiscoveryResult, TwoPhaseDiscovery
from common.utilities.web_crawler.document_query import DocumentQuery, NavigationTarget
from common.utilities.web_crawler.fallbacks import (
    IR_SUBDOMAIN_PATTERNS,
    WithBrowserUseFallback,
    WithSubdomainFallback,
)
from common.utilities.web_crawler.http_fetcher import HttpFetcher

FORM_10K = DocumentQuery(
    start_url="",
    nav_path=[NavigationTarget("investor relations page")],
    document_type="Form 10-K",
    document_filter="most recent",
    hints=(
        "The target is the SEC Form 10-K (the regulatory annual filing), "
        "NOT the glossy shareholder Annual Report PDF. "
        "Look for links labeled '10-K', 'Form 10-K', 'Annual Report on Form 10-K', or inside an 'SEC Filings' section. "
        "Do NOT select a link labeled only 'Annual Report' unless it explicitly says 'Form 10-K'. usually the annual report is on a main page or within a SEC FILINGS section"
    ),
)

FORM_10Q_RECENT = DocumentQuery(
    start_url="",
    nav_path=[NavigationTarget("investor relations page")],
    document_type="Form 10-Q",
    document_filter="most recent",
    hints=(
        "The target is the SEC Form 10-Q (the regulatory quarterly filing). "
        "Look for links labeled '10-Q', 'Form 10-Q', usually these can be found by navigating to a 'SEC Filings' section or a reports/corporate filings section. "
        "Do NOT select earnings press releases, investor presentations, or earnings call transcripts. "
        "Do NOT select a Form 10-K — that is the annual filing, not a quarterly 10-Q."
    ),
)

FORM_10Q_Q1_2025 = DocumentQuery(
    start_url="",
    nav_path=[NavigationTarget("investor relations page")],
    document_type="Form 10-Q",
    document_filter="Q1 2025 (quarter ending around March 2025)",
    hints=(
        "The target is the SEC Form 10-Q (the regulatory quarterly filing). "
        "Look for links labeled '10-Q', 'Form 10-Q', usually these can be found by navigating to a'SEC Filings' section or a reports/corporate filings section. "
        "Do NOT select earnings press releases, investor presentations, or earnings call transcripts."
    ),
)


class CompanyDocumentCrawler:
    """Finds financial documents for a company starting from its root URL.

    Navigates from the company root to the target document and maps the result
    to a FinancialDocument with a full LinkTrace. Metadata fields (period_end,
    fiscal_year, etc.) are left None — populated by a downstream extraction step.

    Must be used as an async context manager so the HTTP session is properly
    opened and closed:

        async with CompanyDocumentCrawler(llm_client) as crawler:
            result = await crawler.find(company, ANNUAL_REPORT)
    """

    def __init__(self, llm_client: LLMClientInterface, fetcher=None) -> None:
        self._fetcher = fetcher or HttpFetcher()
        self._discovery = WithBrowserUseFallback(
            WithSubdomainFallback(
                TwoPhaseDiscovery(self._fetcher, llm_client),
                self._fetcher,
                subdomain_patterns=IR_SUBDOMAIN_PATTERNS,
                activate_when=lambda r: not r.document_urls,
            ),
            activate_when=lambda r: not r.document_urls,
        )

    async def __aenter__(self) -> "CompanyDocumentCrawler":
        await self._fetcher.__aenter__()
        return self

    async def __aexit__(self, *args) -> None:
        await self._fetcher.__aexit__(*args)

    async def find(self, company: Company, query_template: DocumentQuery) -> FinancialDocument:
        """Find documents described by query_template for the given company."""
        query = replace(query_template, start_url=company.root_url)
        result = await self._discovery.find(query)
        return self._to_financial_document(result, company.company_identifier)

    def _to_financial_document(
        self, result: DiscoveryResult, company_identifier: str
    ) -> FinancialDocument:
        return FinancialDocument(
            company_identifier=company_identifier,
            document_type=result.query.document_type,
            document_url=result.document_url or "",
            used_subdomain_fallback=result.used_subdomain_fallback,
            link_trace=self._build_link_trace(result),
            link_retrieval_datetime_utc=datetime.now(timezone.utc),
        )

    def _build_link_trace(self, result: DiscoveryResult) -> LinkTrace:
        """Build a LinkTrace from all pages visited during discovery.

        Includes every page the crawler fetched and the LLM's confidence
        message for each, even when navigation or document search fails.

        Chain: nav_steps (homepage → ... → IR page) → doc_steps (IR page → ... → document)
        """
        steps: list[NavigationStep] = []
        query = result.query
        nav_target_desc = query.nav_path[-1].description if query.nav_path else None

        # Pages visited during the navigation phase (includes company root as step 0).
        # Populated even on failure so we can see where the crawler got stuck.
        for i, step in enumerate(result.nav_steps):
            steps.append(NavigationStep(
                url=step.url,
                page_description="company root" if i == 0 else "navigation",
                confidence_message=step.confidence_message,
            ))

        # If nav_path was empty (e.g. subdomain fallback) there are no nav_steps,
        # so emit the start URL as the company root.
        if not result.nav_steps:
            steps.append(NavigationStep(
                url=query.start_url,
                page_description="company root",
            ))

        # Pages visited during the document search phase.
        # doc_steps[0].url == nav_url (the IR page), subsequent steps are deeper pages.
        for i, step in enumerate(result.doc_steps):
            page_description = nav_target_desc if (i == 0 and nav_target_desc) else "document search"
            steps.append(NavigationStep(
                url=step.url,
                page_description=page_description,
                confidence_message=step.confidence_message,
            ))

        if result.document_url:
            steps.append(NavigationStep(
                url=result.document_url,
                page_description=query.document_type,
            ))

        return LinkTrace(navigation_steps=steps)
