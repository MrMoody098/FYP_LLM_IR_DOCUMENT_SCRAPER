"""Declarative specification for document discovery queries."""

from dataclasses import dataclass


@dataclass(frozen=True)
class NavigationTarget:
    """An AI-understandable description of a page to navigate to.

    Examples: "investor relations page", "SEC filings section",
    "annual reports download page".

    Attributes:
        description: What page to find.
        hints: Optional site-specific navigation guidance injected into the
            prompt. E.g. "Look for a link labeled 'Investors' in the top nav."
    """

    description: str
    hints: str = ""


@dataclass(frozen=True)
class DocumentQuery:
    """Declarative specification for what document to find and where to look.

    Attributes:
        start_url: Entry point URL for the crawl.
        nav_path: Ordered sequence of page descriptions to traverse before
            searching for the document. Each step is an AI-understandable
            description of the target page.
        document_type: AI-understandable description of the document type.
            Examples: "annual report (10-K)", "quarterly report (10-Q)",
            "press release".
        document_filter: AI-understandable criteria describing which specific
            documents to select. Examples: "most recent", "all historical",
            "fiscal year 2025", "Q1 2025", "calendar 2020 to 2026".
        hints: Optional site-specific document-search guidance injected into
            the prompt. E.g. "The 10-K link is on the investor home page,
            not inside SEC Filings."
    """

    start_url: str
    nav_path: list[NavigationTarget]
    document_type: str
    document_filter: str
    hints: str = ""
