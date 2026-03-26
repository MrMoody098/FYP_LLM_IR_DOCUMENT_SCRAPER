from dataclasses import asdict, dataclass, field
from typing import Any, List, Optional
from datetime import datetime, timezone


@dataclass
class NavigationStep:
    """A single step in the navigation path taken to reach a document.

    Represents one page visited during the crawl, forming a sequential
    breadcrumb trail from the company homepage to the target document.
    """
    url: str
    page_description: str
    link_text: Optional[str] = None
    confidence_message: Optional[str] = None


@dataclass
class LinkTrace:
    """Sequential record of navigation steps taken to reach a document.

    Captures the full path from the company homepage through intermediate
    pages (e.g. investor relations, financial reports) to the final document.
    """
    navigation_steps: List[NavigationStep] = field(default_factory=list)


@dataclass
class FinancialDocument:
    """A financial document discovered during a website crawl.

    Attributes:
        company_identifier: Unique identifier for the company that published
            this document.
        document_type: Type of document (e.g. 'annual_report', 'quarterly_report',
            'half_yearly_report', 'investor_presentation').
        document_url: URL where the document can be downloaded.
        period_end: End date of the reporting period covered by the document.
        fiscal_period: Period number within the reporting cycle.
            Quarterly: {1, 2, 3, 4}, Half-yearly: {1, 2}, Annual: {1}.
        reporting_periodicity: Reporting frequency - 'Q' (quarterly),
            'H' (half-yearly), or 'A' (annual).
        fiscal_year: Fiscal year the document belongs to.
        calendar_year: Calendar year the document belongs to.
        date_reported: Date the document was originally published or filed.
        label_from_website: Display text used for the document link on the
            source website.
        link_trace: Navigation path taken from the company homepage to reach
            this document.
        link_retrieval_datetime_utc: UTC timestamp when the document link was
            discovered during the crawl.
        document_download_datetime_utc: UTC timestamp when the document was
            downloaded.
        document_hash: Hash of the document contents for deduplication
            and change detection.
    """
    company_identifier: str
    document_type: str
    document_url: str

    period_end: Optional[datetime] = None
    fiscal_period: Optional[int] = None
    reporting_periodicity: Optional[str] = None

    fiscal_year: Optional[int] = None
    calendar_year: Optional[int] = None

    date_reported: Optional[datetime] = None
    label_from_website: Optional[str] = None

    used_subdomain_fallback: bool = False
    link_trace: Optional[LinkTrace] = None

    link_retrieval_datetime_utc: Optional[datetime] = None
    document_download_datetime_utc: Optional[datetime] = None
    document_hash: Optional[str] = None


@dataclass
class Company:
    """A company whose financial documents are being crawled.

    Attributes:
        company_identifier: Unique identifier for the company.
        company_identifier_type: Source of the company identifier.
        root_url: The company's main website URL used as the crawl entry point.
        fiscal_year_end_month: Month (1-12) when the company's fiscal year ends.
        fiscal_year_end_day: Day of month when the company's fiscal year ends.
    """
    company_identifier: str
    company_identifier_type: str
    company_name: str
    root_url: str
    country: Optional[str] = None
    fiscal_year_end_month: Optional[int] = None
    fiscal_year_end_day: Optional[int] = None


@dataclass
class CrawlContext:
    """Metadata about a crawl operation.

    Attributes:
        crawl_timestamp: When the crawl was executed.
        model_used: The LLM model used for intelligent document discovery.
        invocation_id: Unique identifier for the Azure Function App execution
            that produced this crawl. Useful for searching logs on Azure.
    """
    crawl_timestamp: datetime
    invocation_id: str
    model_used: Optional[str] = None


SCHEMA_VERSION = "1.0"


@dataclass
class CompanyCapture:
    """Root container for a single company's crawl results.

    Aggregates the company metadata, crawl context, all discovered
    financial documents, and any errors encountered during the crawl.

    Attributes:
        errors: Issues reported by the LLM during crawling.
    """
    company: Company
    crawl_context: CrawlContext
    schema_version: str = SCHEMA_VERSION
    documents: List[FinancialDocument] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to a JSON-serializable dict, formatting datetimes as ISO 8601 strings."""
        raw = asdict(self)
        return _make_json_serializable(raw)


def _make_json_serializable(value: Any) -> Any:
    """Recursively convert non-serializable types to JSON-safe format.

    Datetimes are normalized to UTC before formatting as ISO 8601.
    Naive datetimes (no tzinfo) are assumed to be UTC.
    """
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        else:
            value = value.astimezone(timezone.utc)
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _make_json_serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_make_json_serializable(item) for item in value]
    return value
