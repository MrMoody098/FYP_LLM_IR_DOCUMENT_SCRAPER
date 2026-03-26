"""
Unit Tests for Crawler Data Models
"""

import json
from datetime import datetime, timezone
import pytest
from common.utilities.web_crawler.crawler_data import (
    Company,
    CrawlContext,
    CompanyCapture,
    FinancialDocument,
    LinkTrace,
    NavigationStep
)


@pytest.fixture
def company():
    return Company(
        company_identifier="TEST-001",
        company_identifier_type="Worldscope",
        company_name="Test Company",
        root_url="https://example.com",
        country="US",
        fiscal_year_end_month=12,
        fiscal_year_end_day=31
    )


@pytest.fixture
def crawl_context():
    return CrawlContext(
        crawl_timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        invocation_id="test-invocation-001",
        model_used="test-model"
    )


@pytest.fixture
def link_trace():
    return LinkTrace(
        navigation_steps=[
            NavigationStep(
                url="https://example.com",
                page_description="Company homepage",
                link_text="Home"
            ),
            NavigationStep(
                url="https://example.com/ir",
                page_description="Investor relations page",
                link_text="Investor Relations"
            ),
            NavigationStep(
                url="https://example.com/financials",
                page_description="Financial reports listing",
                link_text="Financial Reports"
            ),
            NavigationStep(
                url="https://example.com/documents/report.pdf",
                page_description="Annual report PDF",
                link_text="2024 Annual Report"
            )
        ]
    )


@pytest.fixture
def financial_document(link_trace):
    return FinancialDocument(
        company_identifier="TEST-001",
        document_type="annual_report",
        document_url="https://example.com/documents/report.pdf",
        period_end=datetime(2024, 12, 31, tzinfo=timezone.utc),
        fiscal_period=1,
        reporting_periodicity="A",
        fiscal_year=2024,
        calendar_year=2024,
        date_reported=datetime(2024, 1, 15, tzinfo=timezone.utc),
        label_from_website="Test Annual Report",
        link_trace=link_trace
    )


def test_create_valid_company_capture(company, crawl_context, financial_document):
    """Test creation of a complete CompanyCapture with all nested objects."""
    # ACT
    sut = CompanyCapture(
        company=company,
        crawl_context=crawl_context,
        documents=[financial_document],
        errors=[]
    )

    # ASSERT
    assert sut is not None
    assert sut.company.company_identifier == "TEST-001"
    assert len(sut.documents) == 1
    assert sut.documents[0].document_type == "annual_report"
    assert len(sut.errors) == 0

    # Assert LinkTrace navigation steps
    link_trace = sut.documents[0].link_trace
    assert len(link_trace.navigation_steps) == 4
    assert link_trace.navigation_steps[0].url == "https://example.com"
    assert link_trace.navigation_steps[0].page_description == "Company homepage"
    assert link_trace.navigation_steps[1].url == "https://example.com/ir"
    assert link_trace.navigation_steps[1].link_text == "Investor Relations"
    assert link_trace.navigation_steps[2].url == "https://example.com/financials"
    assert link_trace.navigation_steps[3].url == "https://example.com/documents/report.pdf"
    assert link_trace.navigation_steps[3].page_description == "Annual report PDF"


def test_to_dict_produces_json_serializable_output(company, crawl_context, financial_document):
    """Test JSON serialization round-trip with datetime conversion."""
    # ARRANGE
    sut = CompanyCapture(
        company=company,
        crawl_context=crawl_context,
        documents=[financial_document],
        errors=[]
    )

    # ACT
    result = sut.to_dict()
    json_str = json.dumps(result)  # must not raise TypeError

    # ASSERT
    parsed = json.loads(json_str)
    assert parsed["schema_version"] == "1.0"
    assert parsed["crawl_context"]["crawl_timestamp"] == "2024-01-01T00:00:00+00:00"
    assert parsed["documents"][0]["period_end"] == "2024-12-31T00:00:00+00:00"
    assert parsed["documents"][0]["date_reported"] == "2024-01-15T00:00:00+00:00"
    assert len(parsed["documents"][0]["link_trace"]["navigation_steps"]) == 4


def test_empty_company_capture(company, crawl_context):
    """Test CompanyCapture with no documents and no errors."""
    # ACT
    sut = CompanyCapture(
        company=company,
        crawl_context=crawl_context
    )

    # ASSERT
    assert sut.documents == []
    assert sut.errors == []
    assert sut.schema_version == "1.0"

    result = sut.to_dict()
    json_str = json.dumps(result)  # must not raise TypeError
    parsed = json.loads(json_str)
    assert parsed["documents"] == []
    assert parsed["errors"] == []


def test_minimal_financial_document_serializes(company, crawl_context):
    """Test that a FinancialDocument with only required fields serializes correctly."""
    # ARRANGE
    doc = FinancialDocument(
        company_identifier="TEST-001",
        document_type="annual_report",
        document_url="https://example.com/report.pdf"
    )
    sut = CompanyCapture(
        company=company,
        crawl_context=crawl_context,
        documents=[doc]
    )

    # ACT
    result = sut.to_dict()
    json_str = json.dumps(result)  # must not raise TypeError

    # ASSERT
    parsed = json.loads(json_str)
    parsed_doc = parsed["documents"][0]
    assert parsed_doc["company_identifier"] == "TEST-001"
    assert parsed_doc["document_type"] == "annual_report"
    assert parsed_doc["document_url"] == "https://example.com/report.pdf"
    assert parsed_doc["period_end"] is None
    assert parsed_doc["fiscal_period"] is None
    assert parsed_doc["link_trace"] is None


def test_utc_datetimes_serialize_with_offset(company):
    """Test that UTC-aware datetimes include the +00:00 offset in serialized output."""
    # ARRANGE
    context = CrawlContext(
        crawl_timestamp=datetime(2025, 6, 15, 14, 30, 0, tzinfo=timezone.utc),
        invocation_id="test-utc-001",
        model_used="test-model"
    )
    doc = FinancialDocument(
        company_identifier="TEST-001",
        document_type="annual_report",
        document_url="https://example.com/report.pdf",
        link_retrieval_datetime_utc=datetime(2025, 6, 15, 14, 35, 0, tzinfo=timezone.utc),
        document_download_datetime_utc=datetime(2025, 6, 15, 14, 36, 0, tzinfo=timezone.utc),
    )
    sut = CompanyCapture(
        company=company,
        crawl_context=context,
        documents=[doc]
    )

    # ACT
    result = sut.to_dict()
    json_str = json.dumps(result)
    parsed = json.loads(json_str)

    # ASSERT - all timestamps must end with +00:00
    assert parsed["crawl_context"]["crawl_timestamp"].endswith("+00:00")
    assert parsed["documents"][0]["link_retrieval_datetime_utc"].endswith("+00:00")
    assert parsed["documents"][0]["document_download_datetime_utc"].endswith("+00:00")
