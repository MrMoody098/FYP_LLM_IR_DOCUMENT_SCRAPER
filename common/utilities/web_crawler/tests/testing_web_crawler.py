"""
Spike tests — prints results for manual verification.

Bulk runs (concurrent, all companies):
    pytest common/utilities/web_crawler/tests/testing_web_crawler.py -v -s -k "bulk"

Single-company debugging:
    pytest common/utilities/web_crawler/tests/testing_web_crawler.py -v -s -k "AMD"
    pytest common/utilities/web_crawler/tests/testing_web_crawler.py -v -s -k "Apple or Microsoft"
"""

import asyncio
import json
from pathlib import Path

import pytest

from common.utilities.llm_client.open_ai_client import OpenAIClient
from common.utilities.web_crawler.company_crawler import (
    FORM_10K,
    FORM_10Q_Q1_2025,
    FORM_10Q_RECENT,
    CompanyDocumentCrawler,
)
from common.utilities.web_crawler.crawler_data import Company
from common.utilities.web_crawler.document_query import DocumentQuery
from common.utilities.web_crawler.http_fetcher import CffiHttpFetcher
from common.utilities.web_crawler.load_companies import load_companies_from_csv

CSV_PATH = "common/utilities/web_crawler/US_companies.csv"
RESULTS_DIR = Path("/home/danielmoody/projects/data-acquisition")
CONCURRENCY = 5

_companies = load_companies_from_csv(CSV_PATH)
COMPANIES = [pytest.param(c, id=c.company_name) for c in _companies]


@pytest.fixture(scope="session")
def llm_client():
    return OpenAIClient()


def _link_trace_urls(result) -> list[str]:
    if not result.link_trace:
        return []
    return [s.url for s in result.link_trace.navigation_steps]


async def _run_all(
    companies: list[Company],
    query_template: DocumentQuery,
    results_file: Path,
    llm_client,
    fetcher_factory=None,
) -> list[dict]:
    """Run all companies concurrently, capped at CONCURRENCY simultaneous crawls."""
    semaphore = asyncio.Semaphore(CONCURRENCY)
    write_lock = asyncio.Lock()

    success_file = results_file.with_name(results_file.stem + "_successes.json")
    failure_file = results_file.with_name(results_file.stem + "_failures.json")

    async def _crawl_one(company: Company) -> dict:
        async with semaphore:
            fetcher = fetcher_factory() if fetcher_factory else None
            async with CompanyDocumentCrawler(llm_client, fetcher=fetcher) as crawler:
                result = await crawler.find(company, query_template)
        row = {
            "company": company.company_name,
            "root_url": company.root_url,
            "document_type": result.document_type,
            "status": "success" if result.document_url else "failed",
            "used_subdomain_fallback": result.used_subdomain_fallback,
            "link_trace": _link_trace_urls(result),
            "doc_url": result.document_url,
        }
        print(f"\n[{company.company_name}] {row['status']}: {result.document_url}")
        async with write_lock:
            # Write to main results file
            existing = json.loads(results_file.read_text()) if results_file.exists() else []
            existing.append(row)
            results_file.write_text(json.dumps(existing, indent=2))

            # Write to separate success/failure file
            target = success_file if row["status"] == "success" else failure_file
            target_existing = json.loads(target.read_text()) if target.exists() else []
            target_existing.append(row)
            target.write_text(json.dumps(target_existing, indent=2))
        return row

    rows = await asyncio.gather(*[_crawl_one(c) for c in companies])

    successes = sum(1 for r in rows if r["status"] == "success")
    failures = len(rows) - successes
    summary = {"total": len(rows), "successes": successes, "failures": failures}

    # Append summary to the main results file
    data = json.loads(results_file.read_text()) if results_file.exists() else []
    data.append({"summary": summary})
    results_file.write_text(json.dumps(data, indent=2))

    return rows


# ---------------------------------------------------------------------------
# Bulk concurrent runs — retrieve all companies at once


@pytest.mark.asyncio
async def test_bulk_10k(llm_client):
    results_file = RESULTS_DIR / "discovery_results_10k.json"
    for f in [results_file,
              results_file.with_name("discovery_results_10k_successes.json"),
              results_file.with_name("discovery_results_10k_failures.json")]:
        f.unlink(missing_ok=True)
    rows = await _run_all(_companies, FORM_10K, results_file, llm_client)
    successes = sum(1 for r in rows if r["status"] == "success")
    print(f"\n=== 10-K: {successes}/{len(rows)} succeeded ===")
    assert successes > 0, "No 10-K documents found for any company"


@pytest.mark.asyncio
async def test_bulk_10q(llm_client):
    results_file = RESULTS_DIR / "discovery_results_10q.json"
    for f in [results_file,
              results_file.with_name("discovery_results_10q_successes.json"),
              results_file.with_name("discovery_results_10q_failures.json")]:
        f.unlink(missing_ok=True)
    rows = await _run_all(_companies, FORM_10Q_Q1_2025, results_file, llm_client)
    successes = sum(1 for r in rows if r["status"] == "success")
    print(f"\n=== 10-Q: {successes}/{len(rows)} succeeded ===")
    assert successes > 0, "No 10-Q documents found for any company"


@pytest.mark.asyncio
async def test_bulk_10q_recent(llm_client):
    companies_290 = load_companies_from_csv(CSV_PATH, limit=290)
    results_file = RESULTS_DIR / "discovery_results_10q_recent.json"
    for f in [results_file,
              results_file.with_name("discovery_results_10q_recent_successes.json"),
              results_file.with_name("discovery_results_10q_recent_failures.json")]:
        f.unlink(missing_ok=True)
    rows = await _run_all(companies_290, FORM_10Q_RECENT, results_file, llm_client)
    successes = sum(1 for r in rows if r["status"] == "success")
    print(f"\n=== 10-Q Recent: {successes}/{len(rows)} succeeded ===")
    assert successes > 0, "No recent 10-Q documents found for any company"


# ---------------------------------------------------------------------------
# Bulk concurrent runs with cffi fetcher — for comparison against Playwright


@pytest.mark.asyncio
async def test_cffi_bulk_10k(llm_client):
    results_file = RESULTS_DIR / "cffi_discovery_results_10k.json"
    for f in [results_file,
              results_file.with_name("cffi_discovery_results_10k_successes.json"),
              results_file.with_name("cffi_discovery_results_10k_failures.json")]:
        f.unlink(missing_ok=True)
    rows = await _run_all(_companies, FORM_10K, results_file, llm_client,
                          fetcher_factory=CffiHttpFetcher)
    successes = sum(1 for r in rows if r["status"] == "success")
    print(f"\n=== CFFI 10-K: {successes}/{len(rows)} succeeded ===")
    assert successes > 0, "No 10-K documents found for any company"


@pytest.mark.asyncio
async def test_cffi_bulk_10q_recent(llm_client):
    companies_290 = load_companies_from_csv(CSV_PATH, limit=290)
    results_file = RESULTS_DIR / "cffi_discovery_results_10q_recent.json"
    for f in [results_file,
              results_file.with_name("cffi_discovery_results_10q_recent_successes.json"),
              results_file.with_name("cffi_discovery_results_10q_recent_failures.json")]:
        f.unlink(missing_ok=True)
    rows = await _run_all(companies_290, FORM_10Q_RECENT, results_file, llm_client,
                          fetcher_factory=CffiHttpFetcher)
    successes = sum(1 for r in rows if r["status"] == "success")
    print(f"\n=== CFFI 10-Q Recent: {successes}/{len(rows)} succeeded ===")
    assert successes > 0, "No recent 10-Q documents found for any company"


# ---------------------------------------------------------------------------
# Single-company tests — useful for debugging individual companies


# @pytest.mark.asyncio
# @pytest.mark.parametrize("company", COMPANIES)
# async def test_single_10k(llm_client, company):
#     async with CompanyDocumentCrawler(llm_client) as crawler:
#         result = await crawler.find(company, FORM_10K)
#     print(f"\n[{company.company_name}] {'success' if result.document_url else 'failed'}: {result.document_url}")
#     assert result.document_url, f"No 10-K found for {company.company_name}"


# @pytest.mark.asyncio
# @pytest.mark.parametrize("company", COMPANIES)
# async def test_single_10q(llm_client, company):
#     async with CompanyDocumentCrawler(llm_client) as crawler:
#         result = await crawler.find(company, FORM_10Q_Q1_2025)
#     print(f"\n[{company.company_name}] {'success' if result.document_url else 'failed'}: {result.document_url}")
#     assert result.document_url, f"No 10-Q found for {company.company_name}"
