"""Unit tests for LLMNavigator."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from common.utilities.web_crawler.llm_navigator import LLMNavigator, NavigationDecision

PROMPT = "Target: {target_description}\nVisited: {visited}\nPage: {current_url}\n{content}"


@pytest.fixture
def fetcher():
    mock = MagicMock()
    mock.fetch = AsyncMock(return_value="<page content>")
    return mock


@pytest.fixture
def llm_client():
    mock = MagicMock()
    mock.chat_structured_async = AsyncMock()
    return mock


@pytest.fixture(autouse=True)
def patch_to_thread(monkeypatch):
    """This fixture is no longer needed since we use async methods directly."""
    pass


def make_navigator(fetcher, llm_client, max_depth=5):
    return LLMNavigator(fetcher, llm_client, PROMPT, max_depth=max_depth, label="test")


def found(url: str, confidence_message: str = "") -> NavigationDecision:
    return NavigationDecision(action="FOUND", url=url, confidence_message=confidence_message)


def next_(url: str, confidence_message: str = "") -> NavigationDecision:
    return NavigationDecision(action="NEXT", url=url, confidence_message=confidence_message)


def none_(confidence_message: str = "") -> NavigationDecision:
    return NavigationDecision(action="NONE", url=None, confidence_message=confidence_message)


class TestLLMNavigatorNavigate:
    async def test_found_on_first_page(self, fetcher, llm_client):
        llm_client.chat_structured_async.return_value = found("https://example.com/report.pdf")
        outcome = await make_navigator(fetcher, llm_client).navigate(
            "https://start.com", target_description="IR page"
        )
        assert outcome.url == "https://example.com/report.pdf"

    async def test_follows_next_then_finds_document(self, fetcher, llm_client):
        llm_client.chat_structured_async.side_effect = [
            next_("https://example.com/about"),
            found("https://example.com/report.pdf"),
        ]
        outcome = await make_navigator(fetcher, llm_client).navigate(
            "https://start.com", target_description="IR page"
        )
        assert outcome.url == "https://example.com/report.pdf"

    async def test_none_response_returns_none(self, fetcher, llm_client):
        llm_client.chat_structured_async.return_value = none_()
        outcome = await make_navigator(fetcher, llm_client).navigate(
            "https://start.com", target_description="IR page"
        )
        assert outcome.url is None

    async def test_fetch_failure_returns_none(self, fetcher, llm_client):
        fetcher.fetch.return_value = None
        outcome = await make_navigator(fetcher, llm_client).navigate(
            "https://start.com", target_description="IR page"
        )
        assert outcome.url is None
        llm_client.chat_structured_async.assert_not_called()

    async def test_loop_detection_stops_navigation(self, fetcher, llm_client):
        llm_client.chat_structured_async.return_value = next_("https://start.com")
        outcome = await make_navigator(fetcher, llm_client).navigate(
            "https://start.com", target_description="IR page"
        )
        assert outcome.url is None

    async def test_max_depth_reached_returns_none(self, fetcher, llm_client):
        llm_client.chat_structured_async.side_effect = [
            next_(f"https://example.com/step{i}") for i in range(10)
        ]
        outcome = await make_navigator(fetcher, llm_client, max_depth=3).navigate(
            "https://start.com", target_description="IR page"
        )
        assert outcome.url is None
        assert llm_client.chat_structured_async.call_count == 3
