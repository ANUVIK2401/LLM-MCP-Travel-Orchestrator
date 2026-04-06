from types import SimpleNamespace

import rag


def setup_function():
    rag._summary_cache.clear()


def test_build_metadata_summary_uses_available_fields():
    summary = rag.build_metadata_summary(
        {
            "name": "Ocean View Villa",
            "price": "$240/night",
            "rating": "4.96",
            "desc": "Two-bedroom villa with plunge pool and sunset terrace.",
        }
    )

    assert "Ocean View Villa" in summary
    assert "$240/night" in summary
    assert "4.96/5" in summary


def test_summarize_listing_returns_metadata_when_api_key_missing():
    result = rag.summarize_listing(
        "https://www.airbnb.com/rooms/123456",
        "",
        fallback_metadata={
            "name": "Canal Loft",
            "price": "$145/night",
            "rating": "4.91",
            "desc": "Bright loft near the center with kitchen and balcony.",
        },
    )

    assert result.source == "metadata"
    assert "Canal Loft" in result.text


def test_summarize_listing_prefers_retrieved_context(monkeypatch):
    class FakeRetriever:
        def invoke(self, query):
            return [
                SimpleNamespace(
                    page_content=(
                        "One-bedroom loft near the center with balcony, kitchen, workspace, "
                        "and quick metro access."
                    )
                )
            ]

    class FakeIndex:
        def as_retriever(self, search_kwargs):
            return FakeRetriever()

    class FakeLLM:
        def invoke(self, prompt):
            return SimpleNamespace(
                content=(
                    "One-bedroom loft near the center with a balcony, full kitchen, workspace, "
                    "and fast metro access."
                )
            )

    monkeypatch.setattr(rag, "_get_or_build_index", lambda url, api_key: FakeIndex())
    monkeypatch.setattr(rag, "_get_summary_llm", lambda api_key: FakeLLM())

    result = rag.summarize_listing(
        "https://www.airbnb.com/rooms/777777",
        "test-key",
        fallback_metadata={"name": "Fallback Loft", "desc": "Fallback description"},
    )

    assert result.source == "retrieved"
    assert "balcony" in result.text.lower()


def test_summarize_listing_falls_back_when_retrieval_breaks(monkeypatch):
    monkeypatch.setattr(
        rag,
        "_get_or_build_index",
        lambda url, api_key: (_ for _ in ()).throw(RuntimeError("blocked")),
    )

    result = rag.summarize_listing(
        "https://www.airbnb.com/rooms/888888",
        "test-key",
        fallback_metadata={
            "name": "Forest Cabin",
            "price": "$180/night",
            "rating": "4.8",
            "desc": "Cozy cabin stay with fireplace and deck.",
        },
    )

    assert result.source == "metadata"
    assert "Forest Cabin" in result.text
