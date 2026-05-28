"""Tests for the weekly model-sync script's pure config builder.

The script lives in .github/scripts/update_models.py (outside the web/
package), so we load it by path.  We only test the network-free core —
``build_config`` and its helpers — with fixtures; the HTTP fetchers are
exercised in CI against the live APIs.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / ".github" / "scripts" / "update_models.py"


@pytest.fixture(scope="module")
def um():
    spec = importlib.util.spec_from_file_location("update_models", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def litellm_data():
    # Minimal slice of the LiteLLM dataset shape.
    return {
        "gpt-5.5":          {"litellm_provider": "openai", "input_cost_per_token": 5e-6,  "output_cost_per_token": 30e-6, "supports_vision": True},
        "gpt-5":            {"litellm_provider": "openai", "input_cost_per_token": 1.25e-6, "output_cost_per_token": 10e-6, "supports_vision": True},
        "gpt-4o":           {"litellm_provider": "openai", "input_cost_per_token": 2.5e-6, "output_cost_per_token": 10e-6, "supports_vision": True},
        "text-embedding-3-large": {"litellm_provider": "openai", "input_cost_per_token": 1.3e-7, "output_cost_per_token": 0.0},
        "gemini/gemini-2.5-pro": {"litellm_provider": "gemini", "input_cost_per_token": 1.25e-6, "output_cost_per_token": 5e-6, "supports_vision": True},
        "deepseek-chat":    {"litellm_provider": "deepseek", "input_cost_per_token": 2.7e-7, "output_cost_per_token": 1.1e-6},
        # Anthropic is a supported provider — its rates are included.
        "claude-haiku-4-5": {"litellm_provider": "anthropic", "input_cost_per_token": 1e-6, "output_cost_per_token": 5e-6},
        # A provider we DON'T surface (Cohere) — must be ignored.
        "command-r":        {"litellm_provider": "cohere", "input_cost_per_token": 1e-6, "output_cost_per_token": 5e-6},
    }


@pytest.fixture
def fallback():
    return {
        "providers": {
            "openai":   [{"value": "gpt-4o", "label": "GPT-4o", "vision": True}],
            "google":   [{"value": "gemini-2.5-pro", "label": "Gemini 2.5 Pro", "vision": True}],
            "deepseek": [{"value": "deepseek-chat", "label": "DeepSeek Chat", "vision": False}],
            "mistral":  [{"value": "mistral-large-latest", "label": "Mistral Large", "vision": False}],
        },
        "rates": {"gpt-4o-mini": {"in": 0.15, "out": 0.60}},
    }


def test_build_rates_converts_per_token_to_per_million(um, litellm_data):
    rates = um.build_rates(litellm_data)
    assert rates["gpt-5.5"] == {"in": 5.0, "out": 30.0}
    assert rates["gpt-5"]   == {"in": 1.25, "out": 10.0}
    # Provider-prefixed key is cleaned to the bare id.
    assert "gemini-2.5-pro" in rates
    # Anthropic is supported now — included.
    assert rates["claude-haiku-4-5"] == {"in": 1.0, "out": 5.0}
    # An unsupported provider (Cohere) is excluded.
    assert "command-r" not in rates


def test_newest_picks_five_by_created_desc(um, litellm_data, fallback):
    api = [
        {"id": "gpt-4o",     "created": 100},
        {"id": "gpt-5",      "created": 300},
        {"id": "gpt-5.5",    "created": 400},
        {"id": "gpt-4-turbo","created": 50},
        {"id": "gpt-3.5",    "created": 10},
        {"id": "gpt-9",      "created": 500},
        {"id": "text-embedding-3-large", "created": 999},  # filtered out (non-chat)
    ]
    out = um.newest_for_provider("openai", api, litellm_data, fallback["providers"]["openai"])
    ids = [m["value"] for m in out]
    assert ids == ["gpt-9", "gpt-5.5", "gpt-5", "gpt-4o", "gpt-4-turbo"]  # 5 newest, embed excluded
    assert "text-embedding-3-large" not in ids


def test_newest_falls_back_when_api_missing(um, litellm_data, fallback):
    out = um.newest_for_provider("openai", None, litellm_data, fallback["providers"]["openai"])
    assert out == fallback["providers"]["openai"]
    out2 = um.newest_for_provider("openai", [], litellm_data, fallback["providers"]["openai"])
    assert out2 == fallback["providers"]["openai"]


def test_build_config_keeps_provider_on_missing_api(um, litellm_data, fallback):
    # Only OpenAI has a live list; the rest must keep their fallback entries.
    api = {"openai": [{"id": "gpt-5.5", "created": 400}], "google": None,
           "deepseek": None, "mistral": None}
    cfg = um.build_config(litellm_data, api, fallback)
    assert cfg["providers"]["openai"][0]["value"] == "gpt-5.5"
    assert cfg["providers"]["google"]   == fallback["providers"]["google"]
    assert cfg["providers"]["deepseek"] == fallback["providers"]["deepseek"]
    assert cfg["providers"]["mistral"]  == fallback["providers"]["mistral"]
    # Rates include LiteLLM-derived + unioned fallback-only rates.
    assert cfg["rates"]["gpt-5.5"] == {"in": 5.0, "out": 30.0}
    assert cfg["rates"]["gpt-4o-mini"] == {"in": 0.15, "out": 0.60}  # from fallback


def test_prettify(um):
    assert um._prettify("gpt-4o") == "GPT-4o"
    assert um._prettify("gemini-2.5-pro") == "Gemini 2.5 Pro"
    assert um._prettify("deepseek-chat") == "DeepSeek Chat"


def test_is_chat_model_filters_non_chat(um):
    assert um._is_chat_model("gpt-5.5") is True
    assert um._is_chat_model("text-embedding-3-large") is False
    assert um._is_chat_model("whisper-1") is False
    assert um._is_chat_model("dall-e-3") is False
