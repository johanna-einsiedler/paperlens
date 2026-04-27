"""Tests for providers.py — routing, client construction, token-usage parsing."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import providers


# ── get_provider ─────────────────────────────────────────────────────────────

def test_get_provider_openai_default():
    assert providers.get_provider("gpt-4o") == "openai"
    assert providers.get_provider("gpt-4o-mini") == "openai"


def test_get_provider_gemini():
    assert providers.get_provider("gemini-2.5-flash") == "google"


def test_get_provider_deepseek():
    assert providers.get_provider("deepseek-chat") == "deepseek"


def test_get_provider_vllm_when_base_url_set():
    # Any model name + base_url → vllm
    assert providers.get_provider("gpt-4o", base_url="http://localhost:8000") == "vllm"
    assert providers.get_provider("llama3.2:3b", base_url="http://localhost:11434") == "vllm"


# ── _openai_compat_client ────────────────────────────────────────────────────

def test_openai_compat_client_appends_v1():
    client = providers._openai_compat_client("any-key", "http://localhost:8000")
    assert str(client.base_url).rstrip("/").endswith("/v1")


def test_openai_compat_client_does_not_double_append_v1():
    client = providers._openai_compat_client("any-key", "http://localhost:8000/v1")
    base = str(client.base_url).rstrip("/")
    # Should end in "/v1" exactly once
    assert base.endswith("/v1")
    assert not base.endswith("/v1/v1")


def test_openai_compat_client_uses_dummy_key_when_blank():
    # Should not raise on empty key
    client = providers._openai_compat_client("", "http://localhost:8000")
    assert client is not None


def test_openai_compat_client_no_base_url_uses_default():
    client = providers._openai_compat_client("any-key")
    # Default OpenAI base URL
    assert "openai.com" in str(client.base_url)


# ── _openai_usage / _gemini_usage ────────────────────────────────────────────

def test_openai_usage_extracts_token_counts():
    fake_response = SimpleNamespace(usage=SimpleNamespace(
        prompt_tokens=100, completion_tokens=50, total_tokens=150
    ))
    assert providers._openai_usage(fake_response) == {
        "prompt": 100, "completion": 50, "total": 150
    }


def test_openai_usage_handles_missing_usage():
    fake_response = SimpleNamespace(usage=None)
    assert providers._openai_usage(fake_response) == {
        "prompt": 0, "completion": 0, "total": 0
    }


def test_gemini_usage_extracts_token_counts():
    fake_response = SimpleNamespace(usage_metadata=SimpleNamespace(
        prompt_token_count=200, candidates_token_count=80, total_token_count=280
    ))
    assert providers._gemini_usage(fake_response) == {
        "prompt": 200, "completion": 80, "total": 280
    }


def test_gemini_usage_handles_missing_metadata():
    fake_response = SimpleNamespace()  # no usage_metadata at all
    assert providers._gemini_usage(fake_response) == {
        "prompt": 0, "completion": 0, "total": 0
    }


# ── generate_text routing ────────────────────────────────────────────────────

def _mock_chat_response(text="ok", finish="stop", prompt=10, completion=20):
    """Build a fake OpenAI chat completion response."""
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=text),
            finish_reason=finish,
        )],
        usage=SimpleNamespace(
            prompt_tokens=prompt, completion_tokens=completion,
            total_tokens=prompt + completion,
        ),
    )


def test_generate_text_openai_path():
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = _mock_chat_response(text="hi")
    with patch.object(providers.openai, "OpenAI", return_value=fake_client):
        out = providers.generate_text("gpt-4o-mini", "key", "prompt")
    assert out == "hi"
    fake_client.chat.completions.create.assert_called_once()
    call_kwargs = fake_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "gpt-4o-mini"


def test_generate_text_vllm_uses_custom_base_url():
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = _mock_chat_response(text="vllm-out")
    with patch.object(providers, "_openai_compat_client", return_value=fake_client) as mk:
        out = providers.generate_text(
            "llama3.2:3b", "any-key", "prompt", base_url="http://localhost:11434"
        )
    assert out == "vllm-out"
    mk.assert_called_once_with("any-key", "http://localhost:11434")


def test_generate_text_deepseek_path():
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = _mock_chat_response(text="ds")
    with patch.object(providers.openai, "OpenAI", return_value=fake_client) as mk:
        out = providers.generate_text("deepseek-chat", "key", "prompt")
    assert out == "ds"
    # Should have been constructed with the DeepSeek base URL
    call_kwargs = mk.call_args.kwargs
    assert "deepseek.com" in call_kwargs.get("base_url", "")


# ── extract_with_images ──────────────────────────────────────────────────────

def test_extract_with_images_openai_returns_token_usage():
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = _mock_chat_response(
        text='{"a":1}', prompt=500, completion=200
    )
    with patch.object(providers, "_openai_compat_client", return_value=fake_client):
        result, finish, usage = providers.extract_with_images(
            model="gpt-4o",
            api_key="key",
            content_blocks=[{"type": "text", "text": "go"}],
            extraction_images=[],
            prompt="p",
            page_instruction="",
            n=0,
        )
    assert result == '{"a":1}'
    assert finish == "stop"
    assert usage == {"prompt": 500, "completion": 200, "total": 700}


# ── extract_with_text routing ────────────────────────────────────────────────

def test_extract_with_text_openai_path():
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = _mock_chat_response(
        text='{"ok":true}'
    )
    with patch.object(providers, "_openai_compat_client", return_value=fake_client):
        result, finish, usage = providers.extract_with_text(
            "gpt-4o", "key", "page text", "prompt", "instr"
        )
    assert result == '{"ok":true}'
    assert usage["total"] == 30  # 10 prompt + 20 completion


def test_extract_with_text_deepseek_short_doc_no_chunking():
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = _mock_chat_response(
        text='[{"x":1}]'
    )
    with patch.object(providers.openai, "OpenAI", return_value=fake_client):
        result, finish, usage = providers.extract_with_text(
            "deepseek-chat", "key", "short text", "prompt", "instr"
        )
    assert result == '[{"x":1}]'
    # Single call, no chunking
    assert fake_client.chat.completions.create.call_count == 1


def test_extract_provider_message_from_openai_body():
    """OpenAI exceptions have .body with the parsed error JSON — pull the
    inner message out, not the noisy wrapper."""
    # Fake exception with body in OpenAI's standard shape
    exc = SimpleNamespace(
        body={"error": {"message": "Incorrect API key provided: sk-bogus.",
                        "type": "invalid_request_error",
                        "code": "invalid_api_key"}},
        message="Error code: 401 - {'error': {...}}",
    )
    msg = providers.extract_provider_message(exc)
    assert msg == "Incorrect API key provided: sk-bogus."


def test_extract_provider_message_from_string_body_error():
    """Some APIs put a bare error string in body['error']."""
    exc = SimpleNamespace(body={"error": "Token quota exceeded"})
    assert providers.extract_provider_message(exc) == "Token quota exceeded"


def test_extract_provider_message_falls_back_to_message_attr():
    """Gemini-style: no body, but a .message attribute carries the text."""
    class FakeGeminiError(Exception):
        def __init__(self):
            super().__init__("API error")
            self.message = "Resource exhausted: please try again later"

    msg = providers.extract_provider_message(FakeGeminiError())
    assert msg == "Resource exhausted: please try again later"


def test_extract_provider_message_falls_back_to_str():
    """Last resort: stringify the exception."""
    msg = providers.extract_provider_message(ValueError("plain error"))
    assert msg == "plain error"


def test_extract_provider_message_handles_no_body_no_message():
    """Empty exceptions don't crash."""
    exc = SimpleNamespace()
    # No body, no message; str(exc) is the namespace repr — but it shouldn't raise
    assert isinstance(providers.extract_provider_message(exc), str)


def test_split_markdown_pages_separates_sections():
    md = (
        "--- PDF page 1 of 3 ---\nfirst\n\n"
        "--- PDF page 2 of 3 ---\nsecond\n\n"
        "--- PDF page 3 of 3 ---\nthird\n"
    )
    parts = providers._split_markdown_pages(md)
    # re.split with a lookahead can leave a leading empty string; filter those out
    parts = [p for p in parts if p.strip()]
    assert len(parts) == 3
    assert "first" in parts[0]
    assert "second" in parts[1]
    assert "third" in parts[2]
