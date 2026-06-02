#!/usr/bin/env python3
"""Regenerate web/static/models.json — the model list + pricing the
MASEMiner / MetaPaperLens frontend loads at startup.

Run weekly by .github/workflows/update-models.yml.  Two data sources:

  1. PRICING — the LiteLLM community dataset
     (model_prices_and_context_window.json on GitHub).  No API key
     needed.  Gives USD-per-token input/output rates + supports_vision.

  2. MODEL LISTS — each provider's own /models endpoint, sorted by
     release date to pick the newest 5 chat/vision models.  Needs the
     provider's API key as an env var (OPENAI_API_KEY, GEMINI_API_KEY,
     DEEPSEEK_API_KEY, MISTRAL_API_KEY).  Any provider whose key is
     missing — or whose API call fails — falls back to that provider's
     entry in the EXISTING models.json, so a flaky API never wipes a
     provider from the dropdown.

Design rule: NEVER emit a config that breaks the app.  On any doubt we
keep the previous models.json content for that provider.  Pricing that
can't be found just isn't listed — the frontend hides the cost estimate
for a model with no rate rather than showing a wrong number.

Usage:
    python update_models.py [path/to/models.json]
        (defaults to web/static/models.json relative to repo root)
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path
from typing import Any

LITELLM_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/"
    "model_prices_and_context_window.json"
)

# Our hosted providers → the ``litellm_provider`` value LiteLLM uses.
_LITELLM_PROVIDER_MAP = {
    "openai":    "openai",
    "gemini":    "google",
    "deepseek":  "deepseek",
    "mistral":   "mistral",
    "anthropic": "anthropic",
}
PROVIDERS = ["openai", "google", "anthropic", "deepseek", "mistral"]
NEWEST_N = 5

# Substrings that mark a model id as NOT a chat/vision extraction model
# (embeddings, audio, image-gen, moderation, etc.).  Used to filter the
# provider /models lists down to things MASEMiner can actually use.
_NON_CHAT_MARKERS = (
    "embed", "tts", "whisper", "audio", "realtime", "moderation",
    "image", "dall-e", "transcribe", "search", "rerank", "guard",
    "vision-only", "codestral-embed",
)


# ── HTTP ──────────────────────────────────────────────────────────────────────

def _get_json(url: str, headers: dict[str, str] | None = None, timeout: int = 30) -> Any:
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


# ── Pure helpers (unit-tested) ─────────────────────────────────────────────────

def _clean_model_id(name: str) -> str:
    """Strip a leading ``provider/`` prefix LiteLLM sometimes uses
    (``gemini/gemini-2.5-pro`` → ``gemini-2.5-pro``)."""
    return name.split("/", 1)[1] if "/" in name else name


def _our_provider(litellm_provider: str | None) -> str | None:
    return _LITELLM_PROVIDER_MAP.get(litellm_provider or "")


def _is_chat_model(model_id: str) -> bool:
    mid = model_id.lower()
    return not any(marker in mid for marker in _NON_CHAT_MARKERS)


def _prettify(model_id: str) -> str:
    """Best-effort human label from a model id.  ``gpt-4o`` → ``GPT-4o``;
    ``gemini-2.5-pro`` → ``Gemini 2.5 Pro``; ``deepseek-chat`` →
    ``DeepSeek Chat``.  Not perfect — the weekly diff is reviewed."""
    special = {"gpt": "GPT", "tas": "TAS", "deepseek": "DeepSeek"}
    parts = model_id.replace("_", "-").split("-")
    out: list[str] = []
    for p in parts:
        low = p.lower()
        if low in special:
            tok = special[low]
        elif any(c.isdigit() for c in p):
            tok = p                    # keep version tokens verbatim (4o, 2.5, v3)
        else:
            tok = p.capitalize()
        # Glue a version token directly onto a preceding ALL-CAPS acronym
        # with a hyphen for brand fidelity: "GPT" + "4o" → "GPT-4o".
        if out and out[-1].isupper() and any(c.isdigit() for c in p):
            out[-1] = f"{out[-1]}-{tok}"
        else:
            out.append(tok)
    return " ".join(out)


def build_rates(litellm_data: dict) -> dict[str, dict[str, float]]:
    """USD-per-MILLION-token input/output rates for every model belonging
    to one of our providers that has both costs listed."""
    rates: dict[str, dict[str, float]] = {}
    for name, info in litellm_data.items():
        if not isinstance(info, dict):
            continue
        if _our_provider(info.get("litellm_provider")) is None:
            continue
        cin  = info.get("input_cost_per_token")
        cout = info.get("output_cost_per_token")
        if not isinstance(cin, (int, float)) or not isinstance(cout, (int, float)):
            continue
        rates[_clean_model_id(name)] = {
            "in":  round(cin  * 1_000_000, 4),
            "out": round(cout * 1_000_000, 4),
        }
    return rates


def _litellm_entry(litellm_data: dict, provider: str, model_id: str) -> dict:
    """Look up a model's LiteLLM metadata, trying both the bare id and a
    couple of common provider-prefixed forms."""
    for key in (model_id, f"{provider}/{model_id}", f"gemini/{model_id}",
                f"mistral/{model_id}", f"deepseek/{model_id}", f"anthropic/{model_id}"):
        e = litellm_data.get(key)
        if isinstance(e, dict):
            return e
    return {}


def newest_for_provider(
    provider: str,
    api_models: list[dict] | None,
    litellm_data: dict,
    fallback_list: list[dict],
) -> list[dict]:
    """Pick the newest ``NEWEST_N`` chat/vision models for one provider.

    ``api_models`` is the provider /models response, normalised to a list
    of ``{"id": str, "created": int}``.  When it's None/empty (no key,
    API error) we return the existing models.json list for this provider
    so the dropdown is never emptied by a transient failure."""
    if not api_models:
        return fallback_list
    candidates = [m for m in api_models if m.get("id") and _is_chat_model(m["id"])]
    if not candidates:
        return fallback_list
    candidates.sort(key=lambda m: m.get("created", 0), reverse=True)
    out: list[dict] = []
    seen: set[str] = set()
    for m in candidates:
        mid = m["id"]
        if mid in seen:
            continue
        seen.add(mid)
        info = _litellm_entry(litellm_data, provider, mid)
        out.append({
            "value": mid,
            "label": _prettify(mid),
            "vision": bool(info.get("supports_vision", True)),
        })
        if len(out) >= NEWEST_N:
            break
    return out or fallback_list


def build_config(
    litellm_data: dict,
    api_models_by_provider: dict[str, list[dict] | None],
    fallback: dict,
) -> dict:
    """Assemble the full models.json payload.  Pure — no I/O — so it's
    unit-testable with fixtures."""
    fb_providers = (fallback or {}).get("providers", {}) or {}
    providers: dict[str, list[dict]] = {}
    for prov in PROVIDERS:
        providers[prov] = newest_for_provider(
            prov,
            api_models_by_provider.get(prov),
            litellm_data,
            fb_providers.get(prov, []),
        )
    rates = build_rates(litellm_data)
    # Union in any fallback rates for models we still list but LiteLLM
    # didn't price — keeps a working estimate for known models.
    for model_id, r in (fallback or {}).get("rates", {}).items():
        rates.setdefault(model_id, r)
    return {
        "_comment": (
            "Auto-generated weekly by .github/workflows/update-models.yml "
            "(update_models.py). Manual edits are overwritten on the next "
            "sync. The frontend falls back to baked-in defaults if this file "
            "is missing or malformed."
        ),
        "source": "LiteLLM pricing + provider /models APIs",
        "providers": providers,
        "rates": rates,
    }


# ── Provider /models fetchers (network; return None on any failure) ─────────────

def _fetch_openai() -> list[dict] | None:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        return None
    try:
        data = _get_json("https://api.openai.com/v1/models",
                         {"Authorization": f"Bearer {key}"})
        return [{"id": m["id"], "created": m.get("created", 0)}
                for m in data.get("data", []) if m.get("id", "").startswith(("gpt-", "o"))]
    except Exception as e:  # noqa: BLE001
        print(f"[update_models] OpenAI /models failed: {e}", file=sys.stderr)
        return None


def _fetch_mistral() -> list[dict] | None:
    key = os.environ.get("MISTRAL_API_KEY")
    if not key:
        return None
    try:
        data = _get_json("https://api.mistral.ai/v1/models",
                         {"Authorization": f"Bearer {key}"})
        return [{"id": m["id"], "created": m.get("created", 0)}
                for m in data.get("data", []) if m.get("id")]
    except Exception as e:  # noqa: BLE001
        print(f"[update_models] Mistral /models failed: {e}", file=sys.stderr)
        return None


def _fetch_anthropic() -> list[dict] | None:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    try:
        data = _get_json(
            "https://api.anthropic.com/v1/models?limit=100",
            {"x-api-key": key, "anthropic-version": "2023-06-01"},
        )
        out = []
        for m in data.get("data", []):
            mid = m.get("id")
            if not mid:
                continue
            # Anthropic returns an ISO ``created_at``; convert to epoch for
            # the shared "newest by created" sort.
            ts = 0
            created = m.get("created_at")
            if isinstance(created, str):
                try:
                    from datetime import datetime
                    ts = int(datetime.fromisoformat(created.replace("Z", "+00:00")).timestamp())
                except ValueError:
                    ts = 0
            out.append({"id": mid, "created": ts})
        return out
    except Exception as e:  # noqa: BLE001
        print(f"[update_models] Anthropic /models failed: {e}", file=sys.stderr)
        return None


def _fetch_deepseek() -> list[dict] | None:
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        return None
    try:
        data = _get_json("https://api.deepseek.com/models",
                         {"Authorization": f"Bearer {key}"})
        # DeepSeek doesn't return created timestamps; order is stable.
        return [{"id": m["id"], "created": 0} for m in data.get("data", []) if m.get("id")]
    except Exception as e:  # noqa: BLE001
        print(f"[update_models] DeepSeek /models failed: {e}", file=sys.stderr)
        return None


def _fetch_gemini() -> list[dict] | None:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        return None
    try:
        data = _get_json(
            f"https://generativelanguage.googleapis.com/v1beta/models?key={key}&pageSize=200"
        )
        out = []
        for m in data.get("models", []):
            name = (m.get("name") or "").replace("models/", "")
            methods = m.get("supportedGenerationMethods", [])
            if name.startswith("gemini-") and "generateContent" in methods:
                # No created date from Gemini; approximate recency by the
                # version number embedded in the id (2.5 > 2.0 > 1.5).
                out.append({"id": name, "created": _gemini_recency(name)})
        return out
    except Exception as e:  # noqa: BLE001
        print(f"[update_models] Gemini /models failed: {e}", file=sys.stderr)
        return None


def _gemini_recency(model_id: str) -> int:
    """Crude recency proxy for Gemini ids that lack a created date:
    parse the major.minor version (gemini-2.5-pro → 250)."""
    import re
    m = re.search(r"gemini-(\d+)\.(\d+)", model_id)
    if not m:
        return 0
    return int(m.group(1)) * 100 + int(m.group(2))


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else repo_root / "web" / "static" / "models.json"

    try:
        fallback = json.loads(out_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        fallback = {}

    try:
        litellm_data = _get_json(LITELLM_URL)
    except Exception as e:  # noqa: BLE001
        print(f"[update_models] LiteLLM fetch failed ({e}) — keeping existing models.json", file=sys.stderr)
        return 0  # don't touch the file; existing config stays in place

    api_models = {
        "openai":    _fetch_openai(),
        "google":    _fetch_gemini(),
        "anthropic": _fetch_anthropic(),
        "deepseek":  _fetch_deepseek(),
        "mistral":   _fetch_mistral(),
    }
    config = build_config(litellm_data, api_models, fallback)

    # Safety net: never write a config with an empty provider list.
    for prov in PROVIDERS:
        if not config["providers"].get(prov):
            config["providers"][prov] = (fallback.get("providers", {}) or {}).get(prov, [])

    from datetime import datetime, timezone
    config["generated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    out_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"[update_models] wrote {out_path} "
          f"({sum(len(v) for v in config['providers'].values())} models, "
          f"{len(config['rates'])} rates)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
