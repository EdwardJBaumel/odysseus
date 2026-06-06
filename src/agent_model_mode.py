"""Determine whether an agent round should use native tool schemas."""

from __future__ import annotations

import logging
from typing import List, Optional
from urllib.parse import urlparse

from src.llm_core import _is_ollama_native_url

logger = logging.getLogger(__name__)

_API_HOSTS = frozenset([
    "api.openai.com", "api.anthropic.com",
    "openrouter.ai", "api.groq.com",
    "api.mistral.ai", "api.cohere.com",
    "api.deepseek.com", "deepseek.com",
    "api.together.xyz", "api.fireworks.ai",
    "api.perplexity.ai", "api.x.ai",
    "ollama.com", "api.venice.ai",
    "api.githubcopilot.com",
    "localhost", "127.0.0.1", "host.docker.internal",
])

_MODEL_SUPPORTS_TOOLS = (
    "gpt-4", "gpt-5", "gpt-o", "claude", "gemini", "gemma",
    "qwen3", "qwen2.5", "mixtral", "mistral", "llama-3.1", "llama-3.2",
    "llama-3.3", "llama-4",
    "minimax", "kimi", "yi-", "phi-3", "phi-4", "command-r",
    "glm-4", "internlm", "hermes",
    "deepseek-v", "deepseek-chat",
)

_MODEL_NO_TOOLS = ("deepseek-r1",)


def _is_ollama_openai_compat_url(endpoint_url: str) -> bool:
    try:
        parsed = urlparse(endpoint_url or "")
    except Exception:
        return False
    path = (parsed.path or "").rstrip("/")
    return parsed.port == 11434 and (path == "/v1" or path.startswith("/v1/"))


def _endpoint_lookup_keys(endpoint_url: str) -> List[str]:
    raw = (endpoint_url or "").strip()
    keys: List[str] = []

    def add(value: str):
        value = (value or "").strip()
        if value and value not in keys:
            keys.append(value)
        trimmed = value.rstrip("/")
        if trimmed and trimmed not in keys:
            keys.append(trimmed)
        if trimmed and f"{trimmed}/" not in keys:
            keys.append(f"{trimmed}/")

    add(raw)
    try:
        parsed = urlparse(raw)
        if parsed.scheme and parsed.netloc:
            add(f"{parsed.scheme}://{parsed.netloc}")
            add(f"{parsed.scheme}://{parsed.netloc}/")
    except Exception:
        pass
    return keys


def compute_is_api_model(endpoint_url: str, model: str) -> bool:
    """Return True when native OpenAI-style tool schemas should be sent."""
    endpoint_url = endpoint_url or ""
    model_lc = (model or "").lower()
    endpoint_supports: Optional[bool] = None
    try:
        from core.database import ModelEndpoint, SessionLocal

        db = SessionLocal()
        try:
            ep = None
            for key in _endpoint_lookup_keys(endpoint_url):
                ep = db.query(ModelEndpoint).filter(ModelEndpoint.base_url == key).first()
                if ep is not None:
                    break
            if ep is not None:
                endpoint_supports = ep.supports_tools
        finally:
            db.close()
    except Exception as exc:
        logger.debug("endpoint supports_tools lookup failed: %s", exc)

    model_supports_tools = any(kw in model_lc for kw in _MODEL_SUPPORTS_TOOLS)
    model_no_tools = any(kw in model_lc for kw in _MODEL_NO_TOOLS)
    is_ollama_native = _is_ollama_native_url(endpoint_url)
    ollama_openai_compat = _is_ollama_openai_compat_url(endpoint_url)

    if endpoint_supports is True:
        return True
    if endpoint_supports is False or model_no_tools or is_ollama_native or ollama_openai_compat:
        return False
    return any(host in endpoint_url for host in _API_HOSTS) or model_supports_tools
